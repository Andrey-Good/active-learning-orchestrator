"""Internal split resolution and validation helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence as SequenceABC
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..configs import SplitConfig
from ..dataset.provider import DatasetProvider
from ..exceptions import ConfigurationError, DatasetMismatchError


def validate_persisted_splits(splits: Any, reference_sample_ids: Sequence[str]) -> List[str]:
    issues: List[str] = []
    if not isinstance(splits, Mapping):
        return ["Persisted split membership is invalid: splits must be a mapping."]
    if not splits:
        return issues

    allowed_splits = {"train", "val", "test"}
    reference_ids = {str(sample_id) for sample_id in reference_sample_ids}
    owners: Dict[str, str] = {}
    duplicate_ids_seen = False
    unknown_ids_seen = False
    overlap_seen = False

    for split_name, raw_split_ids in splits.items():
        split_name_str = str(split_name)
        if split_name_str not in allowed_splits:
            issues.append(f"Persisted split membership contains unsupported split {split_name_str!r}.")
        if isinstance(raw_split_ids, (str, bytes)) or not isinstance(raw_split_ids, SequenceABC):
            issues.append(f"Persisted split {split_name_str!r} is invalid: expected a sequence of sample ids.")
            continue

        split_ids = [str(sample_id) for sample_id in raw_split_ids]
        duplicate_ids = sorted(sample_id for sample_id, count in Counter(split_ids).items() if count > 1)
        if duplicate_ids:
            duplicate_ids_seen = True
            issues.append(f"Persisted split {split_name_str!r} contains duplicate sample ids: {duplicate_ids[:10]}.")

        unknown_ids = sorted(sample_id for sample_id in set(split_ids) if sample_id not in reference_ids)
        if unknown_ids:
            unknown_ids_seen = True
            issues.append(f"Persisted split {split_name_str!r} contains unknown sample ids: {unknown_ids[:10]}.")

        overlap_seen = _record_split_owners(
            split_ids,
            split_name_str=split_name_str,
            owners=owners,
            issues=issues,
            overlap_seen=overlap_seen,
        )

    _append_split_coverage_issue(
        issues,
        reference_ids=reference_ids,
        owners=owners,
        duplicate_ids_seen=duplicate_ids_seen,
        unknown_ids_seen=unknown_ids_seen,
        overlap_seen=overlap_seen,
    )
    return issues


def resolve_splits(
    provider: DatasetProvider,
    split_config: SplitConfig,
    sample_ids: Sequence[str],
) -> Dict[str, List[str]]:
    ids = list(sample_ids)
    if split_config.mode == "explicit":
        assert split_config.explicit_splits is not None
        return _resolve_explicit_splits(split_config.explicit_splits, ids)
    if split_config.mode == "column":
        return _resolve_column_splits(provider, split_config, ids)
    return _resolve_random_splits(split_config, ids)


def validate_resolved_split_stability(
    *,
    existing_splits: Any,
    has_dataset_ref: bool,
    resolved_splits: Mapping[str, Sequence[str]],
    split_config: SplitConfig,
) -> None:
    if split_config.mode != "column":
        return
    if not has_dataset_ref or not existing_splits:
        return
    previous = _canonical_split_assignments(existing_splits)
    current = _canonical_split_assignments(resolved_splits)
    if previous != current:
        raise DatasetMismatchError(
            "Resolved column split assignments changed for this project state. "
            "Use a new workdir/project when changing split column values."
        )


def _record_split_owners(
    split_ids: Sequence[str],
    *,
    split_name_str: str,
    owners: Dict[str, str],
    issues: List[str],
    overlap_seen: bool,
) -> bool:
    for sample_id in split_ids:
        previous_split = owners.get(sample_id)
        if previous_split is not None and previous_split != split_name_str:
            overlap_seen = True
            issues.append(
                f"Persisted split overlap: sample {sample_id!r} appears in both "
                f"{previous_split!r} and {split_name_str!r}."
            )
            continue
        owners.setdefault(sample_id, split_name_str)
    return overlap_seen


def _append_split_coverage_issue(
    issues: List[str],
    *,
    reference_ids: set[str],
    owners: Mapping[str, str],
    duplicate_ids_seen: bool,
    unknown_ids_seen: bool,
    overlap_seen: bool,
) -> None:
    assigned_known_ids = {sample_id for sample_id in owners if sample_id in reference_ids}
    missing_from_splits = sorted(reference_ids - assigned_known_ids)
    extra_in_splits = sorted(set(owners) - reference_ids)
    if not (missing_from_splits or extra_in_splits or duplicate_ids_seen or unknown_ids_seen or overlap_seen):
        return

    details: List[str] = []
    if missing_from_splits:
        details.append(f"missing ids: {missing_from_splits[:10]}")
    if extra_in_splits:
        details.append(f"unknown ids: {extra_in_splits[:10]}")
    if duplicate_ids_seen:
        details.append("duplicates present")
    if overlap_seen:
        details.append("overlaps present")
    issues.append("Persisted split coverage is invalid: " + "; ".join(details) + ".")


def _resolve_explicit_splits(explicit_splits: Mapping[str, Sequence[str]], ids: Sequence[str]) -> Dict[str, List[str]]:
    id_set = set(ids)
    resolved: Dict[str, List[str]] = {}
    for split_name, split_ids in explicit_splits.items():
        resolved_ids = [str(sample_id) for sample_id in split_ids]
        _validate_explicit_split_ids(split_name, resolved_ids, id_set)
        resolved[split_name] = resolved_ids

    _validate_explicit_split_coverage(resolved, id_set)
    return resolved


def _validate_explicit_split_ids(split_name: str, resolved_ids: Sequence[str], id_set: set[str]) -> None:
    duplicate_ids = [sample_id for sample_id, count in Counter(resolved_ids).items() if count > 1]
    if duplicate_ids:
        preview = ", ".join(repr(sample_id) for sample_id in duplicate_ids[:5])
        raise ConfigurationError(f"Duplicate sample_id in {split_name!r} split: {preview}")
    unknown = [sample_id for sample_id in resolved_ids if sample_id not in id_set]
    if unknown:
        preview = ", ".join(repr(sample_id) for sample_id in unknown[:5])
        raise ConfigurationError(f"Unknown split sample_id in {split_name!r}: {preview}")


def _validate_explicit_split_coverage(resolved: Mapping[str, Sequence[str]], id_set: set[str]) -> None:
    owners: Dict[str, str] = {}
    overlaps: List[Tuple[str, str, str]] = []
    for split_name, split_ids in resolved.items():
        for sample_id in split_ids:
            previous_split = owners.get(sample_id)
            if previous_split is not None:
                overlaps.append((sample_id, previous_split, split_name))
                continue
            owners[sample_id] = split_name

    if overlaps:
        preview = ", ".join(
            f"{sample_id!r} in {left!r} and {right!r}" for sample_id, left, right in overlaps[:5]
        )
        raise ConfigurationError(f"Explicit split overlap detected: {preview}")

    missing = sorted(id_set - set(owners))
    if missing:
        preview = ", ".join(repr(sample_id) for sample_id in missing[:5])
        raise ConfigurationError(f"Explicit split map does not cover every dataset sample_id: {preview}")


def _resolve_column_splits(
    provider: DatasetProvider,
    split_config: SplitConfig,
    sample_ids: Sequence[str],
) -> Dict[str, List[str]]:
    assert split_config.split_column is not None
    split_column = split_config.split_column
    resolved: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for sample in _get_samples_from_provider(provider, sample_ids):
        raw_value = _sample_split_value(sample, split_column)
        if raw_value is None:
            raise ConfigurationError(f"Missing split column {split_column!r} for sample_id {sample.sample_id!r}.")
        split_name = str(raw_value).strip().lower()
        if split_name not in resolved:
            raise ConfigurationError(
                f"Unknown split value {raw_value!r} in column {split_column!r} "
                f"for sample_id {sample.sample_id!r}; expected one of: train, val, test."
            )
        resolved[split_name].append(sample.sample_id)

    return resolved


def _resolve_random_splits(split_config: SplitConfig, ids: Sequence[str]) -> Dict[str, List[str]]:
    import random

    rng = random.Random(split_config.seed)
    ids_shuffled = list(ids)
    rng.shuffle(ids_shuffled)

    n = len(ids_shuffled)
    n_train = int(n * split_config.train_ratio)
    n_val = int(n * split_config.val_ratio)
    train = ids_shuffled[:n_train]
    val = ids_shuffled[n_train:n_train + n_val]
    test = ids_shuffled[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def _canonical_split_assignments(splits: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    return {
        split_name: sorted(str(sample_id) for sample_id in splits.get(split_name, []))
        for split_name in ("train", "val", "test")
    }


def _sample_split_value(sample: Any, split_column: str) -> Any:
    if split_column in sample.data:
        return sample.data[split_column]
    if split_column in sample.meta:
        return sample.meta[split_column]
    return None


def _get_samples_from_provider(provider: DatasetProvider, sample_ids: Sequence[str]) -> List[Any]:
    getter = getattr(provider, "get_samples", None)
    if callable(getter):
        return list(getter(sample_ids))
    return [provider.get_sample(sample_id) for sample_id in sample_ids]
