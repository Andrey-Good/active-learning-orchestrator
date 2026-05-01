"""
Dataset fingerprinting.

Fingerprinting is a safety feature: it detects if the dataset changed between runs.
Example problem it prevents:
- You start a project in `workdir` with dataset A.
- Later you accidentally run the same `workdir` with dataset B.
Without fingerprinting you could silently mix labels and destroy the experiment.
"""

from __future__ import annotations


import hashlib
import json
from typing import Any

from ..configs import FingerprintConfig
from ..exceptions import ConfigurationError
from .provider import DatasetProvider, _json_safe_value


class DatasetFingerprinter:
    """
    Compute a deterministic dataset fingerprint.

    Modes:
    - fast: uses ids + short text prefixes (good for big datasets)
    - strict: hashes the full normalized text for each sample
    - file: like fast, but can also include a provider file checksum if available

    Attributes:
        config (FingerprintConfig):
            Where: checked/used by all fingerprint methods.
            What: configuration (mode, hash algorithm, normalization).
            Why: lets you choose speed vs strictness and keep results reproducible.
    """

    def __init__(self, config: FingerprintConfig) -> None:
        config.validate()
        self.config = config

    def fingerprint(self, provider: DatasetProvider) -> str:
        """
        Compute the fingerprint string for a dataset provider.

        The output is stored in `ProjectState.dataset_ref.fingerprint` and compared on resume.
        """
        schema = provider.schema()
        sample_ids = sorted(list(provider.iter_sample_ids()))
        hasher = self._new_hasher()

        self._update_hasher(hasher, f"schema:{json.dumps(schema, sort_keys=True)}")
        self._update_hasher(hasher, f"count:{len(sample_ids)}")

        if self.config.mode in {"fast", "file"}:
            if self.config.mode == "file":
                checksum = getattr(provider, "file_checksum", None)
                if callable(checksum):
                    self._update_hasher(hasher, f"file_checksum:{checksum()}")
            for sample_id in sample_ids:
                sample = provider.get_sample(sample_id)
                text = self._normalize_text(str(sample.data.get("text", "")))
                prefix = text[: self.config.text_prefix_chars]
                token = f"{sample_id}|len={len(text)}|pre={prefix}"
                self._update_hasher(hasher, token)
            return self._finalize_hasher(hasher)

        if self.config.mode == "strict":
            for sample_id in sample_ids:
                sample = provider.get_sample(sample_id)
                payload = {
                    "data": self._strict_payload(sample.data),
                    "meta": self._strict_payload(sample.meta),
                    "group_id": self._strict_payload(sample.group_id),
                }
                self._update_hasher(hasher, sample_id)
                self._update_hasher(hasher, self._digest_payload(payload))
            return self._finalize_hasher(hasher)

        raise ConfigurationError(f"Unsupported fingerprint mode: {self.config.mode}")

    def _normalize_text(self, text: str) -> str:
        if not self.config.normalize_text:
            return text
        return " ".join(text.split())

    def _digest_text(self, text: str) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(text.encode("utf-8", errors="replace"))
        return hasher.hexdigest()

    def _strict_payload(self, data: Any) -> Any:
        payload = _json_safe_value(data)
        if isinstance(payload, dict) and "text" in payload:
            payload = dict(payload)
            payload["text"] = self._normalize_text(str(payload["text"]))
        return payload

    def _digest_payload(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return self._digest_text(serialized)

    def _new_hasher(self) -> Any:
        if self.config.hash_algo == "xxhash64":
            try:
                import xxhash  # type: ignore
            except Exception:
                raise ConfigurationError("xxhash is not installed. Install active-learning-sdk[xxhash] or use blake2b.")
            return xxhash.xxh64()  # type: ignore[attr-defined]
        if self.config.hash_algo == "sha256":
            return hashlib.sha256()
        return hashlib.blake2b(digest_size=32)

    def _update_hasher(self, hasher: Any, token: str) -> None:
        hasher.update(token.encode("utf-8", errors="replace"))

    def _finalize_hasher(self, hasher: Any) -> str:
        return hasher.hexdigest()
