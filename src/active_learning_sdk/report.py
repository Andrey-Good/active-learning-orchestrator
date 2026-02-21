from __future__ import annotations

"""
Reporting.

The engine can call `project.generate_report(...)` to produce an HTML file with metrics.
In this scaffold we keep reporting as an optional component (it is not required for the
core loop to work).
"""

from pathlib import Path
from typing import Union

from .state.store import ProjectState


class ReportGenerator:
    """
    Report generator scaffold.

    For juniors:
    - This class is intentionally incomplete. It shows "where reporting would live".
    - Implement `generate_html()` when you want a real report.
    - Prefer keeping heavy dependencies (jinja2/matplotlib) optional.

    Attributes:
        (no stored state in the scaffold):
            Where: this class currently has no fields.
            What: it is just a place for report generation functions.
            Why: keeping it stateless makes it easy to use and test.
    """

    def generate_html(self, state: ProjectState, output_path: Union[str, Path]) -> None:
        """
        Generate an HTML report for a project state.

        Typical contents:
        - metrics over time (accuracy vs rounds or vs labeled count)
        - class distribution
        - basic round timeline
        """
        raise NotImplementedError("Implement report generation with jinja2 + matplotlib in an optional extra.")
