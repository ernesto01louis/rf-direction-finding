"""rfdf — hardware-agnostic RF direction finding, signal classification, and phased-array research.

The platform is **standalone-first**: it runs without `ai-orchestrator-client`. Installing the
`[orchestrator]` extra adds evidence-bundle production, Hindsight memory writes, L5 vault notes,
planner-dispatched GNU Radio flowgraphs, and ntfy alerts (Stage 7 deliverables).

No RF or ML imports happen at module load. All hardware and compute backends live in
``rfdf.backends`` and are pulled in via the entry-point discovery mechanism only when requested.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rfdf")
except PackageNotFoundError:  # pragma: no cover - editable install before metadata is wired
    __version__ = "0.0.1"

__all__ = ["__version__"]
