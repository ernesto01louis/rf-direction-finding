"""Self-test report formatting for ``rfdf hw selftest``.

The per-backend exercises live in ``rfdf.cli.hw``; this module turns the
resulting report dict into the two documented output formats — machine-readable
JSON (for automation) and a colour-coded human tree (the default).
"""

from __future__ import annotations

import json
from typing import Any

#: ANSI colours; empty strings when colour is disabled.
_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_SECTION_ORDER = ("sdr", "rotator", "geometry", "compute")


def format_json(report: dict[str, Any]) -> str:
    """Render the self-test report as indented JSON."""
    return json.dumps(report, indent=2)


def format_human(report: dict[str, Any], *, color: bool = True) -> str:
    """Render the self-test report as a colour-coded human tree.

    Args:
        report: The selftest report — one entry per HAL section, each a dict
            with ``name`` / ``ok`` / ``latency_ms`` / ``error_msg`` and an
            optional ``status`` mapping of device-specific smoke detail.
        color: Emit ANSI colour codes when True.

    Returns:
        A multi-line human-readable report.
    """
    green = _GREEN if color else ""
    red = _RED if color else ""
    dim = _DIM if color else ""
    reset = _RESET if color else ""

    def tick(ok: bool) -> str:
        return f"{green}OK{reset}" if ok else f"{red}FAIL{reset}"

    lines: list[str] = []
    healthy = 0
    total = 0
    for section in _SECTION_ORDER:
        entry = report.get(section)
        if entry is None:
            continue
        total += 1
        ok = bool(entry.get("ok"))
        healthy += int(ok)
        name = entry.get("name", "?")
        latency = entry.get("latency_ms")
        suffix = f" {dim}({latency:.1f} ms){reset}" if isinstance(latency, int | float) else ""
        lines.append(
            f"{section.capitalize()}: {name} {'.' * max(1, 28 - len(name))} {tick(ok)}{suffix}"
        )
        if not ok and entry.get("error_msg"):
            lines.append(f"  {red}error{reset}: {entry['error_msg']}")
        status = entry.get("status")
        if isinstance(status, dict):
            for key, value in status.items():
                if key == "backend":
                    continue
                lines.append(f"  {dim}|- {key}: {value}{reset}")
    summary_ok = healthy == total and total > 0
    lines.append("")
    lines.append(f"{healthy}/{total} backends healthy. {tick(summary_ok)}")
    return "\n".join(lines)


__all__ = ["format_human", "format_json"]
