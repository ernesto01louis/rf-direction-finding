"""Unit tests for the RunPod compute backend.

All RunPod SDK calls are mocked via a fake ``runpod`` module injected into
``sys.modules``.  No real network calls are made; the unit suite must pass
without the ``runpod`` SDK installed (or with it installed but no key set).
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures — fake runpod SDK injected into sys.modules
# ---------------------------------------------------------------------------


def _make_fake_runpod(**overrides: Any) -> types.ModuleType:
    """Build a minimal fake ``runpod`` module for mocking.

    Args:
        **overrides: Replace named attributes on the fake module.

    Returns:
        A ``types.ModuleType`` that looks enough like ``runpod`` for the
        backend to call without errors.
    """
    fake = types.ModuleType("runpod")
    fake.api_key = None  # type: ignore[attr-defined]

    # Default pod dict returned by create_pod / get_pod.
    _default_pod: dict[str, Any] = {
        "id": "pod-abc123",
        "desiredStatus": "RUNNING",
        "uptimeSeconds": 30,
        "costPerHr": 0.44,
        "gpuCount": 1,
        "imageName": "runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
        "name": "rfdf-test",
        "runtime": None,
        "machine": {"gpuDisplayName": "RTX 3090"},
    }

    def _create_pod(**kwargs: Any) -> dict[str, Any]:  # type: ignore[return]
        return dict(_default_pod, name=kwargs.get("name", "rfdf-test"))

    def _get_pod(pod_id: str, api_key: str | None = None) -> dict[str, Any]:
        return dict(_default_pod, id=pod_id)

    def _terminate_pod(pod_id: str) -> dict[str, Any]:
        return {"id": pod_id, "desiredStatus": "TERMINATED"}

    def _stop_pod(pod_id: str) -> dict[str, Any]:
        return {"id": pod_id, "desiredStatus": "EXITED"}

    fake.create_pod = _create_pod  # type: ignore[attr-defined]
    fake.get_pod = _get_pod  # type: ignore[attr-defined]
    fake.terminate_pod = _terminate_pod  # type: ignore[attr-defined]
    fake.stop_pod = _stop_pod  # type: ignore[attr-defined]

    for k, v in overrides.items():
        setattr(fake, k, v)
    return fake


@pytest.fixture()
def fake_runpod(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake runpod module and set RUNPOD_API_KEY in the environment.

    Yields the fake module so tests can configure it further.
    """
    fake = _make_fake_runpod()
    monkeypatch.setitem(sys.modules, "runpod", fake)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-abc")
    # Reload the backend so _require_runpod() picks up the injected module.
    import importlib

    import rfdf.backends.compute.runpod as mod

    importlib.reload(mod)
    return fake


@pytest.fixture()
def backend(fake_runpod: types.ModuleType) -> Any:
    """Return a RunPodCompute instance wired to the fake SDK."""
    import rfdf.backends.compute.runpod as mod

    return mod.create()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(tmp_path: Path, *, gpu_count: int = 1, gpu_model: str | None = None) -> Any:
    from rfdf.hal.compute import ComputeJob

    (tmp_path / "train.py").write_text("print('ok')\n")
    return ComputeJob(
        entry_script="train.py",
        working_dir=tmp_path,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        timeout_s=3600.0,
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_name(backend: Any) -> None:
    assert backend.name == "runpod"


def test_supports_gpu(backend: Any) -> None:
    assert backend.supports_gpu is True


def test_supports_persistent_storage(backend: Any) -> None:
    assert backend.supports_persistent_storage is True


def test_factory_returns_runpod_compute(fake_runpod: types.ModuleType) -> None:
    import rfdf.backends.compute.runpod as mod

    b = mod.create()
    assert b.name == "runpod"


# ---------------------------------------------------------------------------
# Cost estimate arithmetic
# ---------------------------------------------------------------------------


def test_cost_estimate_1x_to_2x_spread(backend: Any, tmp_path: Path) -> None:
    """low <= estimated <= high and estimated = 1.5x low."""
    job = _make_job(tmp_path, gpu_count=1, gpu_model=None)
    est = _run(backend.cost_estimate(job))
    assert est.backend == "runpod"
    assert est.low_usd <= est.estimated_usd <= est.high_usd
    # 1.5x relationship: estimated = 1.5x low (both positive)
    assert abs(est.estimated_usd - 1.5 * est.low_usd) < 1e-9
    assert abs(est.high_usd - 2.0 * est.low_usd) < 1e-9


def test_cost_estimate_known_gpu_model(backend: Any, tmp_path: Path) -> None:
    """A known GPU model uses the correct hourly rate from _RATES."""
    import rfdf.backends.compute.runpod as mod

    job = _make_job(tmp_path, gpu_count=2, gpu_model="RTX 4090")
    est = _run(backend.cost_estimate(job))
    # RTX 4090 rate = 0.74, timeout 1h, 2 GPUs
    # low = 0.74 * 1.0 * 1.0 * 2 = 1.48
    expected_low = mod._RATES["RTX 4090"] * 1.0 * (3600.0 / 3600.0) * 2
    assert abs(est.low_usd - expected_low) < 1e-9
    assert est.rationale  # must be non-empty


def test_cost_estimate_fallback_rate(backend: Any, tmp_path: Path) -> None:
    """Unknown GPU model falls back to _RATES[None]."""
    import rfdf.backends.compute.runpod as mod

    job = _make_job(tmp_path, gpu_count=1, gpu_model="UNKNOWN_GPU_XYZ")
    est = _run(backend.cost_estimate(job))
    runtime_h = job.timeout_s / 3600.0
    expected_low = mod._RATES[None] * 1.0 * runtime_h * 1
    assert abs(est.low_usd - expected_low) < 1e-9


def test_cost_estimate_no_gpu_model(backend: Any, tmp_path: Path) -> None:
    """No gpu_model → fallback rate applied."""
    job = _make_job(tmp_path, gpu_count=0)
    est = _run(backend.cost_estimate(job))
    assert est.estimated_usd >= 0.0
    assert est.low_usd <= est.estimated_usd <= est.high_usd


# ---------------------------------------------------------------------------
# submit / status / cancel
# ---------------------------------------------------------------------------


def test_submit_returns_runpod_handle(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    assert handle.backend == "runpod"
    assert handle.job_id  # non-empty string
    assert handle.submitted_at_s > 0


def test_status_running(backend: Any, fake_runpod: types.ModuleType, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    # get_pod returns RUNNING by default
    status = _run(backend.status(handle))
    assert status == JobStatus.RUNNING


def test_status_completed(backend: Any, fake_runpod: types.ModuleType, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    fake_runpod.get_pod = lambda pod_id, api_key=None: {  # type: ignore[attr-defined]
        "id": pod_id,
        "desiredStatus": "EXITED",
        "uptimeSeconds": 120,
    }
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    status = _run(backend.status(handle))
    assert status == JobStatus.COMPLETED


def test_status_failed(backend: Any, fake_runpod: types.ModuleType, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    fake_runpod.get_pod = lambda pod_id, api_key=None: {  # type: ignore[attr-defined]
        "id": pod_id,
        "desiredStatus": "DEAD",
        "uptimeSeconds": 0,
    }
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    status = _run(backend.status(handle))
    assert status == JobStatus.FAILED


def test_status_none_pod_returns_cancelled(
    backend: Any, fake_runpod: types.ModuleType, tmp_path: Path
) -> None:
    """A None get_pod response (pod deleted) maps to CANCELLED."""
    from rfdf.hal.types import JobStatus

    fake_runpod.get_pod = lambda pod_id, api_key=None: None  # type: ignore[attr-defined]
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    status = _run(backend.status(handle))
    assert status == JobStatus.CANCELLED


def test_cancel_terminates_pod(backend: Any, fake_runpod: types.ModuleType, tmp_path: Path) -> None:
    terminated: list[str] = []

    def _terminate(pod_id: str) -> dict[str, Any]:
        terminated.append(pod_id)
        return {"id": pod_id, "desiredStatus": "TERMINATED"}

    fake_runpod.terminate_pod = _terminate  # type: ignore[attr-defined]

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    _run(backend.cancel(handle))
    assert handle.job_id in terminated


def test_cancel_noop_when_already_terminated(
    backend: Any, fake_runpod: types.ModuleType, tmp_path: Path
) -> None:
    """cancel() on a terminated pod does not call terminate_pod again."""
    terminate_calls: list[str] = []

    fake_runpod.get_pod = lambda pod_id, api_key=None: {  # type: ignore[attr-defined]
        "id": pod_id,
        "desiredStatus": "TERMINATED",
    }

    def _terminate(pod_id: str) -> dict[str, Any]:
        terminate_calls.append(pod_id)
        return {"id": pod_id, "desiredStatus": "TERMINATED"}

    fake_runpod.terminate_pod = _terminate  # type: ignore[attr-defined]

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    _run(backend.cancel(handle))
    # terminate_pod must NOT have been called because status is already CANCELLED.
    assert terminate_calls == []


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def test_logs_yields_status_lines(
    backend: Any, fake_runpod: types.ModuleType, tmp_path: Path
) -> None:
    """logs() yields at least one line and exits when the pod has EXITED."""
    call_count = 0

    def _get_pod(pod_id: str, api_key: str | None = None) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        # Return EXITED on the first poll so the generator terminates.
        return {
            "id": pod_id,
            "desiredStatus": "EXITED",
            "uptimeSeconds": 60,
        }

    fake_runpod.get_pod = _get_pod  # type: ignore[attr-defined]

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))

    async def collect() -> list[str]:
        lines: list[str] = []
        async for line in backend.logs(handle):
            lines.append(line)
        return lines

    lines = _run(collect())
    assert len(lines) >= 1
    assert any("EXITED" in line or "pod=" in line for line in lines)


# ---------------------------------------------------------------------------
# fetch_artifacts
# ---------------------------------------------------------------------------


def test_fetch_artifacts_dest_must_exist(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="runpod", job_id="pod-xyz", submitted_at_s=0.0)
    missing = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        _run(backend.fetch_artifacts(handle, missing))


def test_fetch_artifacts_dest_must_be_directory(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="runpod", job_id="pod-xyz", submitted_at_s=0.0)
    file_dest = tmp_path / "file.txt"
    file_dest.write_text("x")
    with pytest.raises(NotADirectoryError):
        _run(backend.fetch_artifacts(handle, file_dest))


def test_fetch_artifacts_logs_warning_not_raises(
    backend: Any, fake_runpod: types.ModuleType, tmp_path: Path
) -> None:
    """fetch_artifacts on an existing dir should not raise."""
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="runpod", job_id="pod-xyz", submitted_at_s=0.0)
    dest = tmp_path / "out"
    dest.mkdir()
    # Should complete without error even though file transfer is a stub.
    _run(backend.fetch_artifacts(handle, dest))


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_on_submit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """submit() raises RuntimeError when RUNPOD_API_KEY is not set."""
    import importlib

    fake = _make_fake_runpod()
    monkeypatch.setitem(sys.modules, "runpod", fake)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    import rfdf.backends.compute.runpod as mod

    importlib.reload(mod)
    backend = mod.create()
    job = _make_job(tmp_path)
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        _run(backend.submit(job))


# ---------------------------------------------------------------------------
# SDK not installed — clear ImportError
# ---------------------------------------------------------------------------


def test_missing_sdk_raises_import_error_on_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """submit() raises ImportError with install instructions when SDK absent."""
    import rfdf.backends.compute.runpod as mod

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

    def _no_sdk() -> Any:
        raise ImportError(
            "The RunPod backend requires the runpod SDK; "
            "install it with:  pip install rfdf[compute-runpod]"
        )

    monkeypatch.setattr(mod, "_require_runpod", _no_sdk)
    backend = mod.create()
    job = _make_job(tmp_path)

    with pytest.raises(ImportError, match="rfdf\\[compute-runpod\\]"):
        _run(backend.submit(job))


# ---------------------------------------------------------------------------
# Lazy-import guarantee
# ---------------------------------------------------------------------------


def test_import_does_not_pull_runpod_into_sys_modules() -> None:
    """Importing the runpod backend must NOT add 'runpod' to sys.modules."""
    # Remove any pre-existing runpod import so we start clean.
    runpod_was_present = "runpod" in sys.modules
    saved = sys.modules.pop("runpod", None)
    # Also remove the backend module so a fresh import is forced.
    sys.modules.pop("rfdf.backends.compute.runpod", None)

    try:
        import rfdf.backends.compute.runpod  # noqa: F401

        assert "runpod" not in sys.modules, (
            "lazy-import broken: 'runpod' appeared in sys.modules after importing "
            "rfdf.backends.compute.runpod without calling any method"
        )
    finally:
        # Restore whatever was there before.
        if saved is not None:
            sys.modules["runpod"] = saved
        elif not runpod_was_present:
            sys.modules.pop("runpod", None)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance(backend: Any) -> None:
    """RunPodCompute is a structural ComputeBackend."""
    from rfdf.hal.compute import ComputeBackend

    assert isinstance(backend, ComputeBackend)


# ---------------------------------------------------------------------------
# _build_docker_args — docker_args string construction
# ---------------------------------------------------------------------------


def test_build_docker_args_container_image(tmp_path: Path) -> None:
    """A container_image job runs the entry script directly (no bash wrapper)."""
    import rfdf.backends.compute.runpod as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="train.py", working_dir=tmp_path, container_image="my/image:tag")
    assert mod._build_docker_args(job) == "python train.py"


def test_build_docker_args_pip_requirements_quoting(tmp_path: Path) -> None:
    """pip_requirements yield a single, correctly-quoted ``bash -c`` command.

    Each requirement carries a shell metacharacter (``>``); the generated string
    must survive one round of shell parsing as exactly three tokens
    (``bash``, ``-c``, ``<command>``) with the full pip+python command intact —
    a regression guard against single-quote-nesting bugs.
    """
    import shlex

    import rfdf.backends.compute.runpod as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(
        entry_script="train.py",
        working_dir=tmp_path,
        pip_requirements=["torch>=2.0", "numpy>=1.24"],
    )
    tokens = shlex.split(mod._build_docker_args(job))
    assert tokens[:2] == ["bash", "-c"]
    assert len(tokens) == 3, f"docker_args did not parse to 3 tokens: {tokens!r}"
    inner = tokens[2]
    assert "pip install" in inner
    assert "torch>=2.0" in inner
    assert "numpy>=1.24" in inner
    assert "&& python train.py" in inner


def test_build_docker_args_no_requirements(tmp_path: Path) -> None:
    """With neither container_image nor pip_requirements, just run the entry."""
    import shlex

    import rfdf.backends.compute.runpod as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="train.py", working_dir=tmp_path)
    tokens = shlex.split(mod._build_docker_args(job))
    assert tokens == ["bash", "-c", "python train.py"]
