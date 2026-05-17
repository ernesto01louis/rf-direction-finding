"""Unit tests for the SkyPilot multi-cloud compute backend.

All SkyPilot SDK calls are mocked via a fake ``sky`` module injected into
``sys.modules``.  No real network calls are made; the unit suite must pass
without the ``skypilot`` package installed (or with it installed but no cloud
credentials configured).
"""

from __future__ import annotations

import asyncio
import shlex
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fake SkyPilot SDK helpers
# ---------------------------------------------------------------------------


def _make_fake_job_record(
    *,
    job_id: int = 1,
    status_value: str = "RUNNING",
) -> Any:
    """Build a fake ClusterJobRecord-like object."""
    record = types.SimpleNamespace(
        job_id=job_id,
        job_name="rfdf-test",
        username="test",
        submitted_at=1.0,
        status=types.SimpleNamespace(value=status_value),
        resources="",
        log_path="/tmp/log",
    )
    return record


def _make_fake_cluster_status(
    *,
    name: str = "rfdf-test",
    status_value: str = "UP",
) -> Any:
    """Build a fake StatusResponse-like object."""
    return types.SimpleNamespace(
        name=name,
        status=types.SimpleNamespace(value=status_value),
        nodes=1,
    )


class _FakeRequestId(str):
    """Minimal RequestId that wraps the result value."""

    def __new__(cls, value: Any) -> _FakeRequestId:
        instance = super().__new__(cls, "fake-request-id")
        instance._value = value  # type: ignore[attr-defined]
        return instance


def _make_fake_sky(
    *,
    launch_returns: tuple[int | None, Any] = (1, None),
    queue_returns: list[Any] | None = None,
    status_returns: list[Any] | None = None,
    cancel_raises: Exception | None = None,
    down_raises: Exception | None = None,
    no_clouds: bool = False,
) -> types.ModuleType:
    """Build a minimal fake ``sky`` module for mocking.

    Args:
        launch_returns: The (job_id, handle) tuple returned by stream_and_get(launch(...)).
        queue_returns: List of job records from stream_and_get(queue(...)).
        status_returns: List of cluster status records from stream_and_get(status(...)).
        cancel_raises: If set, sky.cancel raises this exception.
        down_raises: If set, sky.down raises this exception.
        no_clouds: If True, get_cached_enabled_clouds_or_refresh returns [].

    Returns:
        A ``types.ModuleType`` shaped like the ``sky`` SDK.
    """
    fake = types.ModuleType("sky")

    # --- Enum-like stubs ---
    class _OptimizeTarget:
        COST = 0

    class _ClusterStatus:
        INIT = types.SimpleNamespace(value="INIT")
        UP = types.SimpleNamespace(value="UP")
        STOPPED = types.SimpleNamespace(value="STOPPED")

    class _JobStatus:
        RUNNING = types.SimpleNamespace(value="RUNNING")
        SUCCEEDED = types.SimpleNamespace(value="SUCCEEDED")
        FAILED = types.SimpleNamespace(value="FAILED")
        CANCELLED = types.SimpleNamespace(value="CANCELLED")

    fake.OptimizeTarget = _OptimizeTarget  # type: ignore[attr-defined]
    fake.ClusterStatus = _ClusterStatus  # type: ignore[attr-defined]
    fake.JobStatus = _JobStatus  # type: ignore[attr-defined]

    # --- Resources / Task stubs ---
    class _Resources:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

    class _Task:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs
            self.docker_image: str | None = None

    fake.Resources = _Resources  # type: ignore[attr-defined]
    fake.Task = _Task  # type: ignore[attr-defined]

    # --- stream_and_get ---
    _launch_result = launch_returns
    _queue_result = queue_returns if queue_returns is not None else [_make_fake_job_record()]
    _status_result = status_returns if status_returns is not None else [_make_fake_cluster_status()]

    def _stream_and_get(request_id: Any, **kwargs: Any) -> Any:
        if hasattr(request_id, "_sky_action"):
            action = request_id._sky_action
            if action == "launch":
                return _launch_result
            if action == "queue":
                return _queue_result
            if action == "status":
                return _status_result
            if action == "cancel":
                if cancel_raises:
                    raise cancel_raises
                return None
            if action == "down":
                if down_raises:
                    raise down_raises
                return None
        # Default: return launch result
        return _launch_result

    fake.stream_and_get = _stream_and_get  # type: ignore[attr-defined]

    # --- Core API functions ---
    def _launch(task: Any, cluster_name: str = "", **kwargs: Any) -> _FakeRequestId:
        req = _FakeRequestId(_launch_result)
        req._sky_action = "launch"  # type: ignore[attr-defined]
        return req

    def _queue(cluster_name: str, **kwargs: Any) -> _FakeRequestId:
        req = _FakeRequestId(_queue_result)
        req._sky_action = "queue"  # type: ignore[attr-defined]
        return req

    def _status(cluster_names: list[str] | None = None, **kwargs: Any) -> _FakeRequestId:
        req = _FakeRequestId(_status_result)
        req._sky_action = "status"  # type: ignore[attr-defined]
        return req

    def _cancel(cluster_name: str, **kwargs: Any) -> _FakeRequestId:
        req = _FakeRequestId(None)
        req._sky_action = "cancel"  # type: ignore[attr-defined]
        return req

    def _down(cluster_name: str, **kwargs: Any) -> _FakeRequestId:
        req = _FakeRequestId(None)
        req._sky_action = "down"  # type: ignore[attr-defined]
        return req

    def _tail_logs(
        cluster_name: str, job_id: Any = None, follow: bool = True, **kwargs: Any
    ) -> str:
        return f"[log] output from cluster {cluster_name}\n"

    fake.launch = _launch  # type: ignore[attr-defined]
    fake.queue = _queue  # type: ignore[attr-defined]
    fake.status = _status  # type: ignore[attr-defined]
    fake.cancel = _cancel  # type: ignore[attr-defined]
    fake.down = _down  # type: ignore[attr-defined]
    fake.tail_logs = _tail_logs  # type: ignore[attr-defined]

    # --- Storage stub ---
    class _Storage:
        def __init__(self, name: str = "", **kwargs: Any) -> None:
            self._name = name

    class _StorageMode:
        MOUNT = "MOUNT"
        COPY = "COPY"

    fake.Storage = _Storage  # type: ignore[attr-defined]
    fake.StorageMode = _StorageMode  # type: ignore[attr-defined]

    # --- check module ---
    fake_check = types.ModuleType("sky.check")
    _enabled_clouds: list[Any] = [] if no_clouds else [object()]  # non-empty = has clouds

    def _get_cached_enabled_clouds_or_refresh(**kwargs: Any) -> list[Any]:
        return _enabled_clouds

    fake_check.get_cached_enabled_clouds_or_refresh = (  # type: ignore[attr-defined]
        _get_cached_enabled_clouds_or_refresh
    )
    fake.check = fake_check  # type: ignore[attr-defined]

    # --- exceptions module ---
    fake_exc = types.ModuleType("sky.exceptions")

    class NoCloudAccessError(RuntimeError):
        pass

    fake_exc.NoCloudAccessError = NoCloudAccessError  # type: ignore[attr-defined]
    fake.exceptions = fake_exc  # type: ignore[attr-defined]

    return fake


@pytest.fixture()
def fake_sky(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake sky module and reload the skypilot backend."""
    fake = _make_fake_sky()
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setitem(sys.modules, "sky.check", fake.check)  # type: ignore[attr-defined]

    # Also inject CloudCapability so _check_clouds can import it
    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"
        STORAGE = "storage"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky.clouds", fake_clouds)
    monkeypatch.setitem(sys.modules, "sky.clouds.cloud", fake_cloud_mod)

    import importlib

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)
    return fake


@pytest.fixture()
def backend(fake_sky: types.ModuleType) -> Any:
    """Return a SkyPilotCompute instance wired to the fake SDK."""
    import rfdf.backends.compute.skypilot as mod

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
    assert backend.name == "skypilot"


def test_supports_gpu(backend: Any) -> None:
    assert backend.supports_gpu is True


def test_supports_persistent_storage(backend: Any) -> None:
    assert backend.supports_persistent_storage is True


def test_factory_returns_skypilot_compute(fake_sky: types.ModuleType) -> None:
    import rfdf.backends.compute.skypilot as mod

    b = mod.create()
    assert b.name == "skypilot"


# ---------------------------------------------------------------------------
# Cost estimate arithmetic (SDK-free)
# ---------------------------------------------------------------------------


def test_cost_estimate_1x_to_2x_spread(backend: Any, tmp_path: Path) -> None:
    """low <= estimated <= high and estimated = 1.5x low."""
    job = _make_job(tmp_path, gpu_count=1, gpu_model=None)
    est = _run(backend.cost_estimate(job))
    assert est.backend == "skypilot"
    assert est.low_usd <= est.estimated_usd <= est.high_usd
    assert abs(est.estimated_usd - 1.5 * est.low_usd) < 1e-9
    assert abs(est.high_usd - 2.0 * est.low_usd) < 1e-9


def test_cost_estimate_known_gpu_model(backend: Any, tmp_path: Path) -> None:
    """A known GPU model uses the correct hourly rate from _RATES."""
    import rfdf.backends.compute.skypilot as mod

    job = _make_job(tmp_path, gpu_count=2, gpu_model="A100")
    est = _run(backend.cost_estimate(job))
    # A100 rate, timeout 1h, 2 GPUs
    expected_low = mod._RATES["A100"] * 1.0 * (3600.0 / 3600.0) * 2
    assert abs(est.low_usd - expected_low) < 1e-9
    assert est.rationale  # must be non-empty


def test_cost_estimate_fallback_rate(backend: Any, tmp_path: Path) -> None:
    """Unknown GPU model falls back to _RATES[None]."""
    import rfdf.backends.compute.skypilot as mod

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


def test_cost_estimate_sdk_free(tmp_path: Path) -> None:
    """cost_estimate works without the sky SDK in sys.modules."""
    saved = sys.modules.pop("sky", None)
    sys.modules.pop("sky.check", None)
    sys.modules.pop("sky.clouds", None)
    sys.modules.pop("sky.clouds.cloud", None)
    sys.modules.pop("rfdf.backends.compute.skypilot", None)
    try:
        import importlib

        import rfdf.backends.compute.skypilot as mod

        importlib.reload(mod)
        b = mod.create()
        (tmp_path / "s.py").write_text("pass\n")
        from rfdf.hal.compute import ComputeJob

        job = ComputeJob(entry_script="s.py", working_dir=tmp_path, timeout_s=1800.0)
        est = _run(b.cost_estimate(job))
        assert est.backend == "skypilot"
        assert est.low_usd <= est.estimated_usd <= est.high_usd
    finally:
        if saved is not None:
            sys.modules["sky"] = saved


def test_cost_estimate_1_5x_factor(backend: Any, tmp_path: Path) -> None:
    """Confirm 1.5x factor: estimated is exactly midway between low and high."""
    import rfdf.backends.compute.skypilot as mod

    job = _make_job(tmp_path, gpu_count=1, gpu_model="H100")
    est = _run(backend.cost_estimate(job))
    runtime_h = job.timeout_s / 3600.0
    rate = mod._RATES["H100"]
    assert abs(est.estimated_usd - rate * 1.5 * runtime_h * 1) < 1e-9
    assert abs(est.low_usd - rate * 1.0 * runtime_h * 1) < 1e-9
    assert abs(est.high_usd - rate * 2.0 * runtime_h * 1) < 1e-9


# ---------------------------------------------------------------------------
# submit / status / cancel
# ---------------------------------------------------------------------------


def test_submit_returns_skypilot_handle(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    assert handle.backend == "skypilot"
    assert handle.job_id  # non-empty string (cluster name)
    assert handle.submitted_at_s > 0


def test_submit_cluster_name_has_rfdf_prefix(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    assert handle.job_id.startswith("rfdf-")


def test_submit_sets_extra_metadata(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path, gpu_count=1, gpu_model="A100")
    handle = _run(backend.submit(job))
    assert "cluster_name" in handle.extra
    assert "sky_job_id" in handle.extra


def test_submit_raises_when_no_clouds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """submit() raises RuntimeError when no cloud credentials are configured."""
    import importlib

    fake = _make_fake_sky(no_clouds=True)
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setitem(sys.modules, "sky.check", fake.check)  # type: ignore[attr-defined]

    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky.clouds", fake_clouds)
    monkeypatch.setitem(sys.modules, "sky.clouds.cloud", fake_cloud_mod)

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    with pytest.raises(RuntimeError, match="no cloud credentials"):
        _run(b.submit(job))


def test_status_running(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    # Default fake_sky queue returns a RUNNING job.
    status = _run(backend.status(handle))
    assert status == JobStatus.RUNNING


def test_status_completed_when_cluster_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When both queue and status fail (cluster torn down), return COMPLETED."""
    import importlib

    # queue returns empty list → falls through to cluster status
    # cluster status also returns empty list → treat as COMPLETED (torn down)
    fake = _make_fake_sky(queue_returns=[], status_returns=[])
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setitem(sys.modules, "sky.check", fake.check)  # type: ignore[attr-defined]

    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky.clouds", fake_clouds)
    monkeypatch.setitem(sys.modules, "sky.clouds.cloud", fake_cloud_mod)

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)
    b = mod.create()

    from rfdf.hal.compute import JobHandle
    from rfdf.hal.types import JobStatus

    handle = JobHandle(backend="skypilot", job_id="rfdf-test", submitted_at_s=0.0)
    status = _run(b.status(handle))
    assert status == JobStatus.COMPLETED


def test_status_succeeded_maps_to_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SUCCEEDED sky job status maps to COMPLETED rfdf status."""
    import importlib

    from rfdf.hal.types import JobStatus

    fake = _make_fake_sky(queue_returns=[_make_fake_job_record(status_value="SUCCEEDED")])
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setitem(sys.modules, "sky.check", fake.check)  # type: ignore[attr-defined]

    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky.clouds", fake_clouds)
    monkeypatch.setitem(sys.modules, "sky.clouds.cloud", fake_cloud_mod)

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))
    status = _run(b.status(handle))
    assert status == JobStatus.COMPLETED


def test_status_failed_maps_correctly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib

    from rfdf.hal.types import JobStatus

    fake = _make_fake_sky(queue_returns=[_make_fake_job_record(status_value="FAILED")])
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setitem(sys.modules, "sky.check", fake.check)  # type: ignore[attr-defined]

    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky.clouds", fake_clouds)
    monkeypatch.setitem(sys.modules, "sky.clouds.cloud", fake_cloud_mod)

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))
    status = _run(b.status(handle))
    assert status == JobStatus.FAILED


def test_cancel_calls_down(backend: Any, tmp_path: Path) -> None:
    """cancel() calls sky.down to release resources."""
    # With a RUNNING job, cancel should proceed without raising.
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    # Should not raise; fake sky's down/cancel are no-ops.
    _run(backend.cancel(handle))


def test_cancel_noop_when_already_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cancel() on a completed job does not call sky.down again."""
    import importlib

    down_calls: list[str] = []

    fake = _make_fake_sky(queue_returns=[_make_fake_job_record(status_value="SUCCEEDED")])

    # Spy on down
    real_down = fake.down

    def _spy_down(cluster_name: str, **kwargs: Any) -> Any:
        down_calls.append(cluster_name)
        return real_down(cluster_name, **kwargs)

    fake.down = _spy_down  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setitem(sys.modules, "sky.check", fake.check)  # type: ignore[attr-defined]

    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sky.clouds", fake_clouds)
    monkeypatch.setitem(sys.modules, "sky.clouds.cloud", fake_cloud_mod)

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))
    _run(b.cancel(handle))
    # No down() call because status is COMPLETED → early return.
    assert down_calls == []


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def test_logs_yields_status_lines(backend: Any, tmp_path: Path) -> None:
    """logs() yields at least one status line."""
    # Default fake_sky returns RUNNING; after one iteration the generator
    # continues — in test we patch the queue to return SUCCEEDED so it exits.
    import importlib

    fake = _make_fake_sky(queue_returns=[_make_fake_job_record(status_value="SUCCEEDED")])

    import sys

    sys.modules["sky"] = fake
    sys.modules["sky.check"] = fake.check  # type: ignore[attr-defined]

    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    sys.modules["sky.clouds"] = fake_clouds
    sys.modules["sky.clouds.cloud"] = fake_cloud_mod

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))

    async def collect() -> list[str]:
        lines: list[str] = []
        async for line in b.logs(handle):
            lines.append(line)
        return lines

    lines = _run(collect())
    assert len(lines) >= 1
    assert any("skypilot" in line.lower() or "status" in line.lower() for line in lines)


def test_logs_yields_error_line_on_queue_failure(tmp_path: Path) -> None:
    """If sky.queue raises, logs() yields an error line and returns."""
    import importlib

    fake = _make_fake_sky()

    def _broken_queue(cluster_name: str, **kwargs: Any) -> Any:
        raise RuntimeError("queue boom")

    fake.queue = _broken_queue  # type: ignore[attr-defined]

    sys.modules["sky"] = fake
    sys.modules["sky.check"] = fake.check  # type: ignore[attr-defined]

    fake_clouds = types.ModuleType("sky.clouds")
    fake_cloud_mod = types.ModuleType("sky.clouds.cloud")

    class _CloudCapability:
        COMPUTE = "compute"

    fake_cloud_mod.CloudCapability = _CloudCapability  # type: ignore[attr-defined]
    fake_clouds.cloud = fake_cloud_mod  # type: ignore[attr-defined]
    sys.modules["sky.clouds"] = fake_clouds
    sys.modules["sky.clouds.cloud"] = fake_cloud_mod

    import rfdf.backends.compute.skypilot as mod

    importlib.reload(mod)

    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="skypilot", job_id="rfdf-test", submitted_at_s=0.0)

    async def collect() -> list[str]:
        lines: list[str] = []
        async for line in mod._stream_sky_logs(handle):
            lines.append(line)
        return lines

    lines = _run(collect())
    assert any("failed" in line.lower() or "boom" in line.lower() for line in lines)

    sys.modules.pop("sky", None)
    sys.modules.pop("rfdf.backends.compute.skypilot", None)


# ---------------------------------------------------------------------------
# fetch_artifacts
# ---------------------------------------------------------------------------


def test_fetch_artifacts_dest_must_exist(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="skypilot", job_id="rfdf-test", submitted_at_s=0.0)
    missing = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        _run(backend.fetch_artifacts(handle, missing))


def test_fetch_artifacts_dest_must_be_directory(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="skypilot", job_id="rfdf-test", submitted_at_s=0.0)
    file_dest = tmp_path / "file.txt"
    file_dest.write_text("x")
    with pytest.raises(NotADirectoryError):
        _run(backend.fetch_artifacts(handle, file_dest))


def test_fetch_artifacts_logs_warning_not_raises(
    backend: Any, fake_sky: types.ModuleType, tmp_path: Path
) -> None:
    """fetch_artifacts on an existing dir should not raise."""
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="skypilot", job_id="rfdf-test", submitted_at_s=0.0)
    dest = tmp_path / "out"
    dest.mkdir()
    _run(backend.fetch_artifacts(handle, dest))


# ---------------------------------------------------------------------------
# Missing SDK — clear ImportError
# ---------------------------------------------------------------------------


def test_missing_sdk_raises_import_error_on_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """submit() raises ImportError with install instructions when SDK absent."""
    import rfdf.backends.compute.skypilot as mod

    def _no_sdk() -> Any:
        raise ImportError(
            "The SkyPilot backend requires the skypilot SDK; "
            "install it with:  pip install rfdf[compute-skypilot]"
        )

    monkeypatch.setattr(mod, "_require_sky", _no_sdk)
    b = mod.create()
    job = _make_job(tmp_path)
    with pytest.raises(ImportError, match="rfdf\\[compute-skypilot\\]"):
        _run(b.submit(job))


def test_missing_sdk_raises_import_error_on_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """status() raises ImportError when SDK absent."""
    import rfdf.backends.compute.skypilot as mod
    from rfdf.hal.compute import JobHandle

    def _no_sdk() -> Any:
        raise ImportError(
            "The SkyPilot backend requires the skypilot SDK; "
            "install it with:  pip install rfdf[compute-skypilot]"
        )

    monkeypatch.setattr(mod, "_require_sky", _no_sdk)
    b = mod.create()
    handle = JobHandle(backend="skypilot", job_id="rfdf-test", submitted_at_s=0.0)
    with pytest.raises(ImportError, match="rfdf\\[compute-skypilot\\]"):
        _run(b.status(handle))


# ---------------------------------------------------------------------------
# Lazy-import guarantee
# ---------------------------------------------------------------------------


def test_import_does_not_pull_sky_into_sys_modules() -> None:
    """Importing the skypilot backend must NOT add 'sky' to sys.modules."""
    sky_was_present = "sky" in sys.modules
    saved = sys.modules.pop("sky", None)
    sys.modules.pop("sky.check", None)
    sys.modules.pop("sky.clouds", None)
    sys.modules.pop("sky.clouds.cloud", None)
    sys.modules.pop("rfdf.backends.compute.skypilot", None)

    try:
        import rfdf.backends.compute.skypilot  # noqa: F401

        assert "sky" not in sys.modules, (
            "lazy-import broken: 'sky' appeared in sys.modules after importing "
            "rfdf.backends.compute.skypilot without calling any method"
        )
    finally:
        if saved is not None:
            sys.modules["sky"] = saved
        elif not sky_was_present:
            sys.modules.pop("sky", None)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance(backend: Any) -> None:
    """SkyPilotCompute is a structural ComputeBackend."""
    from rfdf.hal.compute import ComputeBackend

    assert isinstance(backend, ComputeBackend)


# ---------------------------------------------------------------------------
# _build_run_command — command construction + shlex quoting guard
# ---------------------------------------------------------------------------


def test_build_run_command_basic(tmp_path: Path) -> None:
    """_build_run_command returns a plain python invocation."""
    import rfdf.backends.compute.skypilot as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="train.py", working_dir=tmp_path)
    cmd = mod._build_run_command(job)
    assert "python" in cmd
    assert "train.py" in cmd


def test_build_run_command_parses_cleanly(tmp_path: Path) -> None:
    """The run command produced for any script parses cleanly with shlex."""
    import rfdf.backends.compute.skypilot as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="my_script.py", working_dir=tmp_path)
    cmd = mod._build_run_command(job)
    tokens = shlex.split(cmd)
    assert "python" in tokens
    assert "my_script.py" in tokens


def test_build_sky_resources_gpu_spec(tmp_path: Path, fake_sky: types.ModuleType) -> None:
    """_build_sky_resources encodes gpu_count and gpu_model correctly."""
    import sky

    import rfdf.backends.compute.skypilot as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(
        entry_script="r.py",
        working_dir=tmp_path,
        gpu_count=2,
        gpu_model="A100",
    )
    resources = mod._build_sky_resources(job, sky)
    assert resources._kwargs.get("accelerators") == {"A100": 2}


def test_build_sky_resources_cpu_only(tmp_path: Path, fake_sky: types.ModuleType) -> None:
    """gpu_count=0 produces no accelerators."""
    import sky

    import rfdf.backends.compute.skypilot as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=0)
    resources = mod._build_sky_resources(job, sky)
    assert resources._kwargs.get("accelerators") is None


# ---------------------------------------------------------------------------
# Storage property
# ---------------------------------------------------------------------------


def test_storage_property_returns_skypilot_bucket_storage(backend: Any) -> None:
    from rfdf.backends.compute._remote_storage import SkyPilotBucketStorage

    assert isinstance(backend.storage, SkyPilotBucketStorage)


def test_storage_name(backend: Any) -> None:
    assert backend.storage.name == "skypilot-bucket"
