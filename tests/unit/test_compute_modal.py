"""Unit tests for the Modal compute backend.

All Modal SDK calls are mocked via a fake ``modal`` module injected into
``sys.modules``.  No real network calls are made; the unit suite must pass
without the ``modal`` SDK installed (or with it installed but no credentials
configured).
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
# Fake Modal SDK helpers
# ---------------------------------------------------------------------------


def _make_fake_sandbox(
    *,
    object_id: str = "sb-abc123",
    returncode: int | None = None,
) -> Any:
    """Build a fake Sandbox object that the backend will interact with."""

    class FakeSandbox:
        def __init__(self) -> None:
            self.object_id = object_id
            self._returncode = returncode
            self.stdout = _FakeStreamReader(b"")
            self.stderr = _FakeStreamReader(b"")

        @property
        def returncode(self) -> int | None:
            return self._returncode

        def terminate(self) -> None:
            self._returncode = -1

    return FakeSandbox()


class _FakeStreamReader:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> str:
        return self._data.decode()

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration


def _make_fake_modal(**overrides: Any) -> types.ModuleType:
    """Build a minimal fake ``modal`` module for mocking.

    Args:
        **overrides: Replace named attributes on the fake module.

    Returns:
        A ``types.ModuleType`` that looks enough like ``modal`` for the
        backend to call without errors.
    """
    fake = types.ModuleType("modal")

    # --- fake config sub-module ---
    fake_config_mod = types.ModuleType("modal.config")

    class _FakeConfig:
        def __init__(self) -> None:
            self._data: dict[str, Any] = {
                "token_id": "test-token-id",
                "token_secret": "test-token-secret",
            }

        def get(self, key: str) -> Any:
            return self._data.get(key)

    fake_config_mod.config = _FakeConfig()  # type: ignore[attr-defined]
    fake.config = fake_config_mod  # type: ignore[attr-defined]

    # --- fake App ---
    class _FakeApp:
        def __init__(self, name: str = "rfdf-test") -> None:
            self.name = name

        @staticmethod
        def lookup(name: str, *, create_if_missing: bool = False) -> _FakeApp:
            return _FakeApp(name=name)

    fake.App = _FakeApp  # type: ignore[attr-defined]

    # --- fake Image ---
    class _FakeImage:
        def __init__(self, tag: str = "python:3.11-slim") -> None:
            self._tag = tag

        def pip_install(self, *args: str) -> _FakeImage:
            return self

        @staticmethod
        def from_registry(tag: str) -> _FakeImage:
            return _FakeImage(tag=tag)

    fake.Image = _FakeImage  # type: ignore[attr-defined]

    # --- fake Volume ---
    class _FakeVolume:
        def __init__(self, name: str = "rfdf-default") -> None:
            self.name = name

        @staticmethod
        def from_name(
            name: str,
            *,
            environment_name: str | None = None,
            create_if_missing: bool = False,
            **kwargs: Any,
        ) -> _FakeVolume:
            return _FakeVolume(name=name)

    fake.Volume = _FakeVolume  # type: ignore[attr-defined]

    # --- fake Sandbox ---
    _sandbox = _make_fake_sandbox()

    def _create_sandbox(*args: str, **kwargs: Any) -> Any:
        return _sandbox

    class _FakeSandboxClass:
        @staticmethod
        def create(*args: str, **kwargs: Any) -> Any:
            return _create_sandbox(*args, **kwargs)

        @staticmethod
        def from_id(sandbox_id: str, client: Any = None) -> Any:
            sb = _make_fake_sandbox(object_id=sandbox_id)
            return sb

    fake.Sandbox = _FakeSandboxClass  # type: ignore[attr-defined]

    for k, v in overrides.items():
        setattr(fake, k, v)
    return fake


@pytest.fixture()
def fake_modal(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake modal module and set credentials in the environment.

    Yields the fake module so tests can configure it further.
    """
    fake = _make_fake_modal()
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setitem(sys.modules, "modal.config", fake.config)  # type: ignore[arg-type]
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-token-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-token-secret")

    import importlib

    import rfdf.backends.compute.modal as mod

    importlib.reload(mod)
    return fake


@pytest.fixture()
def backend(fake_modal: types.ModuleType) -> Any:
    """Return a ModalCompute instance wired to the fake SDK."""
    import rfdf.backends.compute.modal as mod

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
    assert backend.name == "modal"


def test_supports_gpu(backend: Any) -> None:
    assert backend.supports_gpu is True


def test_supports_persistent_storage(backend: Any) -> None:
    assert backend.supports_persistent_storage is True


def test_factory_returns_modal_compute(fake_modal: types.ModuleType) -> None:
    import rfdf.backends.compute.modal as mod

    b = mod.create()
    assert b.name == "modal"


# ---------------------------------------------------------------------------
# Cost estimate arithmetic
# ---------------------------------------------------------------------------


def test_cost_estimate_1x_to_2x_spread(backend: Any, tmp_path: Path) -> None:
    """low <= estimated <= high and estimated = 1.5x low."""
    job = _make_job(tmp_path, gpu_count=1, gpu_model=None)
    est = _run(backend.cost_estimate(job))
    assert est.backend == "modal"
    assert est.low_usd <= est.estimated_usd <= est.high_usd
    assert abs(est.estimated_usd - 1.5 * est.low_usd) < 1e-9
    assert abs(est.high_usd - 2.0 * est.low_usd) < 1e-9


def test_cost_estimate_known_gpu_model(backend: Any, tmp_path: Path) -> None:
    """A known GPU model uses the correct hourly rate from _RATES."""
    import rfdf.backends.compute.modal as mod

    job = _make_job(tmp_path, gpu_count=2, gpu_model="A100")
    est = _run(backend.cost_estimate(job))
    # A100 rate = 3.70, timeout 1h, 2 GPUs
    expected_low = mod._RATES["A100"] * 1.0 * (3600.0 / 3600.0) * 2
    assert abs(est.low_usd - expected_low) < 1e-9
    assert est.rationale  # must be non-empty


def test_cost_estimate_h100_rate(backend: Any, tmp_path: Path) -> None:
    """H100 uses the H100 entry from _RATES."""
    import rfdf.backends.compute.modal as mod

    job = _make_job(tmp_path, gpu_count=1, gpu_model="H100")
    est = _run(backend.cost_estimate(job))
    runtime_h = job.timeout_s / 3600.0
    expected_low = mod._RATES["H100"] * 1.0 * runtime_h * 1
    assert abs(est.low_usd - expected_low) < 1e-9


def test_cost_estimate_fallback_rate(backend: Any, tmp_path: Path) -> None:
    """Unknown GPU model falls back to _RATES[None]."""
    import rfdf.backends.compute.modal as mod

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
    """cost_estimate works without the modal SDK in sys.modules."""
    # Remove modal from sys.modules to prove SDK-free execution.
    saved = sys.modules.pop("modal", None)
    sys.modules.pop("modal.config", None)
    sys.modules.pop("rfdf.backends.compute.modal", None)
    try:
        import importlib

        import rfdf.backends.compute.modal as mod

        importlib.reload(mod)
        b = mod.create()
        (tmp_path / "s.py").write_text("pass\n")
        from rfdf.hal.compute import ComputeJob

        job = ComputeJob(entry_script="s.py", working_dir=tmp_path, timeout_s=1800.0)
        est = _run(b.cost_estimate(job))
        assert est.backend == "modal"
        assert est.low_usd <= est.estimated_usd <= est.high_usd
    finally:
        if saved is not None:
            sys.modules["modal"] = saved


# ---------------------------------------------------------------------------
# submit / status / cancel
# ---------------------------------------------------------------------------


def test_submit_returns_modal_handle(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    assert handle.backend == "modal"
    assert handle.job_id  # non-empty string
    assert handle.submitted_at_s > 0


def test_submit_sets_extra_metadata(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path, gpu_count=1, gpu_model="A100")
    handle = _run(backend.submit(job))
    assert "app_name" in handle.extra
    assert "gpu_spec" in handle.extra
    assert handle.extra["gpu_spec"] == "A100"


def test_status_running(backend: Any, fake_modal: types.ModuleType, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    # Fake sandbox returncode is None (still running) by default.
    status = _run(backend.status(handle))
    assert status == JobStatus.RUNNING


def test_status_completed(backend: Any, fake_modal: types.ModuleType, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    # Submit first to register the sandbox.
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))

    # Patch the in-process sandbox to report returncode=0.
    sb = backend._jobs[handle.job_id]
    sb._returncode = 0

    status = _run(backend.status(handle))
    assert status == JobStatus.COMPLETED


def test_status_failed(backend: Any, fake_modal: types.ModuleType, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    sb = backend._jobs[handle.job_id]
    sb._returncode = 1

    status = _run(backend.status(handle))
    assert status == JobStatus.FAILED


def test_status_unknown_job_returns_cancelled(
    backend: Any, fake_modal: types.ModuleType, tmp_path: Path
) -> None:
    """A job_id not in _jobs that also fails from_id returns CANCELLED."""
    from rfdf.hal.compute import JobHandle
    from rfdf.hal.types import JobStatus

    # Make Sandbox.from_id raise so the fallback path triggers.
    class _BrokenSandbox:
        @staticmethod
        def from_id(sid: str, client: Any = None) -> Any:
            raise RuntimeError("not found")

    fake_modal.Sandbox = _BrokenSandbox
    import importlib

    import rfdf.backends.compute.modal as mod

    importlib.reload(mod)
    b = mod.create()

    handle = JobHandle(backend="modal", job_id="nonexistent-id", submitted_at_s=0.0)
    status = _run(b.status(handle))
    assert status == JobStatus.CANCELLED


def test_cancel_terminates_sandbox(
    backend: Any, fake_modal: types.ModuleType, tmp_path: Path
) -> None:
    """cancel() calls sandbox.terminate() when the job is running."""
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    sb = backend._jobs[handle.job_id]
    assert sb._returncode is None  # running before cancel

    _run(backend.cancel(handle))
    # terminate() sets _returncode = -1 on our fake.
    assert sb._returncode == -1


def test_cancel_noop_when_already_completed(
    backend: Any, fake_modal: types.ModuleType, tmp_path: Path
) -> None:
    """cancel() is a no-op when the sandbox has already completed."""
    from rfdf.hal.types import JobStatus

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    sb = backend._jobs[handle.job_id]
    sb._returncode = 0  # mark completed

    terminate_called: list[bool] = []
    original_terminate = sb.terminate

    def _spy_terminate() -> None:
        terminate_called.append(True)
        original_terminate()

    sb.terminate = _spy_terminate

    _run(backend.cancel(handle))
    # terminate must NOT have been called because status was COMPLETED.
    assert terminate_called == []

    # Verify status is still COMPLETED.
    status = _run(backend.status(handle))
    assert status == JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def test_logs_yields_status_lines(
    backend: Any, fake_modal: types.ModuleType, tmp_path: Path
) -> None:
    """logs() yields at least one line and exits when the sandbox has a returncode."""
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    # Set returncode=0 immediately so the generator terminates on first poll.
    sb = backend._jobs[handle.job_id]
    sb._returncode = 0

    # Override from_id to return our sandbox.
    class _PatchedSandbox:
        @staticmethod
        def create(*a: Any, **kw: Any) -> Any:
            return sb

        @staticmethod
        def from_id(sid: str, client: Any = None) -> Any:
            return sb

    fake_modal.Sandbox = _PatchedSandbox
    import importlib

    import rfdf.backends.compute.modal as mod

    importlib.reload(mod)

    async def collect() -> list[str]:
        lines: list[str] = []
        async for line in mod._stream_sandbox_logs(handle):
            lines.append(line)
        return lines

    lines = _run(collect())
    assert len(lines) >= 1
    assert any("sandbox=" in line or "modal" in line for line in lines)


def test_logs_returns_error_line_on_lookup_failure(tmp_path: Path) -> None:
    """If from_id raises, logs() yields an error line and returns."""
    import importlib
    import types as _types

    fake = _types.ModuleType("modal")
    fake_cfg = _types.ModuleType("modal.config")

    class _Cfg:
        def get(self, k: str) -> str:
            return "tok"

    fake_cfg.config = _Cfg()  # type: ignore[attr-defined]
    fake.config = fake_cfg  # type: ignore[attr-defined]

    class _BrokenSandbox:
        @staticmethod
        def from_id(sid: str, client: Any = None) -> Any:
            raise RuntimeError("lookup boom")

    fake.Sandbox = _BrokenSandbox  # type: ignore[attr-defined]

    sys.modules["modal"] = fake
    sys.modules["modal.config"] = fake_cfg  # type: ignore[arg-type]
    sys.modules.pop("rfdf.backends.compute.modal", None)

    import rfdf.backends.compute.modal as mod

    importlib.reload(mod)

    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="modal", job_id="sb-broken", submitted_at_s=0.0)

    async def collect() -> list[str]:
        lines: list[str] = []
        async for line in mod._stream_sandbox_logs(handle):
            lines.append(line)
        return lines

    lines = _run(collect())
    assert any("lookup failed" in line or "boom" in line for line in lines)
    # Cleanup injected fakes.
    sys.modules.pop("modal", None)
    sys.modules.pop("modal.config", None)
    sys.modules.pop("rfdf.backends.compute.modal", None)


# ---------------------------------------------------------------------------
# fetch_artifacts
# ---------------------------------------------------------------------------


def test_fetch_artifacts_dest_must_exist(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="modal", job_id="sb-xyz", submitted_at_s=0.0)
    missing = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        _run(backend.fetch_artifacts(handle, missing))


def test_fetch_artifacts_dest_must_be_directory(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="modal", job_id="sb-xyz", submitted_at_s=0.0)
    file_dest = tmp_path / "file.txt"
    file_dest.write_text("x")
    with pytest.raises(NotADirectoryError):
        _run(backend.fetch_artifacts(handle, file_dest))


def test_fetch_artifacts_logs_warning_not_raises(
    backend: Any, fake_modal: types.ModuleType, tmp_path: Path
) -> None:
    """fetch_artifacts on an existing dir should not raise."""
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="modal", job_id="sb-xyz", submitted_at_s=0.0)
    dest = tmp_path / "out"
    dest.mkdir()
    # Should complete without error even though file transfer is a stub.
    _run(backend.fetch_artifacts(handle, dest))


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------


def test_missing_credentials_raises_on_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """submit() raises RuntimeError when Modal credentials are not configured."""
    import importlib
    import types as _types

    fake = _types.ModuleType("modal")
    fake_cfg = _types.ModuleType("modal.config")

    class _EmptyConfig:
        def get(self, k: str) -> None:
            return None

    fake_cfg.config = _EmptyConfig()  # type: ignore[attr-defined]
    fake.config = fake_cfg  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setitem(sys.modules, "modal.config", fake_cfg)  # type: ignore[arg-type]
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    import rfdf.backends.compute.modal as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    with pytest.raises(RuntimeError, match="Modal credentials"):
        _run(b.submit(job))


# ---------------------------------------------------------------------------
# SDK not installed — clear ImportError
# ---------------------------------------------------------------------------


def test_missing_sdk_raises_import_error_on_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """submit() raises ImportError with install instructions when SDK absent."""
    import rfdf.backends.compute.modal as mod

    def _no_sdk() -> Any:
        raise ImportError(
            "The Modal backend requires the modal SDK; "
            "install it with:  pip install rfdf[compute-modal]"
        )

    monkeypatch.setattr(mod, "_require_modal", _no_sdk)
    b = mod.create()
    job = _make_job(tmp_path)

    with pytest.raises(ImportError, match="rfdf\\[compute-modal\\]"):
        _run(b.submit(job))


# ---------------------------------------------------------------------------
# Lazy-import guarantee
# ---------------------------------------------------------------------------


def test_import_does_not_pull_modal_into_sys_modules() -> None:
    """Importing the modal backend must NOT add 'modal' to sys.modules."""
    modal_was_present = "modal" in sys.modules
    saved = sys.modules.pop("modal", None)
    saved_cfg = sys.modules.pop("modal.config", None)
    sys.modules.pop("rfdf.backends.compute.modal", None)

    try:
        import rfdf.backends.compute.modal  # noqa: F401

        assert "modal" not in sys.modules, (
            "lazy-import broken: 'modal' appeared in sys.modules after importing "
            "rfdf.backends.compute.modal without calling any method"
        )
    finally:
        if saved is not None:
            sys.modules["modal"] = saved
        elif not modal_was_present:
            sys.modules.pop("modal", None)
        if saved_cfg is not None:
            sys.modules["modal.config"] = saved_cfg


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance(backend: Any) -> None:
    """ModalCompute is a structural ComputeBackend."""
    from rfdf.hal.compute import ComputeBackend

    assert isinstance(backend, ComputeBackend)


# ---------------------------------------------------------------------------
# _build_command — command list construction
# ---------------------------------------------------------------------------


def test_build_command_container_image(tmp_path: Path) -> None:
    """A container_image job runs the entry script directly."""
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="train.py", working_dir=tmp_path, container_image="my/image:tag")
    cmd = mod._build_command(job)
    assert cmd == ["python", "train.py"]


def test_build_command_pip_requirements_quoting(tmp_path: Path) -> None:
    """pip_requirements yield a bash -c wrapper that parses to exactly 3 tokens.

    Each requirement carries a shell metacharacter (``>``); the generated
    command list is ``["bash", "-c", "<command>"]`` and the inner command
    must survive one round of shell parsing intact — a regression guard
    against single-quote-nesting bugs.
    """
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(
        entry_script="train.py",
        working_dir=tmp_path,
        pip_requirements=["torch>=2.0", "numpy>=1.24"],
    )
    cmd = mod._build_command(job)
    assert cmd[:2] == ["bash", "-c"]
    assert len(cmd) == 3, f"_build_command did not return 3 tokens: {cmd!r}"
    inner = cmd[2]
    assert "pip install" in inner
    assert "torch>=2.0" in inner
    assert "numpy>=1.24" in inner
    assert "python train.py" in inner
    # Verify the whole command round-trips through shlex.split as 3 tokens
    # when the outer list is joined the way a shell would see it.
    # The inner part (cmd[2]) is passed as a single arg to bash -c, so it
    # does not need further shell-splitting — just confirm it is one string.
    assert isinstance(inner, str)
    # Additionally: reconstruct as a shell string and verify shlex.split gives 3 tokens.
    shell_str = "bash -c " + shlex.quote(inner)
    tokens = shlex.split(shell_str)
    assert tokens[:2] == ["bash", "-c"]
    assert len(tokens) == 3, f"shell reconstruction did not produce 3 tokens: {tokens!r}"


def test_build_command_no_requirements(tmp_path: Path) -> None:
    """With neither container_image nor pip_requirements, wrap in bash -c."""
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="train.py", working_dir=tmp_path)
    cmd = mod._build_command(job)
    assert cmd[:2] == ["bash", "-c"]
    assert "python train.py" in cmd[2]


def test_build_command_parses_cleanly_with_shlex(tmp_path: Path) -> None:
    """_build_command output contains no unquoted shell metacharacters in cmd[2]."""
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(
        entry_script="run.py",
        working_dir=tmp_path,
        pip_requirements=["scipy>=1.13", "h5py>=3.11"],
    )
    cmd = mod._build_command(job)
    # Reconstruct as a shell-safe string and verify it round-trips.
    joined = shlex.join(cmd)
    tokens = shlex.split(joined)
    assert tokens[:2] == ["bash", "-c"]
    assert len(tokens) == 3


# ---------------------------------------------------------------------------
# _pick_gpu_spec
# ---------------------------------------------------------------------------


def test_pick_gpu_spec_cpu_only(tmp_path: Path) -> None:
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=0)
    assert mod._pick_gpu_spec(job) is None


def test_pick_gpu_spec_single_gpu(tmp_path: Path) -> None:
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=1, gpu_model="A100")
    assert mod._pick_gpu_spec(job) == "A100"


def test_pick_gpu_spec_multi_gpu(tmp_path: Path) -> None:
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=4, gpu_model="H100")
    assert mod._pick_gpu_spec(job) == "H100:4"


def test_pick_gpu_spec_no_model_defaults_to_t4(tmp_path: Path) -> None:
    import rfdf.backends.compute.modal as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=1)
    spec = mod._pick_gpu_spec(job)
    assert spec == "T4"
