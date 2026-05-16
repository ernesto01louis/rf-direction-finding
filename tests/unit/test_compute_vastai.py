"""Unit tests for the Vast.ai compute backend.

All Vast.ai SDK calls are mocked via a fake ``vastai`` module injected into
``sys.modules``.  No real network calls are made; the unit suite must pass
without the ``vastai`` SDK installed (or with it installed but no key set).
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
# Fake Vast.ai SDK helpers
# ---------------------------------------------------------------------------


def _make_fake_instance(
    *,
    instance_id: int = 12345,
    actual_status: str = "running",
    dph_total: float = 0.20,
) -> dict[str, Any]:
    """Build a fake Vast.ai instance dict."""
    return {
        "id": instance_id,
        "actual_status": actual_status,
        "dph_total": dph_total,
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "ssh_host": "1.2.3.4",
        "ssh_port": 22222,
        "label": "rfdf-test",
    }


def _make_fake_offer(*, offer_id: int = 999, dph_total: float = 0.18) -> dict[str, Any]:
    """Build a fake Vast.ai offer dict."""
    return {
        "id": offer_id,
        "dph_total": dph_total,
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "rentable": True,
        "verified": True,
    }


def _make_fake_vastai(**overrides: Any) -> types.ModuleType:
    """Build a minimal fake ``vastai`` module for mocking.

    Args:
        **overrides: Replace named attributes on the fake module.

    Returns:
        A ``types.ModuleType`` that looks enough like ``vastai`` for the
        backend to call without errors.
    """
    fake = types.ModuleType("vastai")

    _default_offer = _make_fake_offer()
    _default_instance = _make_fake_instance()

    class _FakeVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            self.api_key = api_key

        def search_offers(
            self, query=None, type="on-demand", order="dph_total", limit=None, **kwargs
        ):
            return [_default_offer]

        def create_instance(
            self,
            id,
            image=None,
            disk=10,
            env=None,
            onstart_cmd=None,
            label=None,
            cancel_unavail=False,
            **kwargs,
        ):
            return {"new_contract": _default_instance["id"], "id": _default_instance["id"]}

        def show_instance(self, id):
            return dict(_default_instance, id=id)

        def destroy_instance(self, id):
            return {"success": True}

        def logs(self, instance_id, tail=None, filter=None, daemon_logs=False):
            return f"[log] instance {instance_id} output\n"

    fake.VastAI = _FakeVastAI  # type: ignore[attr-defined]

    # _base submodule — needed for _resolve_api_key used by VastAI.__init__
    fake_base = types.ModuleType("vastai._base")
    _SENTINEL = object()
    fake_base._APIKEY_SENTINEL = _SENTINEL  # type: ignore[attr-defined]

    def _resolve_api_key(key):
        if key is _SENTINEL:
            return "fake-resolved-key"
        return key

    fake_base._resolve_api_key = _resolve_api_key  # type: ignore[attr-defined]
    fake._base = fake_base  # type: ignore[attr-defined]

    for k, v in overrides.items():
        setattr(fake, k, v)
    return fake


@pytest.fixture()
def fake_vastai(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake vastai module and set VAST_API_KEY in the environment.

    Returns the fake module so tests can configure it further.
    """
    fake = _make_fake_vastai()
    monkeypatch.setitem(sys.modules, "vastai", fake)
    monkeypatch.setitem(sys.modules, "vastai._base", fake._base)  # type: ignore[attr-defined]
    monkeypatch.setenv("VAST_API_KEY", "test-key-abc")

    import importlib

    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    return fake


@pytest.fixture()
def backend(fake_vastai: types.ModuleType) -> Any:
    """Return a VastAiCompute instance wired to the fake SDK."""
    import rfdf.backends.compute.vastai as mod

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
    assert backend.name == "vastai"


def test_supports_gpu(backend: Any) -> None:
    assert backend.supports_gpu is True


def test_supports_persistent_storage(backend: Any) -> None:
    # Vast.ai uses per-instance local disk — not a shared persistent volume.
    assert backend.supports_persistent_storage is False


def test_factory_returns_vastai_compute(fake_vastai: types.ModuleType) -> None:
    import rfdf.backends.compute.vastai as mod

    b = mod.create()
    assert b.name == "vastai"


# ---------------------------------------------------------------------------
# Cost estimate arithmetic (SDK-free)
# ---------------------------------------------------------------------------


def test_cost_estimate_1x_to_2x_spread(backend: Any, tmp_path: Path) -> None:
    """low <= estimated <= high and estimated = 1.5x low."""
    job = _make_job(tmp_path, gpu_count=1, gpu_model=None)
    est = _run(backend.cost_estimate(job))
    assert est.backend == "vastai"
    assert est.low_usd <= est.estimated_usd <= est.high_usd
    assert abs(est.estimated_usd - 1.5 * est.low_usd) < 1e-9
    assert abs(est.high_usd - 2.0 * est.low_usd) < 1e-9


def test_cost_estimate_known_gpu_model(backend: Any, tmp_path: Path) -> None:
    """A known GPU model uses the correct hourly rate from _RATES."""
    import rfdf.backends.compute.vastai as mod

    job = _make_job(tmp_path, gpu_count=2, gpu_model="RTX 4090")
    est = _run(backend.cost_estimate(job))
    # RTX 4090 rate = 0.35, timeout 1h, 2 GPUs
    expected_low = mod._RATES["RTX 4090"] * 1.0 * (3600.0 / 3600.0) * 2
    assert abs(est.low_usd - expected_low) < 1e-9
    assert est.rationale  # must be non-empty


def test_cost_estimate_fallback_rate(backend: Any, tmp_path: Path) -> None:
    """Unknown GPU model falls back to _RATES[None]."""
    import rfdf.backends.compute.vastai as mod

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
    """cost_estimate works without the vastai SDK in sys.modules."""
    saved = sys.modules.pop("vastai", None)
    saved_base = sys.modules.pop("vastai._base", None)
    sys.modules.pop("rfdf.backends.compute.vastai", None)
    try:
        import importlib

        import rfdf.backends.compute.vastai as mod

        importlib.reload(mod)
        b = mod.create()
        (tmp_path / "s.py").write_text("pass\n")
        from rfdf.hal.compute import ComputeJob

        job = ComputeJob(entry_script="s.py", working_dir=tmp_path, timeout_s=1800.0)
        est = _run(b.cost_estimate(job))
        assert est.backend == "vastai"
        assert est.low_usd <= est.estimated_usd <= est.high_usd
    finally:
        if saved is not None:
            sys.modules["vastai"] = saved
        if saved_base is not None:
            sys.modules["vastai._base"] = saved_base


def test_cost_estimate_1_5x_factor(backend: Any, tmp_path: Path) -> None:
    """Confirm 1.5x factor: estimated is exactly between low and high."""
    import rfdf.backends.compute.vastai as mod

    job = _make_job(tmp_path, gpu_count=1, gpu_model="A100 SXM4")
    est = _run(backend.cost_estimate(job))
    runtime_h = job.timeout_s / 3600.0
    rate = mod._RATES["A100 SXM4"]
    expected_estimated = rate * 1.5 * runtime_h * 1
    expected_low = rate * 1.0 * runtime_h * 1
    expected_high = rate * 2.0 * runtime_h * 1
    assert abs(est.estimated_usd - expected_estimated) < 1e-9
    assert abs(est.low_usd - expected_low) < 1e-9
    assert abs(est.high_usd - expected_high) < 1e-9


# ---------------------------------------------------------------------------
# submit / status / cancel
# ---------------------------------------------------------------------------


def test_submit_returns_vastai_handle(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    assert handle.backend == "vastai"
    assert handle.job_id  # non-empty string
    assert handle.submitted_at_s > 0


def test_submit_sets_extra_metadata(backend: Any, tmp_path: Path) -> None:
    job = _make_job(tmp_path, gpu_count=1, gpu_model="RTX 4090")
    handle = _run(backend.submit(job))
    assert "offer_id" in handle.extra
    assert "label" in handle.extra
    assert "image" in handle.extra
    assert "query" in handle.extra


def test_submit_raises_when_no_offers(fake_vastai: types.ModuleType, tmp_path: Path) -> None:
    """submit() raises RuntimeError when no matching offers exist."""
    import importlib

    class _EmptyVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def search_offers(self, **kwargs):
            return []  # no offers

    fake_vastai.VastAI = _EmptyVastAI  # type: ignore[attr-defined]
    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    with pytest.raises(RuntimeError, match=r"no Vast\.ai offers"):
        _run(b.submit(job))


def test_status_running(backend: Any, fake_vastai: types.ModuleType, tmp_path: Path) -> None:
    from rfdf.hal.types import JobStatus

    job = _make_job(tmp_path)
    handle = _run(backend.submit(job))
    # show_instance returns actual_status="running" by default.
    status = _run(backend.status(handle))
    assert status == JobStatus.RUNNING


def test_status_completed(backend: Any, fake_vastai: types.ModuleType, tmp_path: Path) -> None:
    import importlib

    from rfdf.hal.types import JobStatus

    class _ExitedVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def search_offers(self, **kwargs):
            return [_make_fake_offer()]

        def create_instance(self, *a, **kw):
            return {"new_contract": 12345, "id": 12345}

        def show_instance(self, id):
            return _make_fake_instance(instance_id=id, actual_status="exited")

        def destroy_instance(self, id):
            return {}

        def logs(self, *a, **kw):
            return ""

    fake_vastai.VastAI = _ExitedVastAI  # type: ignore[attr-defined]
    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))
    status = _run(b.status(handle))
    assert status == JobStatus.COMPLETED


def test_status_failed(backend: Any, fake_vastai: types.ModuleType, tmp_path: Path) -> None:
    import importlib

    from rfdf.hal.types import JobStatus

    class _ErrorVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def search_offers(self, **kwargs):
            return [_make_fake_offer()]

        def create_instance(self, *a, **kw):
            return {"new_contract": 12345, "id": 12345}

        def show_instance(self, id):
            return _make_fake_instance(instance_id=id, actual_status="error")

        def destroy_instance(self, id):
            return {}

        def logs(self, *a, **kw):
            return ""

    fake_vastai.VastAI = _ErrorVastAI  # type: ignore[attr-defined]
    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))
    status = _run(b.status(handle))
    assert status == JobStatus.FAILED


def test_status_instance_lookup_failure_returns_cancelled(
    backend: Any, fake_vastai: types.ModuleType, tmp_path: Path
) -> None:
    """show_instance raising an exception maps to CANCELLED."""
    import importlib

    from rfdf.hal.compute import JobHandle
    from rfdf.hal.types import JobStatus

    class _BrokenVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def show_instance(self, id):
            raise RuntimeError("not found")

    fake_vastai.VastAI = _BrokenVastAI  # type: ignore[attr-defined]
    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    b = mod.create()
    handle = JobHandle(backend="vastai", job_id="99999", submitted_at_s=0.0)
    status = _run(b.status(handle))
    assert status == JobStatus.CANCELLED


def test_cancel_destroys_instance(
    backend: Any, fake_vastai: types.ModuleType, tmp_path: Path
) -> None:
    destroyed: list[int] = []

    class _SpyVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def search_offers(self, **kwargs):
            return [_make_fake_offer()]

        def create_instance(self, *a, **kw):
            return {"new_contract": 12345}

        def show_instance(self, id):
            return _make_fake_instance(instance_id=id, actual_status="running")

        def destroy_instance(self, id):
            destroyed.append(id)
            return {}

        def logs(self, *a, **kw):
            return ""

    import importlib

    fake_vastai.VastAI = _SpyVastAI  # type: ignore[attr-defined]
    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))
    _run(b.cancel(handle))
    assert destroyed  # destroy_instance was called


def test_cancel_noop_when_already_cancelled(
    backend: Any, fake_vastai: types.ModuleType, tmp_path: Path
) -> None:
    """cancel() on a destroyed instance does not call destroy_instance again."""
    destroy_calls: list[int] = []

    class _DestroyedVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def search_offers(self, **kwargs):
            return [_make_fake_offer()]

        def create_instance(self, *a, **kw):
            return {"new_contract": 12345}

        def show_instance(self, id):
            return _make_fake_instance(instance_id=id, actual_status="destroyed")

        def destroy_instance(self, id):
            destroy_calls.append(id)
            return {}

        def logs(self, *a, **kw):
            return ""

    import importlib

    fake_vastai.VastAI = _DestroyedVastAI  # type: ignore[attr-defined]
    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    handle = _run(b.submit(job))
    _run(b.cancel(handle))
    assert destroy_calls == []  # not called because status is CANCELLED


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def test_logs_yields_status_lines(
    backend: Any, fake_vastai: types.ModuleType, tmp_path: Path
) -> None:
    """logs() yields at least one line and exits when the instance has exited."""
    import importlib

    class _ExitingVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def search_offers(self, **kwargs):
            return [_make_fake_offer()]

        def create_instance(self, *a, **kw):
            return {"new_contract": 12345}

        def show_instance(self, id):
            return _make_fake_instance(instance_id=id, actual_status="exited")

        def logs(self, instance_id, tail=None, filter=None, daemon_logs=False):
            return f"[log] output from {instance_id}"

    fake_vastai.VastAI = _ExitingVastAI  # type: ignore[attr-defined]
    import rfdf.backends.compute.vastai as mod

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
    assert any("exited" in line or "vastai" in line.lower() for line in lines)


def test_logs_yields_error_line_on_lookup_failure(tmp_path: Path) -> None:
    """If show_instance raises, logs() yields an error line and returns."""
    import importlib
    import types as _types

    fake = _types.ModuleType("vastai")

    class _BrokenVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            pass

        def show_instance(self, id):
            raise RuntimeError("lookup boom")

    fake.VastAI = _BrokenVastAI  # type: ignore[attr-defined]
    sys.modules["vastai"] = fake
    sys.modules.pop("rfdf.backends.compute.vastai", None)

    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)

    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="vastai", job_id="99999", submitted_at_s=0.0)

    async def collect() -> list[str]:
        lines: list[str] = []
        async for line in mod._poll_instance_logs(handle):
            lines.append(line)
        return lines

    lines = _run(collect())
    assert any("failed" in line or "boom" in line for line in lines)

    sys.modules.pop("vastai", None)
    sys.modules.pop("rfdf.backends.compute.vastai", None)


# ---------------------------------------------------------------------------
# fetch_artifacts
# ---------------------------------------------------------------------------


def test_fetch_artifacts_dest_must_exist(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="vastai", job_id="12345", submitted_at_s=0.0)
    missing = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        _run(backend.fetch_artifacts(handle, missing))


def test_fetch_artifacts_dest_must_be_directory(backend: Any, tmp_path: Path) -> None:
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="vastai", job_id="12345", submitted_at_s=0.0)
    file_dest = tmp_path / "file.txt"
    file_dest.write_text("x")
    with pytest.raises(NotADirectoryError):
        _run(backend.fetch_artifacts(handle, file_dest))


def test_fetch_artifacts_logs_warning_not_raises(
    backend: Any, fake_vastai: types.ModuleType, tmp_path: Path
) -> None:
    """fetch_artifacts on an existing dir should not raise."""
    from rfdf.hal.compute import JobHandle

    handle = JobHandle(backend="vastai", job_id="12345", submitted_at_s=0.0)
    dest = tmp_path / "out"
    dest.mkdir()
    _run(backend.fetch_artifacts(handle, dest))


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_on_submit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """submit() raises RuntimeError when no Vast.ai API key is available."""
    import importlib

    class _KeyErrorVastAI:
        def __init__(self, api_key=None, raw=False, quiet=False, **kwargs):
            raise RuntimeError(
                "No API key found. Pass api_key=, set VAST_API_KEY, "
                "or save your key to ~/.config/vastai/vast_api_key"
            )

    fake = _make_fake_vastai()
    fake.VastAI = _KeyErrorVastAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vastai", fake)
    monkeypatch.delenv("VAST_API_KEY", raising=False)

    import rfdf.backends.compute.vastai as mod

    importlib.reload(mod)
    b = mod.create()
    job = _make_job(tmp_path)
    with pytest.raises(RuntimeError, match="API key"):
        _run(b.submit(job))


# ---------------------------------------------------------------------------
# SDK not installed — clear ImportError
# ---------------------------------------------------------------------------


def test_missing_sdk_raises_import_error_on_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """submit() raises ImportError with install instructions when SDK absent."""
    import rfdf.backends.compute.vastai as mod

    def _no_sdk() -> Any:
        raise ImportError(
            "The Vast.ai backend requires the vastai SDK; "
            "install it with:  pip install rfdf[compute-vastai]"
        )

    monkeypatch.setattr(mod, "_require_vastai", _no_sdk)
    b = mod.create()
    job = _make_job(tmp_path)
    with pytest.raises(ImportError, match="rfdf\\[compute-vastai\\]"):
        _run(b.submit(job))


# ---------------------------------------------------------------------------
# Lazy-import guarantee
# ---------------------------------------------------------------------------


def test_import_does_not_pull_vastai_into_sys_modules() -> None:
    """Importing the vastai backend must NOT add 'vastai' to sys.modules."""
    vastai_was_present = "vastai" in sys.modules
    saved = sys.modules.pop("vastai", None)
    saved_base = sys.modules.pop("vastai._base", None)
    sys.modules.pop("rfdf.backends.compute.vastai", None)

    try:
        import rfdf.backends.compute.vastai  # noqa: F401

        assert "vastai" not in sys.modules, (
            "lazy-import broken: 'vastai' appeared in sys.modules after importing "
            "rfdf.backends.compute.vastai without calling any method"
        )
    finally:
        if saved is not None:
            sys.modules["vastai"] = saved
        elif not vastai_was_present:
            sys.modules.pop("vastai", None)
        if saved_base is not None:
            sys.modules["vastai._base"] = saved_base


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance(backend: Any) -> None:
    """VastAiCompute is a structural ComputeBackend."""
    from rfdf.hal.compute import ComputeBackend

    assert isinstance(backend, ComputeBackend)


# ---------------------------------------------------------------------------
# _build_onstart_cmd — command string construction + shlex quoting guard
# ---------------------------------------------------------------------------


def test_build_onstart_cmd_container_image(tmp_path: Path) -> None:
    """A container_image job runs the entry script directly (no bash wrapper)."""
    import rfdf.backends.compute.vastai as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="train.py", working_dir=tmp_path, container_image="my/image:tag")
    assert mod._build_onstart_cmd(job) == "python train.py"


def test_build_onstart_cmd_pip_requirements_quoting(tmp_path: Path) -> None:
    """pip_requirements yield a bash -c wrapper that parses to exactly 3 tokens.

    Each requirement carries a shell metacharacter (``>``); the generated
    command string must survive one round of shell parsing as exactly three
    tokens (``bash``, ``-c``, ``<command>``) — a regression guard against
    single-quote-nesting bugs.
    """
    import rfdf.backends.compute.vastai as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(
        entry_script="train.py",
        working_dir=tmp_path,
        pip_requirements=["torch>=2.0", "numpy>=1.24"],
    )
    cmd_str = mod._build_onstart_cmd(job)
    tokens = shlex.split(cmd_str)
    assert tokens[:2] == ["bash", "-c"], f"Expected bash -c prefix; got {tokens!r}"
    assert len(tokens) == 3, f"Expected 3 tokens from shlex.split; got {tokens!r}"
    inner = tokens[2]
    assert "pip install" in inner
    assert "torch>=2.0" in inner
    assert "numpy>=1.24" in inner
    assert "python train.py" in inner


def test_build_onstart_cmd_no_requirements(tmp_path: Path) -> None:
    """With neither container_image nor pip_requirements, wrap in bash -c."""
    import rfdf.backends.compute.vastai as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="train.py", working_dir=tmp_path)
    cmd_str = mod._build_onstart_cmd(job)
    tokens = shlex.split(cmd_str)
    assert tokens[:2] == ["bash", "-c"]
    assert "python train.py" in tokens[2]


def test_build_onstart_cmd_parses_cleanly_with_special_chars(tmp_path: Path) -> None:
    """Metacharacters in requirements do not break the single-quoted bash -c wrapper."""
    import rfdf.backends.compute.vastai as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(
        entry_script="run.py",
        working_dir=tmp_path,
        pip_requirements=["scipy>=1.13", "h5py>=3.11", "torch>=2.1,<3.0"],
    )
    cmd_str = mod._build_onstart_cmd(job)
    tokens = shlex.split(cmd_str)
    assert len(tokens) == 3, f"shlex.split should yield 3 tokens; got {tokens!r}"
    assert tokens[0] == "bash"
    assert tokens[1] == "-c"


# ---------------------------------------------------------------------------
# _build_search_query
# ---------------------------------------------------------------------------


def test_build_search_query_gpu_model(tmp_path: Path) -> None:
    import rfdf.backends.compute.vastai as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=2, gpu_model="RTX 4090")
    q = mod._build_search_query(job)
    assert "num_gpus=2" in q
    assert "RTX_4090" in q


def test_build_search_query_min_vram(tmp_path: Path) -> None:
    import rfdf.backends.compute.vastai as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=1, gpu_min_vram_gb=24)
    q = mod._build_search_query(job)
    assert "gpu_ram>=24" in q


def test_build_search_query_no_gpu(tmp_path: Path) -> None:
    import rfdf.backends.compute.vastai as mod
    from rfdf.hal.compute import ComputeJob

    job = ComputeJob(entry_script="r.py", working_dir=tmp_path, gpu_count=0)
    q = mod._build_search_query(job)
    # No explicit count → fallback to >=1
    assert "num_gpus>=1" in q
