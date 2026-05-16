"""Tests for LocalCompute container-image (``docker run``) execution.

The docker-command construction and the docker-absent error path are unit-
tested directly. A real ``docker run`` end-to-end test is marked ``integration``
and only runs when ``RFDF_DOCKER_IT`` is set (with the docker CLI available) —
this keeps it out of normal CI, where pulling an image is slow and a network
flake would be a recurring liability across every later PR's CI run.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from rfdf.backends.compute.local import LocalCompute, _build_docker_command
from rfdf.hal.compute import ComputeJob
from rfdf.hal.types import JobStatus


def _container_job(working_dir: Path, **overrides: object) -> ComputeJob:
    """Build a container-image ComputeJob for tests."""
    params: dict[str, object] = {
        "entry_script": "run.py",
        "container_image": "python:3.11-slim",
        "working_dir": working_dir,
    }
    params.update(overrides)
    return ComputeJob(**params)


def test_build_docker_command_basic(tmp_path: Path) -> None:
    """The docker argv mounts working_dir at /workspace and runs the entry script."""
    cmd = _build_docker_command(_container_job(tmp_path))
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert f"{tmp_path.resolve()}:/workspace" in cmd
    assert cmd[cmd.index("-w") + 1] == "/workspace"
    assert cmd[-3:] == ["python:3.11-slim", "python", "run.py"]


def test_build_docker_command_env(tmp_path: Path) -> None:
    """job.env entries are forwarded as -e KEY=VALUE flags."""
    cmd = _build_docker_command(_container_job(tmp_path, env={"FOO": "bar"}))
    assert "-e" in cmd
    assert "FOO=bar" in cmd


def test_build_docker_command_gpu(tmp_path: Path) -> None:
    """A GPU request adds --gpus all; a CPU job does not."""
    gpu_cmd = _build_docker_command(_container_job(tmp_path, gpu_count=1))
    assert gpu_cmd[gpu_cmd.index("--gpus") + 1] == "all"
    assert "--gpus" not in _build_docker_command(_container_job(tmp_path))


def test_submit_container_without_docker_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """submit() of a container job raises a clear error when docker is absent."""
    (tmp_path / "run.py").write_text("print('hi')\n")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    backend = LocalCompute()
    with pytest.raises(RuntimeError, match="docker"):
        asyncio.run(backend.submit(_container_job(tmp_path)))


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RFDF_DOCKER_IT") or shutil.which("docker") is None,
    reason="set RFDF_DOCKER_IT=1 with the docker CLI available to run this",
)
def test_submit_container_runs_real(tmp_path: Path) -> None:
    """A real container job runs the entry script through to completion."""
    (tmp_path / "run.py").write_text("print('container-ran-ok')\n")
    backend = LocalCompute()
    job = _container_job(tmp_path)

    async def _run() -> tuple[JobStatus, list[str]]:
        handle = await backend.submit(job)
        lines = [line async for line in backend.logs(handle)]
        return await backend.status(handle), lines

    status, lines = asyncio.run(_run())
    assert status == JobStatus.COMPLETED
    assert any("container-ran-ok" in line for line in lines)
