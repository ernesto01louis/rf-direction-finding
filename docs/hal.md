# Hardware abstraction layer

The HAL is the contract every backend implements. Four Protocol classes live
in `src/rfdf/hal/`; concrete backends live in `src/rfdf/backends/<group>/` and
register via `pyproject.toml` entry-points (see
[adding-a-backend.md](adding-a-backend.md)).

All Protocol methods at the I/O boundary are `async`. Internal DSP code
(Stage 3+) is sync NumPy/SciPy; the boundary crosses via `asyncio.to_thread`.

## SdrSource — IQ capture

```python
class SdrSource(Protocol):
    @property
    def num_channels(self) -> int: ...
    @property
    def supports_coherent(self) -> bool: ...
    @property
    def tuning_range_hz(self) -> tuple[float, float]: ...
    @property
    def max_sample_rate_hz(self) -> float: ...

    async def configure(self, config: SdrConfig) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def stream(self) -> AsyncIterator[StreamBlock]: ...
    async def capture(self, duration_s: float) -> Recording: ...
    async def status(self) -> dict[str, object]: ...
    async def calibration_pilot(self, freq_hz: float, power_dbm: float) -> None: ...
    async def close(self) -> None: ...
```

Lifetime: `configure → start → stream/capture → stop → close`. The
`calibration_pilot` call is gated by the EIRP cap (see
[configuration.md](configuration.md)); replay-only backends raise
`NotImplementedError`.

`status()` (added in `v0.1.0-alpha`, Stage 5) returns a free-form
device-health mapping — hardware backends report reference-clock / GPSDO lock,
USB topology, board temperatures; synthetic backends report whatever is
meaningful. Every key is optional except `"backend"`; callers (notably
`rfdf hw selftest`) probe with `.get(...)`. It is safe to call at any point in
the lifecycle, including before `configure()`.

| Backend | Channels | Pilot tone | Notes |
|---|---|---|---|
| `mock` | configurable (default 5) | yes | Synthetic emitters with array-factor signal model. |
| `file-replay` | 1 | no | SigMF playback. Multi-channel deferred to Stage 5. |

## RotatorController — mechanical AZ/EL

```python
class RotatorController(Protocol):
    @property
    def supports_azimuth(self) -> bool: ...
    @property
    def supports_elevation(self) -> bool: ...
    @property
    def azimuth_range_deg(self) -> tuple[float, float]: ...
    @property
    def elevation_range_deg(self) -> tuple[float, float]: ...
    @property
    def max_speed_deg_per_s(self) -> float: ...
    @property
    def positioning_accuracy_deg(self) -> float: ...

    async def goto(self, azimuth_deg: float, elevation_deg: float) -> None: ...
    async def park(self) -> None: ...
    async def stop(self) -> None: ...
    async def position(self) -> tuple[float, float]: ...
    def stream_position(self) -> AsyncIterator[tuple[float, float]]: ...
    async def calibrate(self) -> CalibrationReport: ...
```

| Backend | Notes |
|---|---|
| `mock` | Constant-velocity slew + Gaussian post-settle noise. Stream-position yields linearly-interpolated samples. |

## GeometryController — per-antenna 3D position

```python
class GeometryController(Protocol):
    @property
    def num_antennas(self) -> int: ...
    @property
    def is_morphable(self) -> bool: ...
    @property
    def positioning_repeatability_mm(self) -> float: ...

    async def positions(self) -> np.ndarray: ...
    async def goto_preset(self, preset_name: str) -> None: ...
    async def goto_positions(self, positions: np.ndarray) -> None: ...
    async def list_presets(self) -> list[str]: ...
    async def save_preset(self, name: str, positions: np.ndarray) -> None: ...
    async def calibrate(self) -> CalibrationReport: ...
```

| Backend | Morphable | Notes |
|---|---|---|
| `static` | no | Fixed `(x, y, z)` list. Morphing methods raise. |
| `mock-morph` | yes | Simulated motorized rails with Gaussian repeatability noise; TOML-backed presets. |

## ComputeBackend — ML / batch dispatch

```python
class ComputeBackend(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def supports_gpu(self) -> bool: ...
    @property
    def supports_persistent_storage(self) -> bool: ...

    async def submit(self, job: ComputeJob) -> JobHandle: ...
    async def status(self, handle: JobHandle) -> JobStatus: ...
    def logs(self, handle: JobHandle) -> AsyncIterator[str]: ...
    async def fetch_artifacts(self, handle: JobHandle, dest: Path) -> None: ...
    async def cancel(self, handle: JobHandle) -> None: ...
    async def cost_estimate(self, job: ComputeJob) -> CostEstimate: ...
```

| Backend | GPU | Persistent storage | Notes |
|---|---|---|---|
| `local` | depends on host (`nvidia-smi` probe) | yes | Subprocess execution. `container_image` deferred to Stage 4. |

## Contract guarantees

Every backend conforming to a Protocol MUST satisfy the contract tests in
`tests/contracts/`. Property-based suites (Hypothesis) parametrize over each
HAL group's registered entry-points; backend authors get free coverage just by
registering.

Failures in contract tests are blocking — a backend that doesn't pass cannot
be used by the algorithm layer (Stage 3+).

## Discovery

```python
from rfdf.hal import discover_backends, list_backends, load_backend

list_backends()
# {'rfdf.backends.sdr': ['file-replay', 'mock'], ...}

sdr = load_backend("rfdf.backends.sdr", "mock", block_samples=4096)
```

Edge cases — broken `.load()`, non-callable target, duplicate names — log
WARN and skip; they never raise out of `discover_backends`. Factory exceptions
raised inside `load_backend` are wrapped in `BackendLoadError` with `__cause__`
preserved.
