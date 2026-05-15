# Adding a backend

A new backend is five steps. The most common case — a new SDR or rotator —
ships as its own package (`rfdf-backend-myradio`) and registers via the
entry-point group `rfdf.backends.<group>`. The platform discovers it
automatically; contract tests run against it for free.

## 1. Pick a Protocol

| Group | Protocol | Examples |
|---|---|---|
| `sdr` | `SdrSource` | UHD-based B210, HackRF, SoapySDR, KrakenSDR, SigMF replay |
| `rotator` | `RotatorController` | AntRunner (HTTP), hamlib `rotctld`, SPID, Yaesu |
| `geometry` | `GeometryController` | static fixtures, motorized rails, Stewart platform |
| `compute` | `ComputeBackend` | RunPod, Modal, Vast.ai, SkyPilot |

Read the Protocol source in `src/rfdf/hal/` and the existing reference
backend in `src/rfdf/backends/<group>/`. The references are intentionally
small; copy + adapt is the expected path.

## 2. Implement the Protocol

A backend is any object that satisfies the Protocol — no inheritance required.
`@runtime_checkable` means `isinstance(obj, SdrSource)` works at runtime
too, which the contract tests use.

```python
# rfdf_backend_myradio/backend.py
class MyRadio:
    @property
    def num_channels(self) -> int: ...
    async def configure(self, config: SdrConfig) -> None: ...
    # ...
```

Optional dependencies belong inside the backend module's import block. The
base `rfdf` package guarantees zero domain-specific imports
(`zero-domain-deps` CI check), so a missing `uhd` or `pyadi-iio` must NOT
break `import rfdf`.

## 3. Register the entry-point

Either in your separate distribution's `pyproject.toml`:

```toml
[project.entry-points."rfdf.backends.sdr"]
myradio = "rfdf_backend_myradio.backend:create"
```

…or, for a contribution, in the main repo's `pyproject.toml`. The
``create(**kwargs)`` factory must return a configured backend instance.

After installation, ``rfdf hw list-backends`` shows the new entry.

## 4. Run the contract tests

`tests/contracts/` parametrizes over every registered entry-point. Once
your backend is installed (`pip install -e .`), the suites pick it up:

```bash
pytest tests/contracts -v
```

Hardware-only backends register their identifier in the `hardware_only` set
in `tests/contracts/conftest.py` so they're skipped in CI without
`HARDWARE_REQUIRED=1`.

If a contract test fails, the backend does NOT satisfy the protocol; fix
the implementation, not the test.

## 5. Document the backend

Add a row to the relevant table in `docs/hal.md`. Note any caveats (single-
channel only, GPU-required, etc.). For hardware backends, document the
required OS packages + permissions in your distribution's README.

## Edge cases the discovery layer handles for you

The catalog (`discover_backends`) tolerates:

* **Broken `.load()`** (missing optional dep): WARN + skip. Your backend
  package can declare optional deps cleanly without crashing the platform.
* **Non-callable target**: WARN + skip.
* **Duplicate name**: first-wins, WARN logs both distributions. This is the
  expected behaviour when editable + wheel installs of the same project
  shadow each other.
* **Factory raises**: `load_backend` wraps in `BackendLoadError` with
  `__cause__` preserved. Operators see the backend identity in the
  traceback chain.

You don't need to defend against any of these in your factory — the
discovery layer absorbs the friction.
