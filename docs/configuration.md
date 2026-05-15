# Configuration

`rfdf` resolves a single `RfdfConfig` (Pydantic-settings) from four sources.
**The precedence rule is documented in exactly one place:**
[`ARCHITECTURE.md` §4](../ARCHITECTURE.md#4-configuration) — see there
for the rule and the rationale. This page is for operators:

* the on-disk TOML layout,
* the environment-variable naming convention,
* the `rfdf config` CLI surface.

## TOML layout

Default location: `$XDG_CONFIG_HOME/rfdf/config.toml` (on Linux, this resolves
to `~/.config/rfdf/config.toml`). Override via the `RFDF_CONFIG` environment
variable:

```bash
export RFDF_CONFIG=/etc/rfdf/lab.toml
rfdf config show
```

Worked example:

```toml
[default]
log_level = "info"
data_dir = "~/.local/share/rfdf"

[sdr]
backend = "mock"               # or "file-replay", "b210" (Stage 5), ...
center_freq_hz = 868e6
sample_rate_hz = 2e6
bandwidth_hz = 0               # 0 => sample_rate_hz
rx_gain_db = 30
antenna = ""                   # "" => backend default
channels = [0]
coherent = false
reference_clock = "internal"   # "external" or "gpsdo" with real hardware
timing_source = "internal"

[rotator]
backend = "mock"

[geometry]
backend = "static"             # or "mock-morph" for simulated rails
antennas = [
  [0.0,  0.0,  0.0],
  [0.17, 0.0,  0.0],
  [0.0,  0.17, 0.0],
  [-0.17, 0.0, 0.0],
  [0.0, -0.17, 0.0],
]

[compute]
backend = "local"              # remote backends land in Stage 4

[eirp]
max_eirp_dbm = 14              # 25 mW EU SRD general limit
override_explicit = false      # set true ONLY when raising max_eirp_dbm
```

## Environment variables

All env-var overrides use the `RFDF_` prefix and `__` as the nested
delimiter. Examples:

| Env var | Equivalent TOML |
|---|---|
| `RFDF_SDR__CENTER_FREQ_HZ=2400e6` | `[sdr]\ncenter_freq_hz = 2400e6` |
| `RFDF_SDR__BACKEND=file-replay` | `[sdr]\nbackend = "file-replay"` |
| `RFDF_EIRP__MAX_EIRP_DBM=20` | `[eirp]\nmax_eirp_dbm = 20` |
| `RFDF_GEOMETRY__BACKEND=mock-morph` | `[geometry]\nbackend = "mock-morph"` |

Env overrides take precedence over the TOML file but yield to CLI flags
passed explicitly. See [`ARCHITECTURE.md` §4](../ARCHITECTURE.md#4-configuration).

## CLI

### Inspect

```bash
rfdf config show              # rich table, with origin per section
rfdf config show --format=json
```

The "source" column shows where each section was resolved from:

| Source | Meaning |
|---|---|
| `default` | Hard-coded built-in defaults |
| `toml` | Loaded from `config.toml` |
| `env` | Overridden by `RFDF_*` environment variables |
| `cli` | Overridden by an explicit CLI override (programmatic — Stage 3+) |

### Validate

```bash
rfdf config validate
```

Re-resolves + re-validates the config and reports invalid fields with the
Pydantic ValidationError detail. Exit 0 on success, exit 1 on failure.
Useful as a CI gate when configs land via deploy automation.

## EIRP cap policy

`max_eirp_dbm` is enforced by `rfdf.core.eirp.requires_eirp_check` on every
TX-initiating call (currently `SdrSource.calibration_pilot`). The override
gate (`override_explicit = true`) MUST be set explicitly when a request
exceeds the cap — the flag is visible in config diffs, audit logs, and
`rfdf config show`. See [`SECURITY.md` §3](../SECURITY.md) for the
regulatory context.

The default cap (14 dBm / 25 mW) reflects the EU SRD general limit. Raise
it consciously and only when your station declaration / license covers the
new value.
