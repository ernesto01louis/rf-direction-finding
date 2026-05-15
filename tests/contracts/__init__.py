"""Hypothesis property-based contract tests.

These suites parametrize over every backend registered under the relevant
``rfdf.backends.*`` entry-point group. A new backend (e.g. Stage 5's UHD-based
B210) automatically opts in by registering its entry-point — the contract is
the only place Stage 5 needs to look to know how to satisfy the HAL.

Hardware-marked backends are filtered out at collection time so CI without
``HARDWARE_REQUIRED=1`` skips them cleanly. Stage 2 ships zero hardware
backends, so all registered backends run unconditionally.
"""
