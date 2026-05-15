"""Stage 1 placeholder.

The `test-demo-no-hardware` CI job runs against this directory. Stage 2 ships a
mock-SDR pipeline smoke test here that wires the HAL through to a stubbed DOA
call; Stage 3 extends it to CRLB-bounded MUSIC + ESPRIT assertions.

For Stage 1, we assert the suite *exists* and the runner works. Replacing this
placeholder with the real pipeline test is a Stage 2 deliverable.
"""


def test_stage1_placeholder() -> None:
    """The demo-no-hardware CI job must have at least one passing test."""
    assert True, "Stage 2 replaces this with a real mock-SDR pipeline smoke test."
