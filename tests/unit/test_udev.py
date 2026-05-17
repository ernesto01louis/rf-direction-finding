"""Unit tests for the udev-rules generator + installer."""

from __future__ import annotations

import pytest

from rfdf.hw.udev import (
    KNOWN_DEVICES,
    UdevRule,
    install_rules,
    render_rule,
    render_rules_file,
)


def test_udev_rule_rejects_bad_vendor_id() -> None:
    """A USB ID that is not 4 hex digits is rejected."""
    with pytest.raises(ValueError, match="4 hex digits"):
        UdevRule(vendor_id="25", product_id="0020", symlink="x_%n", description="bad")


def test_udev_rule_rejects_non_octal_mode() -> None:
    """A non-octal permission mode is rejected."""
    with pytest.raises(ValueError, match="octal"):
        UdevRule(vendor_id="2500", product_id="0020", symlink="x_%n", mode="999", description="bad")


def test_udev_rule_normalises_hex_case() -> None:
    """Hex IDs are lower-cased on validation."""
    rule = UdevRule(vendor_id="0BDA", product_id="2838", symlink="rtl_%n", description="RTL-SDR")
    assert rule.vendor_id == "0bda"


def test_render_rule_has_required_match_keys() -> None:
    """A rendered rule carries the USB match keys + permission directives."""
    rule = UdevRule(
        vendor_id="2500", product_id="0020", symlink="b210_%n", description="Ettus USRP B210"
    )
    line = render_rule(rule)
    assert 'SUBSYSTEM=="usb"' in line
    assert 'ATTRS{idVendor}=="2500"' in line
    assert 'ATTRS{idProduct}=="0020"' in line
    assert 'MODE="0666"' in line
    assert 'SYMLINK+="b210_%n"' in line
    assert "Ettus USRP B210" in line


def test_known_devices_includes_b210_and_rtlsdr() -> None:
    """The shipped device list covers the B210 and the RTL-SDR."""
    ids = {(d.vendor_id, d.product_id) for d in KNOWN_DEVICES}
    assert ("2500", "0020") in ids  # Ettus B210
    assert ("0bda", "2838") in ids  # RTL-SDR


def test_render_rules_file_has_header_and_one_line_per_device() -> None:
    """The full file carries a header and one rule per known device."""
    content = render_rules_file()
    assert content.startswith("# udev rules for rfdf")
    rule_lines = [ln for ln in content.splitlines() if ln.startswith("SUBSYSTEM==")]
    assert len(rule_lines) == len(KNOWN_DEVICES)


def test_install_rules_writes_file(tmp_path) -> None:
    """install_rules writes the content to the target path (reload disabled)."""
    target = tmp_path / "70-rfdf.rules"
    content = render_rules_file()
    summary = install_rules(content, path=target, reload=False)
    assert target.read_text(encoding="utf-8") == content
    assert str(target) in summary
