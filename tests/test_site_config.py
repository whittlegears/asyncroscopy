"""Lab machine addresses stay out of git: configs use ${NAME} placeholders."""

import ipaddress
import re
import subprocess
from pathlib import Path

import pytest

from startup_scripts.config_schema import ConfigError, fill_placeholders, site_values

ROOT = Path(__file__).parents[1]
IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
# Binary or generated files that cannot hold a hand-written address.
SKIP = re.compile(r"(^stubs/|^uv\.lock$|\.(png|jpg|jpeg|gif|pdf|h5|whl|db|zip)$)")


def _address(text: str):
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None  # not an address, e.g. a version such as 1.15.0.484


def _allowed(address) -> bool:
    """Loopback, "any interface" and the documentation ranges are fine to publish."""
    return address.is_loopback or address.is_unspecified or any(
        address in ipaddress.ip_network(net) for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
    )


def test_no_lab_addresses_in_tracked_files():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    found = []
    for name in tracked:
        if SKIP.search(name) or not (ROOT / name).is_file():
            continue
        text = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        for match in IPV4.finditer(text):
            address = _address(match.group(1))
            # Private addresses anywhere; any address at all in the configs.
            if address and not _allowed(address) and (address.is_private or name.startswith("configs/")):
                found.append(f"{name}: {address}")
    assert not found, "Put addresses in configs/site.env, not in git: " + ", ".join(found)


def test_placeholders_are_filled_from_the_site_file_and_the_environment(tmp_path, monkeypatch):
    site = tmp_path / "site.env"
    site.write_text('# comment\nASYNCROSCOPY_A=192.0.2.1\nASYNCROSCOPY_B="192.0.2.2"\n')
    monkeypatch.setenv("ASYNCROSCOPY_B", "192.0.2.3")  # the environment wins

    values = site_values(site)
    raw = {"host": "${ASYNCROSCOPY_A}", "devices": [{"url": "http://${ASYNCROSCOPY_B}:9091"}], "port": 9094}

    assert fill_placeholders(raw, values) == {"host": "192.0.2.1", "devices": [{"url": "http://192.0.2.3:9091"}], "port": 9094}


def test_a_missing_value_is_named_in_the_error():
    with pytest.raises(ConfigError, match="ASYNCROSCOPY_NOT_SET"):
        fill_placeholders({"host": "${ASYNCROSCOPY_NOT_SET}"}, values={})


def test_a_malformed_site_line_is_refused(tmp_path):
    site = tmp_path / "site.env"
    site.write_text("just a host name\n")
    with pytest.raises(ConfigError, match="line 1"):
        site_values(site)


def test_the_example_lists_every_name_the_configs_use():
    used = set()
    for config in (ROOT / "configs").glob("*.yaml"):
        used |= set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", config.read_text()))
    example = (ROOT / "configs" / "site.env.example").read_text()
    listed = set(re.findall(r"^([A-Z_][A-Z0-9_]*)=", example, re.M))
    assert used <= listed
