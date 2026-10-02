"""Basic host identity used before full telemetry arrives in M7."""

from __future__ import annotations

import os
import platform
import socket
from pathlib import Path
from dataclasses import dataclass

from llama_profile_lab.domain import sha256_json


@dataclass(frozen=True, slots=True)
class BasicHostInfo:
    """Minimal reproducibility metadata sufficient for M5 run provenance."""

    hostname: str
    hardware_fingerprint: str
    cpu: dict[str, object]
    ram_bytes: int
    gpus: list[dict[str, object]]
    os_info: dict[str, object]


def detect_basic_host() -> BasicHostInfo:
    """Capture stable host metadata without vendor-specific telemetry tools."""
    cpu: dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    ram_bytes = _total_ram_bytes()
    os_info: dict[str, object] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
    }
    gpus = _linux_gpu_inventory()
    fingerprint_payload = {
        "cpu": cpu,
        "ram_bytes": ram_bytes,
        "os_system": os_info["system"],
        "machine": cpu["machine"],
        "gpus": gpus,
    }
    return BasicHostInfo(
        hostname=socket.gethostname(),
        hardware_fingerprint=sha256_json(fingerprint_payload),
        cpu=cpu,
        ram_bytes=ram_bytes,
        gpus=gpus,
        os_info=os_info,
    )


def _total_ram_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return 0
    return max(0, pages * page_size)



def _linux_gpu_inventory() -> list[dict[str, object]]:
    """Capture stable DRM/PCI GPU identity without vendor tooling."""
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        return []

    gpus: list[dict[str, object]] = []
    for card in sorted(drm.glob("card[0-9]*"), key=lambda path: path.name):
        device = card / "device"
        if not device.exists():
            continue

        entry: dict[str, object] = {
            "drm_card": card.name,
            "pci_address": device.resolve().name,
        }
        for name in ("vendor", "device", "subsystem_vendor", "subsystem_device"):
            value = _read_text(device / name)
            if value is not None:
                entry[name] = value

        driver = device / "driver"
        if driver.exists():
            entry["driver"] = driver.resolve().name

        gpus.append(entry)
    return gpus


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
