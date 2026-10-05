"""llama.cpp logical-device inventory parsing and physical correlation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_profile_lab.domain import AcceleratorDevice, physical_device_key
from llama_profile_lab.llama.capabilities import CapabilitySet

_MIB = 1024 * 1024
_DEVICE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9_-]*\d+)\s*:\s*"
    r"(?P<product>.+?)"
    r"(?:\s+\((?P<total>\d+)\s+MiB,\s*"
    r"(?P<free>\d+)\s+MiB\s+free\))?\s*$"
)
_BACKEND_RE = re.compile(r"^(?P<backend>[A-Za-z][A-Za-z0-9_-]*?)(?:\d+)$")


class LlamaDeviceListError(ValueError):
    """Raised when device discovery cannot be invoked or normalized."""


@dataclass(frozen=True, slots=True)
class LlamaDeviceInventory:
    """Normalized logical devices plus the exact raw listing evidence."""

    devices: tuple[AcceleratorDevice, ...]
    stdout: str
    stderr: str


class LlamaDeviceListAdapter:
    """Build and parse llama.cpp --list-devices invocations."""

    def build_argv(
        self,
        *,
        binary_path: Path,
        capabilities: CapabilitySet,
    ) -> tuple[str, ...]:
        if not capabilities.supports("--list-devices"):
            raise LlamaDeviceListError(
                "selected binary does not advertise --list-devices"
            )
        return (str(binary_path), "--list-devices")

    def parse_output(
        self,
        stdout: str,
        stderr: str = "",
    ) -> LlamaDeviceInventory:
        combined = "\n".join(part for part in (stdout, stderr) if part)
        devices = parse_device_list(combined)
        return LlamaDeviceInventory(
            devices=devices,
            stdout=stdout,
            stderr=stderr,
        )


def parse_device_list(text: str) -> tuple[AcceleratorDevice, ...]:
    """Parse the stable 'Available devices' rows emitted by llama.cpp."""
    devices: list[AcceleratorDevice] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = _DEVICE_RE.match(line)
        if match is None:
            continue
        logical_name = match.group("name")
        backend_match = _BACKEND_RE.match(logical_name)
        if backend_match is None:
            continue
        if logical_name in seen:
            raise LlamaDeviceListError(
                f"duplicate logical device in --list-devices output: {logical_name}"
            )
        seen.add(logical_name)
        total_mib = match.group("total")
        free_mib = match.group("free")
        devices.append(
            AcceleratorDevice(
                logical_device_name=logical_name,
                backend=backend_match.group("backend").upper(),
                product_name=match.group("product").strip(),
                total_memory_bytes=(
                    int(total_mib) * _MIB if total_mib is not None else None
                ),
                free_memory_bytes=(
                    int(free_mib) * _MIB if free_mib is not None else None
                ),
            )
        )

    if not devices:
        raise LlamaDeviceListError(
            "could not find any logical devices in --list-devices output"
        )
    return tuple(devices)


def correlate_physical_devices(
    logical_devices: Sequence[AcceleratorDevice],
    host_gpus: Sequence[Mapping[str, Any]],
) -> tuple[AcceleratorDevice, ...]:
    """Map devices only when UUID or PCI identity provides an unambiguous match."""
    by_uuid: dict[str, Mapping[str, Any]] = {}
    by_pci: dict[str, Mapping[str, Any]] = {}
    for gpu in host_gpus:
        raw_uuid = gpu.get("uuid")
        if isinstance(raw_uuid, str) and raw_uuid:
            by_uuid[raw_uuid.lower()] = gpu
        raw_pci = gpu.get("pci_bus_id", gpu.get("pci_address"))
        if isinstance(raw_pci, str) and raw_pci:
            by_pci[raw_pci.lower()] = gpu

    mapped: list[AcceleratorDevice] = []
    for device in logical_devices:
        match: Mapping[str, Any] | None = None
        if device.uuid is not None:
            match = by_uuid.get(device.uuid.lower())
        if match is None and device.pci_bus_id is not None:
            match = by_pci.get(device.pci_bus_id.lower())
        if match is None:
            mapped.append(device)
            continue

        pci = _string_value(match.get("pci_bus_id", match.get("pci_address")))
        uuid = _string_value(match.get("uuid"))
        key = physical_device_key(pci_bus_id=pci, uuid=uuid)
        if key is None:
            mapped.append(device)
            continue
        mapped.append(
            device.model_copy(
                update={
                    "mapping_status": "mapped",
                    "physical_device_key": key,
                    "pci_bus_id": pci,
                    "uuid": uuid,
                    "vendor": device.vendor or _string_value(match.get("vendor_name")),
                    "driver": device.driver or _string_value(match.get("driver")),
                }
            )
        )
    return tuple(mapped)


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
