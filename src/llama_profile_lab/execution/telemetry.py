"""Linux CPU/GPU telemetry sampling and run-quality classification."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from threading import Event, Lock, Thread
from typing import Protocol

from llama_profile_lab.domain.telemetry import (
    GpuTelemetrySample,
    GpuTelemetrySummary,
    RunQuality,
    RunQualityAssessment,
    TelemetryPhase,
    TelemetrySample,
    TelemetrySummary,
)


class TelemetryProvider(Protocol):
    """Source of host/process telemetry observations."""

    logical_cpu_count: int

    def sample(self, *, pid: int | None, phase: TelemetryPhase) -> TelemetrySample:
        """Capture one best-effort observation."""


class GpuTelemetryProvider(Protocol):
    """Source of per-GPU observations."""

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        """Capture current GPU state."""


@dataclass(frozen=True, slots=True)
class RunQualityPolicy:
    """Initial conservative thresholds for obvious benchmark contamination."""

    external_cpu_pct: float = 20.0
    noisy_cpu_pct: float = 10.0
    external_gpu_pct: float = 15.0
    noisy_gpu_pct: float = 5.0
    cpu_temperature_c: float = 95.0
    gpu_temperature_c: float = 90.0


class TelemetrySampler:
    """Collect telemetry asynchronously while the benchmark subprocess runs."""

    def __init__(
        self,
        provider: TelemetryProvider,
        *,
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds < 0.5:
            raise ValueError("telemetry interval must be at least 0.5 seconds")
        self.provider = provider
        self.interval_seconds = interval_seconds
        self._samples: list[TelemetrySample] = []
        self._errors: list[str] = []
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._pid: int | None = None

    def capture_before(self) -> None:
        self._capture(pid=None, phase="before")

    def start(self, pid: int) -> None:
        """Associate the benchmark PID and begin periodic sampling."""
        if self._thread is not None:
            raise RuntimeError("telemetry sampler already started")
        self._pid = pid
        self._capture(pid=pid, phase="during")
        self._thread = Thread(
            target=self._run,
            name=f"llprof-telemetry-{pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> tuple[TelemetrySample, ...]:
        """Stop periodic sampling and add a final host-only snapshot."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._capture(pid=None, phase="after")
        with self._lock:
            return tuple(self._samples)

    @property
    def errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            pid = self._pid
            if pid is None:
                return
            self._capture(pid=pid, phase="during")

    def _capture(self, *, pid: int | None, phase: TelemetryPhase) -> None:
        try:
            sample = self.provider.sample(pid=pid, phase=phase)
        except Exception as exc:  # telemetry must never terminate a benchmark
            with self._lock:
                self._errors.append(f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            self._samples.append(sample)


class LinuxTelemetryProvider:
    """Best-effort Linux telemetry using /proc, /sys, and optional GPU tools."""

    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
        gpu_provider: GpuTelemetryProvider | None = None,
    ) -> None:
        self.proc_root = proc_root
        self.sys_root = sys_root
        self.logical_cpu_count = os.cpu_count() or 1
        self._clock_ticks = os.sysconf("SC_CLK_TCK")
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._previous_cpu = self._read_cpu_ticks()
        self._previous_process: dict[int, tuple[int, int]] = {}
        self._previous_rapl: dict[Path, tuple[int, int]] = {}
        self.gpu_provider = gpu_provider or AutoGpuTelemetryProvider(sys_root=sys_root)

    def sample(self, *, pid: int | None, phase: TelemetryPhase) -> TelemetrySample:
        now_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        current_cpu = self._read_cpu_ticks()
        cpu_values, per_core = _cpu_percentages(self._previous_cpu, current_cpu)
        self._previous_cpu = current_cpu

        process = self._read_process(pid, monotonic_ns) if pid is not None else {}
        memory = self._read_memory()
        frequencies = self._read_cpu_frequencies()
        cpu_temperature = self._read_cpu_temperature()
        package_power = self._read_package_power(monotonic_ns)
        load_1m: float | None
        load_5m: float | None
        try:
            load_1m, load_5m, _ = os.getloadavg()
        except OSError:
            load_1m = load_5m = None

        try:
            gpus = self.gpu_provider.sample()
        except Exception:
            gpus = ()

        extra: dict[str, object] = {}
        if package_power is not None:
            extra["cpu_package_power_w"] = package_power

        return TelemetrySample(
            timestamp_ns=now_ns,
            phase=phase,
            cpu_system_pct=cpu_values.get("system_total"),
            cpu_user_pct=cpu_values.get("user"),
            cpu_system_mode_pct=cpu_values.get("system"),
            cpu_iowait_pct=cpu_values.get("iowait"),
            process_cpu_pct_normalized=process.get("cpu_normalized"),
            process_cpu_pct_raw=process.get("cpu_raw"),
            process_user_time_ns=_as_int(process.get("user_time_ns")),
            process_system_time_ns=_as_int(process.get("system_time_ns")),
            process_threads=_as_int(process.get("threads")),
            cpu_freq_avg_hz=frequencies[0],
            cpu_freq_min_hz=frequencies[1],
            cpu_freq_max_hz=frequencies[2],
            cpu_temperature_c=cpu_temperature,
            load_avg_1m=load_1m,
            load_avg_5m=load_5m,
            ram_used_bytes=memory.get("ram_used"),
            ram_available_bytes=memory.get("ram_available"),
            swap_used_bytes=memory.get("swap_used"),
            process_rss_bytes=_as_int(process.get("rss_bytes")),
            gpus=gpus,
            cpu_per_core_pct=per_core,
            extra=extra,
        )

    def _read_cpu_ticks(self) -> dict[str, tuple[int, ...]]:
        path = self.proc_root / "stat"
        result: dict[str, tuple[int, ...]] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return result
        for line in lines:
            parts = line.split()
            if not parts or not parts[0].startswith("cpu"):
                continue
            if parts[0] != "cpu" and not parts[0][3:].isdigit():
                continue
            try:
                result[parts[0]] = tuple(int(value) for value in parts[1:9])
            except ValueError:
                continue
        return result

    def _read_process(self, pid: int, monotonic_ns: int) -> dict[str, float | int]:
        stat_path = self.proc_root / str(pid) / "stat"
        try:
            raw = stat_path.read_text(encoding="utf-8").strip()
        except OSError:
            return {}
        close = raw.rfind(")")
        if close < 0:
            return {}
        fields = raw[close + 2 :].split()
        if len(fields) < 22:
            return {}
        try:
            user_ticks = int(fields[11])
            system_ticks = int(fields[12])
            threads = int(fields[17])
            rss_pages = int(fields[21])
        except (ValueError, IndexError):
            return {}

        total_ticks = user_ticks + system_ticks
        cpu_raw: float | None = None
        previous = self._previous_process.get(pid)
        if previous is not None:
            previous_ns, previous_ticks = previous
            elapsed = (monotonic_ns - previous_ns) / 1_000_000_000
            if elapsed > 0:
                cpu_raw = (
                    (total_ticks - previous_ticks)
                    / self._clock_ticks
                    / elapsed
                    * 100.0
                )
        self._previous_process[pid] = (monotonic_ns, total_ticks)

        result: dict[str, float | int] = {
            "user_time_ns": int(user_ticks / self._clock_ticks * 1_000_000_000),
            "system_time_ns": int(system_ticks / self._clock_ticks * 1_000_000_000),
            "threads": max(0, threads),
            "rss_bytes": max(0, rss_pages * self._page_size),
        }
        if cpu_raw is not None:
            result["cpu_raw"] = max(0.0, cpu_raw)
            result["cpu_normalized"] = max(
                0.0,
                cpu_raw / max(1, self.logical_cpu_count),
            )
        return result

    def _read_memory(self) -> dict[str, int]:
        try:
            lines = (self.proc_root / "meminfo").read_text(encoding="utf-8").splitlines()
        except OSError:
            return {}
        values: dict[str, int] = {}
        for line in lines:
            name, sep, rest = line.partition(":")
            if not sep:
                continue
            parts = rest.split()
            if not parts:
                continue
            try:
                values[name] = int(parts[0]) * 1024
            except ValueError:
                continue
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        swap_total = values.get("SwapTotal")
        swap_free = values.get("SwapFree")
        result: dict[str, int] = {}
        if total is not None and available is not None:
            result["ram_used"] = max(0, total - available)
            result["ram_available"] = max(0, available)
        if swap_total is not None and swap_free is not None:
            result["swap_used"] = max(0, swap_total - swap_free)
        return result

    def _read_cpu_frequencies(self) -> tuple[int | None, int | None, int | None]:
        values: list[int] = []
        for path in sorted(
            (self.sys_root / "devices/system/cpu").glob(
                "cpu[0-9]*/cpufreq/scaling_cur_freq"
            )
        ):
            raw = _read_text(path)
            if raw is None:
                continue
            try:
                values.append(int(raw) * 1000)
            except ValueError:
                continue
        if not values:
            try:
                lines = (self.proc_root / "cpuinfo").read_text(
                    encoding="utf-8"
                ).splitlines()
            except OSError:
                lines = []
            for line in lines:
                if not line.lower().startswith("cpu mhz"):
                    continue
                _, _, value = line.partition(":")
                try:
                    values.append(int(float(value.strip()) * 1_000_000))
                except ValueError:
                    continue
        if not values:
            return None, None, None
        return int(fmean(values)), min(values), max(values)

    def _read_cpu_temperature(self) -> float | None:
        values: list[float] = []
        hwmon_root = self.sys_root / "class/hwmon"
        for hwmon in hwmon_root.glob("hwmon*"):
            name = (_read_text(hwmon / "name") or "").lower()
            if name not in {"coretemp", "k10temp", "zenpower", "cpu_thermal"}:
                continue
            for path in hwmon.glob("temp*_input"):
                value = _millidegree_c(path)
                if value is not None:
                    values.append(value)
        for zone in (self.sys_root / "class/thermal").glob("thermal_zone*"):
            zone_type = (_read_text(zone / "type") or "").lower()
            if not any(token in zone_type for token in ("cpu", "pkg", "x86_pkg")):
                continue
            value = _millidegree_c(zone / "temp")
            if value is not None:
                values.append(value)
        return max(values) if values else None

    def _read_package_power(self, monotonic_ns: int) -> float | None:
        watts: list[float] = []
        for energy_path in sorted(
            (self.sys_root / "class/powercap").glob("intel-rapl:*/energy_uj")
        ):
            raw = _read_text(energy_path)
            if raw is None:
                continue
            try:
                energy_uj = int(raw)
            except ValueError:
                continue
            previous = self._previous_rapl.get(energy_path)
            self._previous_rapl[energy_path] = (monotonic_ns, energy_uj)
            if previous is None:
                continue
            previous_ns, previous_energy = previous
            elapsed = (monotonic_ns - previous_ns) / 1_000_000_000
            delta = energy_uj - previous_energy
            if elapsed > 0 and delta >= 0:
                watts.append((delta / 1_000_000) / elapsed)
        return sum(watts) if watts else None


class CompositeGpuTelemetryProvider:
    """Sample multiple GPU providers and merge only strongly correlated devices."""

    def __init__(
        self,
        providers: tuple[tuple[str, GpuTelemetryProvider], ...],
    ) -> None:
        self.providers = providers

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        merged: list[GpuTelemetrySample] = []
        for source, provider in self.providers:
            try:
                samples = provider.sample()
            except Exception:
                continue
            for sample in samples:
                observed = _with_source(sample, source)
                match_index = next(
                    (
                        index
                        for index, existing in enumerate(merged)
                        if _same_physical_gpu(existing, observed)
                    ),
                    None,
                )
                if match_index is None:
                    merged.append(observed)
                else:
                    merged[match_index] = _merge_gpu_samples(
                        merged[match_index],
                        observed,
                    )
        return tuple(merged)


class AutoGpuTelemetryProvider:
    """Compose NVIDIA management telemetry with generic DRM/sysfs telemetry."""

    def __init__(self, *, sys_root: Path = Path("/sys")) -> None:
        executable = shutil.which("nvidia-smi")
        self.nvidia = (
            NvidiaSmiGpuTelemetryProvider(Path(executable))
            if executable is not None
            else None
        )
        self.sysfs = SysfsGpuTelemetryProvider(sys_root=sys_root)
        providers: list[tuple[str, GpuTelemetryProvider]] = []
        if self.nvidia is not None:
            providers.append(("nvidia-smi", self.nvidia))
        providers.append(("sysfs", self.sysfs))
        self.composite = CompositeGpuTelemetryProvider(tuple(providers))

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        return self.composite.sample()


class NvidiaSmiGpuTelemetryProvider:
    """Read NVIDIA telemetry through one XML nvidia-smi query per sample."""

    def __init__(self, executable: Path = Path("nvidia-smi")) -> None:
        self.executable = executable

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        try:
            completed = subprocess.run(
                (str(self.executable), "-q", "-x"),
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ()
        if completed.returncode != 0 or not completed.stdout.strip():
            return ()
        try:
            root = ET.fromstring(completed.stdout)
        except ET.ParseError:
            return ()

        result: list[GpuTelemetrySample] = []
        for index, gpu in enumerate(root.findall("gpu")):
            active_reasons = tuple(
                child.tag
                for container_name in ("clocks_event_reasons", "clocks_throttle_reasons")
                for container in gpu.findall(container_name)
                for child in container
                if (child.text or "").strip().lower() == "active"
            )
            raw_pci = _xml_text(gpu, "pci/pci_bus_id")
            pci_bus_id = normalize_pci_bus_id(raw_pci)
            uuid = _xml_text(gpu, "uuid")
            result.append(
                GpuTelemetrySample(
                    device=pci_bus_id or raw_pci or f"gpu{index}",
                    name=_xml_text(gpu, "product_name"),
                    uuid=uuid,
                    pci_bus_id=pci_bus_id,
                    stable_device_key=_stable_gpu_key(
                        pci_bus_id=pci_bus_id,
                        uuid=uuid,
                    ),
                    sources=("nvidia-smi",),
                    utilization_pct=_parse_number(_xml_text(gpu, "utilization/gpu_util")),
                    vram_used_bytes=_parse_mib(_xml_text(gpu, "fb_memory_usage/used")),
                    vram_total_bytes=_parse_mib(_xml_text(gpu, "fb_memory_usage/total")),
                    temperature_c=_parse_number(_xml_text(gpu, "temperature/gpu_temp")),
                    power_w=_parse_number(
                        _xml_text(gpu, "gpu_power_readings/power_draw")
                        or _xml_text(gpu, "power_readings/power_draw")
                    ),
                    graphics_clock_hz=_parse_mhz(
                        _xml_text(gpu, "clocks/graphics_clock")
                    ),
                    memory_clock_hz=_parse_mhz(
                        _xml_text(gpu, "clocks/mem_clock")
                    ),
                    thermal_throttled=any(
                        "thermal" in reason.lower() for reason in active_reasons
                    ),
                    extra={"active_clock_reasons": list(active_reasons)},
                )
            )
        return tuple(result)


class SysfsGpuTelemetryProvider:
    """Generic DRM/sysfs telemetry, useful for AMD and other Linux GPUs."""

    def __init__(self, *, sys_root: Path = Path("/sys")) -> None:
        self.sys_root = sys_root

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        result: list[GpuTelemetrySample] = []
        for card in sorted((self.sys_root / "class/drm").glob("card[0-9]*")):
            device = card / "device"
            if not device.exists():
                continue
            busy = _read_float(device / "gpu_busy_percent")
            used = _read_int(device / "mem_info_vram_used")
            total = _read_int(device / "mem_info_vram_total")
            temp = _first_hwmon_temperature(device)
            power_w = _first_hwmon_power(device)
            pci_bus_id = normalize_pci_bus_id(device.resolve().name)
            result.append(
                GpuTelemetrySample(
                    device=pci_bus_id or device.resolve().name,
                    pci_bus_id=pci_bus_id,
                    stable_device_key=_stable_gpu_key(
                        pci_bus_id=pci_bus_id,
                        uuid=None,
                    ),
                    sources=("sysfs",),
                    utilization_pct=busy,
                    vram_used_bytes=used,
                    vram_total_bytes=total,
                    temperature_c=temp,
                    power_w=power_w,
                )
            )
        return tuple(result)


def summarize_telemetry(samples: Iterable[TelemetrySample]) -> TelemetrySummary:
    """Reduce raw telemetry into stable scalar resource metrics."""
    all_samples = tuple(samples)
    during = tuple(sample for sample in all_samples if sample.phase == "during")
    gpu_samples = tuple(gpu for sample in all_samples for gpu in sample.gpus)
    gpu_devices = _summarize_gpu_devices(gpu_samples)

    process_user = [s.process_user_time_ns for s in during if s.process_user_time_ns is not None]
    process_system = [
        s.process_system_time_ns for s in during if s.process_system_time_ns is not None
    ]
    freq_avg_values = [s.cpu_freq_avg_hz for s in all_samples if s.cpu_freq_avg_hz is not None]
    freq_min_values = [s.cpu_freq_min_hz for s in all_samples if s.cpu_freq_min_hz is not None]
    freq_max_values = [s.cpu_freq_max_hz for s in all_samples if s.cpu_freq_max_hz is not None]

    external_cpu = [
        max(0.0, sample.cpu_system_pct - sample.process_cpu_pct_normalized)
        for sample in during
        if sample.cpu_system_pct is not None
        and sample.process_cpu_pct_normalized is not None
    ]
    baseline_gpu = [
        gpu.utilization_pct
        for sample in all_samples
        if sample.phase in {"before", "after"}
        for gpu in sample.gpus
        if gpu.utilization_pct is not None
    ]

    return TelemetrySummary(
        sample_count=len(all_samples),
        during_sample_count=len(during),
        cpu_system_avg_pct=_avg(s.cpu_system_pct for s in during),
        cpu_system_peak_pct=_max(s.cpu_system_pct for s in during),
        process_cpu_avg_pct_normalized=_avg(
            s.process_cpu_pct_normalized for s in during
        ),
        process_cpu_peak_pct_normalized=_max(
            s.process_cpu_pct_normalized for s in during
        ),
        process_cpu_avg_pct_raw=_avg(s.process_cpu_pct_raw for s in during),
        process_cpu_peak_pct_raw=_max(s.process_cpu_pct_raw for s in during),
        cpu_freq_avg_hz=int(fmean(freq_avg_values)) if freq_avg_values else None,
        cpu_freq_min_hz=min(freq_min_values) if freq_min_values else None,
        cpu_freq_max_hz=max(freq_max_values) if freq_max_values else None,
        cpu_temperature_peak_c=_max(s.cpu_temperature_c for s in all_samples),
        process_user_time_ns=(
            max(process_user) - min(process_user) if len(process_user) >= 2 else None
        ),
        process_system_time_ns=(
            max(process_system) - min(process_system)
            if len(process_system) >= 2
            else None
        ),
        ram_used_peak_bytes=_max_int(s.ram_used_bytes for s in all_samples),
        ram_available_min_bytes=_min_int(
            s.ram_available_bytes for s in all_samples
        ),
        swap_used_peak_bytes=_max_int(s.swap_used_bytes for s in all_samples),
        process_rss_peak_bytes=_max_int(s.process_rss_bytes for s in during),
        gpu_utilization_avg_pct=_avg(gpu.utilization_pct for gpu in gpu_samples),
        gpu_utilization_peak_pct=_max(
            gpu.utilization_pct for gpu in gpu_samples
        ),
        gpu_temperature_peak_c=_max(gpu.temperature_c for gpu in gpu_samples),
        gpu_vram_used_peak_bytes=_max_int(
            gpu.vram_used_bytes for gpu in gpu_samples
        ),
        gpu_power_avg_w=_avg(gpu.power_w for gpu in gpu_samples),
        gpu_power_peak_w=_max(gpu.power_w for gpu in gpu_samples),
        gpu_devices=gpu_devices,
        external_cpu_peak_pct=max(external_cpu) if external_cpu else None,
        baseline_gpu_utilization_peak_pct=(
            max(baseline_gpu) if baseline_gpu else None
        ),
    )


def classify_run_quality(
    samples: Iterable[TelemetrySample],
    *,
    expect_gpu: bool,
    sampler_errors: Iterable[str] = (),
    policy: RunQualityPolicy | None = None,
) -> RunQualityAssessment:
    """Assign one primary quality label while retaining every observed reason."""
    effective = policy or RunQualityPolicy()
    sample_tuple = tuple(samples)
    summary = summarize_telemetry(sample_tuple)
    errors = tuple(sampler_errors)
    reasons: list[str] = []

    has_process_cpu = any(
        sample.phase == "during" and sample.process_cpu_pct_normalized is not None
        for sample in sample_tuple
    )
    has_system_cpu = any(sample.cpu_system_pct is not None for sample in sample_tuple)
    has_gpu = any(sample.gpus for sample in sample_tuple)
    telemetry_complete = (
        summary.during_sample_count > 0
        and has_process_cpu
        and has_system_cpu
        and (has_gpu or not expect_gpu)
        and not errors
    )
    if not telemetry_complete:
        reasons.append("required telemetry was unavailable or sampling reported errors")

    thermal = any(
        (sample.cpu_temperature_c is not None
         and sample.cpu_temperature_c >= effective.cpu_temperature_c)
        or any(
            gpu.thermal_throttled is True
            or (
                gpu.temperature_c is not None
                and gpu.temperature_c >= effective.gpu_temperature_c
            )
            for gpu in sample.gpus
        )
        for sample in sample_tuple
    )
    if thermal:
        reasons.append("thermal threshold or explicit GPU thermal-throttle signal observed")

    external_cpu = (
        summary.external_cpu_peak_pct is not None
        and summary.external_cpu_peak_pct >= effective.external_cpu_pct
    )
    if external_cpu:
        reasons.append(
            f"estimated external CPU load reached {summary.external_cpu_peak_pct:.1f}%"
        )

    external_gpu = (
        summary.baseline_gpu_utilization_peak_pct is not None
        and summary.baseline_gpu_utilization_peak_pct >= effective.external_gpu_pct
    )
    if external_gpu:
        reasons.append(
            "GPU utilization was already elevated outside the benchmark interval"
        )

    noisy = (
        (
            summary.external_cpu_peak_pct is not None
            and summary.external_cpu_peak_pct >= effective.noisy_cpu_pct
        )
        or (
            summary.baseline_gpu_utilization_peak_pct is not None
            and summary.baseline_gpu_utilization_peak_pct >= effective.noisy_gpu_pct
        )
    )
    if noisy and not (external_cpu or external_gpu):
        reasons.append("background resource activity exceeded the noise threshold")

    quality: RunQuality
    if thermal:
        quality = "thermal_throttle"
    elif external_cpu:
        quality = "external_cpu_load"
    elif external_gpu:
        quality = "external_gpu_load"
    elif not telemetry_complete:
        quality = "telemetry_incomplete"
    elif noisy:
        quality = "noisy"
    else:
        quality = "clean"

    return RunQualityAssessment(
        quality=quality,
        reasons=tuple(reasons),
        telemetry_complete=telemetry_complete,
        summary=summary,
    )


def summary_metrics(summary: TelemetrySummary) -> dict[str, int | float]:
    """Render aggregate and per-device telemetry into generic metric keys."""
    raw = summary.model_dump(mode="python", exclude={"gpu_devices"})
    metrics: dict[str, int | float] = {}
    for name, value in raw.items():
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[f"telemetry.{name}"] = value

    for gpu in summary.gpu_devices:
        prefix = f"telemetry.gpu.{gpu.metric_device_id}"
        values = {
            "utilization_avg_pct": gpu.utilization_avg_pct,
            "utilization_peak_pct": gpu.utilization_peak_pct,
            "vram_used_peak_bytes": gpu.vram_used_peak_bytes,
            "temperature_peak_c": gpu.temperature_peak_c,
            "power_avg_w": gpu.power_avg_w,
            "power_peak_w": gpu.power_peak_w,
        }
        for name, value in values.items():
            if value is not None:
                metrics[f"{prefix}.{name}"] = value
    return metrics


_PCI_BUS_RE = re.compile(
    r"^(?P<domain>[0-9a-fA-F]{4,8}):"
    r"(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)


def normalize_pci_bus_id(value: str | None) -> str | None:
    """Normalize Linux/NVIDIA PCI bus IDs to domain:bus:device.function."""
    if value is None:
        return None
    raw = value.strip()
    if raw.lower().startswith("pci:"):
        raw = raw[4:]
    match = _PCI_BUS_RE.fullmatch(raw)
    if match is None:
        return None
    domain_value = int(match.group("domain"), 16)
    domain = (
        f"{domain_value:04x}"
        if domain_value <= 0xFFFF
        else f"{domain_value:08x}"
    )
    return (
        f"{domain}:{match.group('bus').lower()}:"
        f"{match.group('device').lower()}."
        f"{match.group('function')}"
    )


def _stable_gpu_key(
    *,
    pci_bus_id: str | None,
    uuid: str | None,
) -> str | None:
    if pci_bus_id is not None:
        return f"pci:{pci_bus_id}"
    if uuid:
        return f"uuid:{uuid.strip().lower()}"
    return None


def _with_source(
    sample: GpuTelemetrySample,
    source: str,
) -> GpuTelemetrySample:
    sources = tuple(dict.fromkeys((*sample.sources, source)))
    pci_bus_id = sample.pci_bus_id or normalize_pci_bus_id(sample.device)
    stable_device_key = _stable_gpu_key(
        pci_bus_id=pci_bus_id,
        uuid=sample.uuid,
    ) or sample.stable_device_key
    return sample.model_copy(
        update={
            "pci_bus_id": pci_bus_id,
            "stable_device_key": stable_device_key,
            "sources": sources,
        }
    )


def _same_physical_gpu(
    left: GpuTelemetrySample,
    right: GpuTelemetrySample,
) -> bool:
    left_pci = left.pci_bus_id or normalize_pci_bus_id(left.device)
    right_pci = right.pci_bus_id or normalize_pci_bus_id(right.device)
    if left_pci is not None and right_pci is not None:
        return left_pci == right_pci

    if left.uuid and right.uuid:
        return left.uuid.strip().lower() == right.uuid.strip().lower()

    left_key = left.stable_device_key
    right_key = right.stable_device_key
    if left_key is not None and right_key is not None:
        return left_key == right_key
    return False


def _merge_gpu_samples(
    preferred: GpuTelemetrySample,
    fallback: GpuTelemetrySample,
) -> GpuTelemetrySample:
    """Prefer the first provider and fill only missing values from later providers."""
    sources = tuple(dict.fromkeys((*preferred.sources, *fallback.sources)))
    pci_bus_id = preferred.pci_bus_id or fallback.pci_bus_id
    uuid = preferred.uuid or fallback.uuid
    stable_device_key = (
        _stable_gpu_key(pci_bus_id=pci_bus_id, uuid=uuid)
        or preferred.stable_device_key
        or fallback.stable_device_key
    )
    extra = dict(fallback.extra)
    extra.update(preferred.extra)
    return preferred.model_copy(
        update={
            "name": preferred.name or fallback.name,
            "uuid": uuid,
            "pci_bus_id": pci_bus_id,
            "stable_device_key": stable_device_key,
            "sources": sources,
            "utilization_pct": (
                preferred.utilization_pct
                if preferred.utilization_pct is not None
                else fallback.utilization_pct
            ),
            "vram_used_bytes": (
                preferred.vram_used_bytes
                if preferred.vram_used_bytes is not None
                else fallback.vram_used_bytes
            ),
            "vram_total_bytes": (
                preferred.vram_total_bytes
                if preferred.vram_total_bytes is not None
                else fallback.vram_total_bytes
            ),
            "temperature_c": (
                preferred.temperature_c
                if preferred.temperature_c is not None
                else fallback.temperature_c
            ),
            "power_w": (
                preferred.power_w
                if preferred.power_w is not None
                else fallback.power_w
            ),
            "graphics_clock_hz": (
                preferred.graphics_clock_hz
                if preferred.graphics_clock_hz is not None
                else fallback.graphics_clock_hz
            ),
            "memory_clock_hz": (
                preferred.memory_clock_hz
                if preferred.memory_clock_hz is not None
                else fallback.memory_clock_hz
            ),
            "thermal_throttled": (
                preferred.thermal_throttled
                if preferred.thermal_throttled is not None
                else fallback.thermal_throttled
            ),
            "extra": extra,
        }
    )


def _summary_gpu_key(sample: GpuTelemetrySample) -> str:
    stable = sample.stable_device_key or _stable_gpu_key(
        pci_bus_id=sample.pci_bus_id or normalize_pci_bus_id(sample.device),
        uuid=sample.uuid,
    )
    if stable is not None:
        return stable
    source = sample.sources[0] if sample.sources else "unknown"
    return f"source:{source}:{sample.device}"


def _metric_device_id(stable_device_key: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", stable_device_key)
    return normalized.strip("_").lower() or "gpu"


def _summarize_gpu_devices(
    samples: Iterable[GpuTelemetrySample],
) -> tuple[GpuTelemetrySummary, ...]:
    grouped: dict[str, list[GpuTelemetrySample]] = {}
    for sample in samples:
        grouped.setdefault(_summary_gpu_key(sample), []).append(sample)

    summaries: list[GpuTelemetrySummary] = []
    for key in sorted(grouped):
        device_samples = grouped[key]
        first = device_samples[0]
        sources = tuple(
            dict.fromkeys(
                source
                for sample in device_samples
                for source in sample.sources
            )
        )
        name = next(
            (sample.name for sample in device_samples if sample.name),
            None,
        )
        summaries.append(
            GpuTelemetrySummary(
                stable_device_key=key,
                metric_device_id=_metric_device_id(key),
                device=first.device,
                name=name,
                sources=sources,
                sample_count=len(device_samples),
                utilization_avg_pct=_avg(
                    sample.utilization_pct for sample in device_samples
                ),
                utilization_peak_pct=_max(
                    sample.utilization_pct for sample in device_samples
                ),
                vram_used_peak_bytes=_max_int(
                    sample.vram_used_bytes for sample in device_samples
                ),
                temperature_peak_c=_max(
                    sample.temperature_c for sample in device_samples
                ),
                power_avg_w=_avg(
                    sample.power_w for sample in device_samples
                ),
                power_peak_w=_max(
                    sample.power_w for sample in device_samples
                ),
            )
        )
    return tuple(summaries)


def _cpu_percentages(
    previous: dict[str, tuple[int, ...]],
    current: dict[str, tuple[int, ...]],
) -> tuple[dict[str, float], tuple[float, ...]]:
    aggregate = _cpu_line_percent(previous.get("cpu"), current.get("cpu"))
    per_core: list[float] = []
    for name in sorted(
        (key for key in current if key != "cpu"),
        key=lambda value: int(value[3:]),
    ):
        values = _cpu_line_percent(previous.get(name), current.get(name))
        total = values.get("system_total")
        if total is not None:
            per_core.append(total)
    return aggregate, tuple(per_core)


def _cpu_line_percent(
    previous: tuple[int, ...] | None,
    current: tuple[int, ...] | None,
) -> dict[str, float]:
    if previous is None or current is None or len(previous) < 8 or len(current) < 8:
        return {}
    delta = tuple(max(0, right - left) for left, right in zip(previous, current, strict=True))
    total = sum(delta)
    if total <= 0:
        return {}
    user, nice, system, idle, iowait, irq, softirq, steal = delta
    busy = total - idle - iowait
    return {
        "system_total": max(0.0, min(100.0, busy / total * 100)),
        "user": max(0.0, min(100.0, (user + nice) / total * 100)),
        "system": max(
            0.0,
            min(100.0, (system + irq + softirq) / total * 100),
        ),
        "iowait": max(0.0, min(100.0, iowait / total * 100)),
        "steal": max(0.0, min(100.0, steal / total * 100)),
    }


def _avg(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _max(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _max_int(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _min_int(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _read_int(path: Path) -> int | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _read_float(path: Path) -> float | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _millidegree_c(path: Path) -> float | None:
    value = _read_float(path)
    return None if value is None else value / 1000.0


def _first_hwmon_temperature(device: Path) -> float | None:
    values = [
        value
        for hwmon in (device / "hwmon").glob("hwmon*")
        for path in hwmon.glob("temp*_input")
        if (value := _millidegree_c(path)) is not None
    ]
    return max(values) if values else None


def _first_hwmon_power(device: Path) -> float | None:
    values = [
        value / 1_000_000
        for hwmon in (device / "hwmon").glob("hwmon*")
        for path in hwmon.glob("power*_average")
        if (value := _read_float(path)) is not None
    ]
    return sum(values) if values else None


def _xml_text(element: ET.Element, path: str) -> str | None:
    child = element.find(path)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value if value and value.upper() != "N/A" else None


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    token = value.split()[0]
    try:
        return max(0.0, float(token))
    except ValueError:
        return None


def _parse_mib(value: str | None) -> int | None:
    number = _parse_number(value)
    return None if number is None else int(number * 1024 * 1024)


def _parse_mhz(value: str | None) -> int | None:
    number = _parse_number(value)
    return None if number is None else int(number * 1_000_000)


def _as_int(value: float | int | None) -> int | None:
    return None if value is None else int(value)
