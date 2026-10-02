"""Linux /proc and /sys telemetry provider tests."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.execution import telemetry as telemetry_module
from llama_profile_lab.execution.telemetry import (
    LinuxTelemetryProvider,
    SysfsGpuTelemetryProvider,
)


def _write_proc_stat(path: Path, *, user: int, system: int, idle: int) -> None:
    path.write_text(
        "\n".join(
            (
                f"cpu  {user} 0 {system} {idle} 0 0 0 0",
                f"cpu0 {user} 0 {system} {idle} 0 0 0 0",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_pid_stat(path: Path, *, user: int, system: int, rss_pages: int) -> None:
    fields = ["S"] + ["0"] * 21
    fields[11] = str(user)
    fields[12] = str(system)
    fields[17] = "8"
    fields[21] = str(rss_pages)
    path.write_text(f"123 (fake bench) {' '.join(fields)}\n", encoding="utf-8")


def test_linux_provider_reads_process_cpu_memory_frequency_and_temperature(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proc = tmp_path / "proc"
    sysroot = tmp_path / "sys"
    (proc / "123").mkdir(parents=True)
    cpu_freq = sysroot / "devices/system/cpu/cpu0/cpufreq"
    cpu_freq.mkdir(parents=True)
    hwmon = sysroot / "class/hwmon/hwmon0"
    hwmon.mkdir(parents=True)

    _write_proc_stat(proc / "stat", user=100, system=50, idle=850)
    _write_pid_stat(proc / "123/stat", user=10, system=5, rss_pages=100)
    (proc / "meminfo").write_text(
        "MemTotal: 1000 kB\n"
        "MemAvailable: 600 kB\n"
        "SwapTotal: 200 kB\n"
        "SwapFree: 150 kB\n",
        encoding="utf-8",
    )
    (proc / "cpuinfo").write_text("cpu MHz : 4000.000\n", encoding="utf-8")
    (cpu_freq / "scaling_cur_freq").write_text("4200000\n", encoding="utf-8")
    (hwmon / "name").write_text("k10temp\n", encoding="utf-8")
    (hwmon / "temp1_input").write_text("65000\n", encoding="utf-8")

    ticks = iter((1_000_000_000, 2_000_000_000))
    monkeypatch.setattr(telemetry_module.time, "monotonic_ns", lambda: next(ticks))

    provider = LinuxTelemetryProvider(
        proc_root=proc,
        sys_root=sysroot,
        gpu_provider=SysfsGpuTelemetryProvider(sys_root=sysroot),
    )

    _write_proc_stat(proc / "stat", user=130, system=60, idle=910)
    first = provider.sample(pid=123, phase="during")

    _write_proc_stat(proc / "stat", user=160, system=70, idle=970)
    _write_pid_stat(proc / "123/stat", user=20, system=10, rss_pages=120)
    second = provider.sample(pid=123, phase="during")

    assert first.cpu_system_pct is not None
    assert second.process_cpu_pct_raw is not None
    assert second.process_cpu_pct_normalized is not None
    assert second.process_threads == 8
    assert second.process_rss_bytes == 120 * provider._page_size
    assert second.ram_used_bytes == 400 * 1024
    assert second.ram_available_bytes == 600 * 1024
    assert second.swap_used_bytes == 50 * 1024
    assert second.cpu_freq_avg_hz == 4_200_000_000
    assert second.cpu_temperature_c == 65.0


def test_sysfs_gpu_provider_reads_generic_drm_metrics(tmp_path: Path) -> None:
    sysroot = tmp_path / "sys"
    device = sysroot / "class/drm/card0/device"
    hwmon = device / "hwmon/hwmon0"
    hwmon.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text("73\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("1024\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("4096\n", encoding="utf-8")
    (hwmon / "temp1_input").write_text("71000\n", encoding="utf-8")
    (hwmon / "power1_average").write_text("125000000\n", encoding="utf-8")

    samples = SysfsGpuTelemetryProvider(sys_root=sysroot).sample()

    assert len(samples) == 1
    assert samples[0].utilization_pct == 73
    assert samples[0].vram_used_bytes == 1024
    assert samples[0].vram_total_bytes == 4096
    assert samples[0].temperature_c == 71
    assert samples[0].power_w == 125
