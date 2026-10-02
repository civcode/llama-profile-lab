# V1 known limitations

- V1 is a local single-host benchmark system. There is no distributed scheduler.
- The API/UI defaults to loopback and does not provide remote-user authentication.
- Model registry path resolution is not a complete deployment artifact manager; benchmark/server commands still accept explicit model paths.
- Automatic placement depends on the selected llama-fit-params build and its detected capability surface.
- GPU telemetry is best on NVIDIA systems with `nvidia-smi`; generic DRM/sysfs fallback exposes fewer metrics.
- Request-latency estimates derived from PP/TG curves are compute-only estimates, not network/service latency measurements.
- SPEED-Bench/server validation is separate from llama-bench microbenchmark screening and requires explicit finalist selection.
- Promotion emits a patch/proposed snapshot and never applies production launcher configuration automatically.
- Promotion rejects performance-profile drift after experiment creation; V1 has no automatic rebase of a historical experiment onto a changed launcher profile.
- Promotion intentionally rejects launcher changes it cannot encode safely, including some placement/device/tensor override changes and tuple-valued extra arguments.
- Archive artifacts are caller-selected files; V1 does not automatically discover every external trace/profile file on disk.
- SQLite archives are point-in-time snapshots. Ongoing work after the archive was created is not included.
- The browser uses SSE with polling fallback but does not implement a separate durable web job queue.
- V1 analysis deliberately avoids a universal optimization score. Users choose explicit filters, comparisons, and Pareto objectives.
- Final release acceptance on a target workstation still depends on the availability of the intended llama.cpp builds, target/draft Qwen models, and representative hardware.
