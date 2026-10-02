# V1 release checklist

The automated suite validates synthetic/reference behavior. The final workstation pass validates the actual executable/model/hardware combination that CI cannot provide.

## Automated gates

Run from a clean checkout:

~~~bash
uv lock --check
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest

cd frontend
npm ci
npm run typecheck
npm run test
npm run build
~~~

The test suite covers all migration upgrade paths, interruption/resume and orphan recovery, persisted OOM failures, malformed llama-bench JSON, unsupported arguments, native/custom capability differences, server startup failure, noisy/incomplete telemetry, launcher-profile drift, archive creation/restore integrity, reference database growth, and representative indexed query plans.

## Primary workstation acceptance

1. Confirm `llprof --version` reports `1.0.0`.
2. Register the exact native/custom llama.cpp binaries and SPEED-Bench.
3. Select the production Flash Next 128K launcher profile.
4. Preview the reference grid and verify 12 raw / 11 valid / 1 rejected.
5. Verify PP2K, PP8K, TG256@4K, TG256@50% produce 44 cases.
6. Run with per-Candidate fit at 128K and at least three repetitions.
7. Confirm CPU process/system and available GPU telemetry are persisted.
8. Interrupt once and resume; verify completed work is retained.
9. Inspect a batch × ubatch matrix, exact workload filters, Candidate comparison, and caller-defined Pareto view.
10. Validate finalists through llama-server + SPEED-Bench using the frozen resolved placement.
11. Generate and review a launcher promotion patch.
12. Run `llprof database check`; require integrity OK, zero FK violations, and indexed representative plans.
13. Create an archive, restore it to a new database path, and rerun `llprof database check` on the restored copy.
14. Open historical Candidate/run/validation records and confirm model/build/host/config provenance is understandable without external notes.

## Tagging

Only after the workstation acceptance above passes, tag the exact accepted `main` commit as `v1.0.0`. Record the commit SHA, binary hashes, launcher profile identifier, target/draft model identities, workstation hardware fingerprint, and archive hash in the release notes.
