# V1 troubleshooting

## Binary reports an unsupported argument

Run `llprof binary inspect` again against the exact executable and compare its capability surface with the expected build. Parameter controls are enabled only when the selected tool kind and advertised option support the dimension. A required non-default setting that the binary cannot represent is an error, not a silent omission.

## Registered binary changed on disk

Executors recheck SHA-256 identity before use. Re-register the new executable with `llprof binary inspect`; do not reuse the old binary record.

## Experiment stopped or the process was killed

Run `llprof experiment resume` with the same experiment and required binary/model inputs. Stale `running` attempts are recovered as `interrupted`; completed cases are not discarded.

## OOM, timeout, parser failure, fit failure, or benchmark failure

Failures are persisted observations. Inspect the run and captured logs rather than deleting the case. Correct the environmental/configuration problem and resume; retryable cases remain incomplete until a successful observation exists.

For malformed llama-bench output, verify the selected executable and its `--output json` behavior. The parser intentionally rejects invalid JSON, multiple result objects, missing sample arrays, and repetition-count mismatches.

## Telemetry is noisy or incomplete

Run-quality labels do not convert successful benchmark execution into failure. `external_cpu_load`, `external_gpu_load`, `noisy`, `thermal_throttle`, and `telemetry_incomplete` identify measurements that may need rerunning or filtering. Reduce background work, allow thermals to settle, and verify OS/GPU telemetry permissions.

## llama-server fails before readiness

Inspect the persisted server-run stderr and argv. Common causes are unsupported flags, wrong model paths, bind-port conflicts, or model-load failures. A process that exits before `/health` returns HTTP 200 is persisted as `start_failed`.

## Promotion says the launcher profile changed

The experiment freezes the source performance Candidate. M13 promotion compares the current launcher profile with that frozen base and refuses to generate a patch after drift. Create a new experiment from the current launcher profile (or deliberately restore the original launcher source) instead of applying a stale optimization.

## Database health or unexpected slowness

Run:

~~~bash
uv run --frozen llprof database check --database data/benchmarks.db
~~~

The command reports SQLite integrity, foreign-key violations, database/WAL size, row counts, and representative query plans. Critical workflow queries should report `indexed`.

## Archive restore fails

Restore verifies safe member paths, manifest structure, file sizes, SHA-256 hashes, SQLite `integrity_check`, and schema version. A failure indicates a damaged/modified archive or an incompatible layout. Restore always targets a path that does not already exist.

## Browser is unavailable

Build the production frontend first:

~~~bash
cd frontend
npm ci
npm run build
~~~

Then run `llprof ui`. The UI server refuses a missing build directory.

## API/UI remote exposure

The default host is `127.0.0.1`. V1 does not add remote-user authentication. Do not bind the API/UI to untrusted networks without an external access-control layer.
