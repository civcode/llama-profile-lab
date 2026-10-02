# Frontend

The M11 browser UI is a React + TypeScript + Vite client over the local FastAPI service. It does not read SQLite directly and does not reimplement planning, execution, placement, analysis, or server-validation behavior.

## Requirements

- Node.js 22+
- npm
- a synchronized Python environment for the API (uv sync --frozen)

Frontend dependencies are pinned by package-lock.json.

## Development

Start the API from the repository root:

~~~bash
uv run --frozen llprof api \
  --launcher-config /path/to/llama-profile-launcher/config/hosts/workstation.json \
  --database data/benchmarks.db
~~~

Then, in another shell:

~~~bash
cd frontend
npm ci
npm run dev
~~~

Vite binds to 127.0.0.1:5173 and proxies /api to 127.0.0.1:8000.

## Production-style local serving

Build the static bundle:

~~~bash
cd frontend
npm ci
npm run build
cd ..
~~~

Then serve the bundle and API on one local origin:

~~~bash
uv run --frozen llprof ui \
  --host 127.0.0.1 \
  --port 8000 \
  --launcher-config /path/to/llama-profile-launcher/config/hosts/workstation.json \
  --database data/benchmarks.db
~~~

llprof ui serves frontend/dist by default. The generated bundle is not committed.

## Quality gates

~~~bash
cd frontend
npm ci
npm run typecheck
npm run test
npm run build
~~~

CI runs these gates together with the Python lock check, Ruff, strict mypy, and pytest.

## Browser workflow

The V1 UI provides:

- an experiment list with profile, status, progress, baseline, and validation state;
- form-driven experiment creation from a launcher profile whose typed Candidate mapping is supplied by the Python backend;
- generic N-dimensional parameter sweeps driven by backend parameter metadata;
- authoritative debounced plan previews from the same Python planning engine used for persisted experiments;
- binary-capability-aware disabling of unsupported parameters;
- workload editing for prefill, decode, combined, and optional SPEED-Bench cases;
- pre-execution raw/valid Candidate, benchmark-case, and repetition counts;
- per-Candidate or fixed-placement selection;
- start, pause, resume, and cancel controls;
- Server-Sent Events plus polling fallback for live progress;
- current Candidate/WorkloadCase, latest throughput, CPU, GPU, RAM/VRAM, thermal, power, recent-run, and failure visibility;
- sparse matrix/heatmap exploration with X/Y selection, exact higher-dimensional slices, and facets;
- workload-specific Pareto frontier exploration without a synthetic overall score;
- Candidate detail with configuration, resolved placement, workload results, measured decode-depth curve, resource/stability metrics, baseline deltas, and compute-only request latency;
- comparison of 2–5 Candidates against the configured baseline;
- M9 llama-server/SPEED-Bench finalist validation and validation history.

Content-address IDs, hashes, raw Candidate JSON, and detailed placement/configuration are collapsed under Advanced details by default; primary live views use Candidate ordinals and workload labels. Failures stay visible because they are persisted experiment data.
