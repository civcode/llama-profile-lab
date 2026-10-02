# V1 benchmark workflow

This tutorial describes the reference V1 workflow. Substitute the launcher profile, model paths, and executable paths for the target workstation.

## 1. Prepare the environment

~~~bash
uv sync --frozen
cd frontend && npm ci && npm run build && cd ..
~~~

Inspect the local tools so the database records exact executable hashes and detected capabilities:

~~~bash
uv run --frozen llprof binary inspect \
  /path/to/llama-bench \
  /path/to/llama-fit-params \
  /path/to/llama-server \
  /path/to/speed_bench.py \
  --database data/benchmarks.db
~~~

Use `llprof binary list` and `llprof binary compare` to confirm native/custom builds expose the arguments required by the intended Candidate dimensions.

## 2. Start the browser workflow

~~~bash
uv run --frozen llprof ui \
  --launcher-config /path/to/llama-profile-launcher/config/hosts/workstation.json \
  --database data/benchmarks.db
~~~

Select the existing Flash Next 128K launcher profile. Configure:

- batch size: 2048, 4096, 8192
- ubatch size: 512, 1024, 2048, 4096
- constraint: `compute.ubatch_size <= compute.batch_size`
- PP 2048 at depth 0
- PP 8192 at depth 0
- TG 256 at depth 4096
- TG 256 at 50% context
- per-Candidate placement
- at least three repetitions

The preview must report 12 raw combinations, 1 rejection, 11 Candidates, and 44 llama-bench cases.

## 3. Execute and resume

Run from the UI or CLI. Per-Candidate mode requires the registered fit binary and fits at the Candidate's full production context.

~~~bash
uv run --frozen llprof experiment run EXPERIMENT_ID \
  --binary BENCH_BINARY_ID \
  --fit-binary FIT_BINARY_ID \
  --model-path /path/to/target.gguf \
  --database data/benchmarks.db
~~~

After an interruption, resume the same experiment:

~~~bash
uv run --frozen llprof experiment resume EXPERIMENT_ID \
  --binary BENCH_BINARY_ID \
  --fit-binary FIT_BINARY_ID \
  --model-path /path/to/target.gguf \
  --database data/benchmarks.db
~~~

Completed cases are preserved. Failed/OOM/interrupted attempts remain visible in run history.

## 4. Analyze

Use the browser matrix and Candidate views or the CLI:

~~~bash
uv run --frozen llprof results matrix EXPERIMENT_ID \
  --x compute.batch_size \
  --y compute.ubatch_size \
  --metric throughput.median \
  --filter workload.kind=microbench-prefill \
  --filter workload.prompt_tokens=8192 \
  --database data/benchmarks.db
~~~

Pareto objectives are explicit. For example, maximize PP8K and TG4K rather than collapsing them into an overall score.

## 5. Validate finalists

Register the exact llama-server and SPEED-Bench builds, then validate selected Candidates with their already-resolved placement:

~~~bash
uv run --frozen llprof server validate EXPERIMENT_ID CANDIDATE_ID \
  --server-binary SERVER_BINARY_ID \
  --speed-bench-binary SPEED_BINARY_ID \
  --model-path /path/to/target.gguf \
  --draft-model-path /path/to/draft.gguf \
  --placement PLACEMENT_ID \
  --model-name finalist \
  --database data/benchmarks.db
~~~

## 6. Propose launcher promotion

Only a Candidate with completed server validation can be promoted:

~~~bash
uv run --frozen llprof profile promote EXPERIMENT_ID CANDIDATE_ID \
  --launcher-config /path/to/launcher/config/hosts/workstation.json \
  --output candidate.patch \
  --database data/benchmarks.db
~~~

Review the patch. The command does not modify launcher configuration. If the source launcher performance profile changed after experiment creation, promotion is rejected so the experiment can be rebased deliberately.

## 7. Check and archive

~~~bash
uv run --frozen llprof database check --database data/benchmarks.db

uv run --frozen llprof experiment export EXPERIMENT_ID \
  --output experiment.json \
  --database data/benchmarks.db

uv run --frozen llprof archive \
  --output experiment-archive.tar.gz \
  --database data/benchmarks.db
~~~

Test restoration into a new path before long-term storage:

~~~bash
uv run --frozen llprof archive \
  --restore experiment-archive.tar.gz \
  --database data/restored.db
~~~
