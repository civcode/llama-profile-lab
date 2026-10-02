# Getting started

This guide is the shortest path from a fresh checkout to a useful llama.cpp tuning experiment.

## What llama-profile-lab does

`llama-profile-lab` helps you compare llama.cpp configurations in a reproducible way. Instead of manually changing flags and writing results down, you define an **experiment**:

- a starting **Candidate** (the model/profile configuration);
- the settings you want to vary, such as batch and ubatch size;
- one or more **workloads** to measure;
- a measurement policy, such as three repetitions.

The planner expands those choices into concrete Candidates and benchmark cases. Results, failures, telemetry, exact binary hashes, placements, and validation history are stored in SQLite so you can resume work and understand later exactly what was tested.

The normal workflow is:

~~~text
launcher profile
    ↓
experiment + parameter sweep
    ↓
plan
    ↓
llama-fit-params + llama-bench
    ↓
compare results
    ↓
llama-server / SPEED-Bench validation
    ↓
review launcher patch
~~~

For beginners, the browser UI is the easiest way to create and inspect experiments. The CLI uses the same backend services and is useful for automation and troubleshooting.

## 1. Install the project

Requirements:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+ and npm
- local llama.cpp binaries when you want to run real benchmarks

From the repository root:

~~~bash
uv sync --frozen

cd frontend
npm ci
npm run build
cd ..
~~~

Check that the CLI works:

~~~bash
uv run --frozen llprof --version
uv run --frozen llprof --help
~~~

## 2. Register your llama.cpp tools

Before benchmarking, register the exact executables you plan to use. This records their SHA-256 hashes and detects which command-line options each build supports.

~~~bash
uv run --frozen llprof binary inspect \
  /path/to/llama-bench \
  /path/to/llama-fit-params \
  /path/to/llama-server \
  /path/to/speed_bench.py \
  --database data/benchmarks.db
~~~

You do not need every tool immediately. `llama-bench` is used for microbenchmarks, `llama-fit-params` for placement, and `llama-server` plus SPEED-Bench for finalist validation.

List the registered binaries with:

~~~bash
uv run --frozen llprof binary list --database data/benchmarks.db
~~~

## 3. Start the browser UI

If you already use llama-profile-launcher, point the UI at its host configuration:

~~~bash
uv run --frozen llprof ui \
  --launcher-config /path/to/llama-profile-launcher/config/hosts/workstation.json \
  --database data/benchmarks.db
~~~

Open the local address printed by the command. The server binds to `127.0.0.1` by default.

## 4. Create your first experiment

A good first experiment varies only batch and ubatch size.

For example:

~~~text
batch_size:   2048, 4096, 8192
ubatch_size:  512, 1024, 2048, 4096
constraint:   ubatch_size <= batch_size
~~~

With that constraint, the planner produces **11 valid Candidates**.

Add a small set of workloads such as:

~~~text
PP2048 @ depth 0
PP8192 @ depth 0
TG256  @ depth 4096
TG256  @ 50% context
~~~

Here, **PP** means *prompt processing* (prefill): how quickly the model reads an input prompt. `PP2048` therefore measures a 2,048-token prompt, while `PP8192` measures a larger 8,192-token prompt. These use depth 0 because the benchmark starts without an existing conversation history and measures the prompt itself being loaded into context.

**TG** means *token generation* (decode): how quickly the model produces new tokens after context already exists. `TG256` means “generate 256 tokens.” The depth tells llama-bench how much context is already active before generation starts:

| Workload | What it approximates | Why include it |
| --- | --- | --- |
| `PP2048 @ depth 0` | Reading a short/medium 2K-token prompt | Shows prompt-processing performance for common requests |
| `PP8192 @ depth 0` | Reading a larger 8K-token prompt | Exposes configurations that behave differently on larger prefills |
| `TG256 @ depth 4096` | Generating 256 tokens with about 4K tokens already in context | Measures decode performance early in a conversation |
| `TG256 @ 50% context` | Generating 256 tokens around the middle of the Candidate's context window | Checks decode performance when the KV cache is substantially larger |

The `50% context` form is relative to each Candidate's configured context size. During planning, llama-profile-lab converts it into a concrete token depth while reserving room for the generated tokens and any configured safety margin. For a 128K Candidate with the default zero safety margin, `TG256 @ 50% context` becomes a depth of 65,408 tokens rather than simply 65,536.

These four cases give a useful first screen because they cover both major phases of inference—reading input and generating output—and test decode performance at both shallow and much deeper context. They are not universal requirements; choose workloads that resemble your real traffic once you understand the workflow.

With 11 Candidates, these four workloads produce **44 benchmark cases**: 11 Candidates × 4 workloads.

Use at least three repetitions for real comparisons. Before anything runs, use the plan/preview screen to confirm the Candidate and case counts look reasonable.

## 5. Run the experiment

For per-Candidate placement, llama-profile-lab first uses `llama-fit-params` at the Candidate's full production context. The resolved placement is then kept fixed across that Candidate's workloads.

Start the run in the UI, or use the CLI:

~~~bash
uv run --frozen llprof experiment run EXPERIMENT_ID \
  --binary BENCH_BINARY_ID \
  --fit-binary FIT_BINARY_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db
~~~

If the process is interrupted, resume it instead of starting over:

~~~bash
uv run --frozen llprof experiment resume EXPERIMENT_ID \
  --binary BENCH_BINARY_ID \
  --fit-binary FIT_BINARY_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db
~~~

Completed cases are preserved. OOMs, parser failures, interrupted attempts, and other failures remain in the database as useful diagnostic data.

## 6. Read the results

Start with the browser results page. Useful views include:

- batch × ubatch matrices;
- workload filters;
- Candidate-to-baseline comparison;
- CPU, GPU, memory, and stability metrics;
- Pareto views for multiple objectives.

There is deliberately no automatic “best profile” score. Different workloads trade off throughput, latency, memory, and stability, so you choose the objectives that matter for your deployment.

## 7. Validate a finalist

Microbenchmarks are for screening. Before changing a production launcher profile, validate a promising Candidate with `llama-server` and SPEED-Bench.

~~~bash
uv run --frozen llprof server validate EXPERIMENT_ID CANDIDATE_ID \
  --server-binary SERVER_BINARY_ID \
  --speed-bench-binary SPEED_BINARY_ID \
  --model-path /path/to/model.gguf \
  --placement PLACEMENT_ID \
  --model-name finalist \
  --database data/benchmarks.db
~~~

After successful validation, generate a reviewable launcher patch:

~~~bash
uv run --frozen llprof profile promote EXPERIMENT_ID CANDIDATE_ID \
  --launcher-config /path/to/launcher/config/hosts/workstation.json \
  --output candidate.patch \
  --database data/benchmarks.db
~~~

Promotion never silently edits your launcher configuration.

## 8. Keep the experiment reproducible

Check the database before archiving:

~~~bash
uv run --frozen llprof database check --database data/benchmarks.db
~~~

Create a portable snapshot:

~~~bash
uv run --frozen llprof archive \
  --output experiment-archive.tar.gz \
  --database data/benchmarks.db
~~~

Archives contain a consistent SQLite snapshot plus a hash manifest and can be restored into a new database path.

## Where to go next

If something fails, see [Troubleshooting](troubleshooting-v1.md). For the complete reference workflow, see [V1 benchmark workflow](benchmark-workflow-v1.md). For architectural background, see [V1 architecture](architecture-v1.md).
