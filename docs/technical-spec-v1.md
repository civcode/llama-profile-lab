# llama-profile-lab — V1 Technical Specification

Status: Draft for implementation  
Repository: civcode/llama-profile-lab  
CLI name: llprof  
Primary implementation language: Python 3.12+  
Primary datastore: SQLite  
UI: local web application backed by the same Python service and SQLite database

## 1. Purpose

llama-profile-lab is a local experiment, benchmarking, and tuning environment for llama.cpp profiles.

Its purpose is not merely to run llama-bench. It should provide a reproducible scientific record of model-performance experiments across:

- model and quantization choices;
- llama.cpp runtime parameters;
- placement and memory-fitting choices;
- prompt/prefill workloads;
- token-generation workloads;
- context depths;
- speculative decoding configurations;
- CPU and GPU utilization;
- llama.cpp builds;
- hardware and driver changes;
- repeated measurements over time.

The system must make simple experiments easy while retaining enough structure to represent arbitrary N-dimensional parameter sweeps and workload spaces.

The primary UX principle is:

> Simple experiments should feel like filling out a small form, while the underlying model remains expressive enough for arbitrarily high-dimensional sweeps.

Users should normally think in terms such as “vary batch size and ubatch size” and “measure four workloads,” not database IDs, hashes, or llama.cpp implementation details.

## 2. V1 goals

V1 SHALL support:

1. Importing and resolving model profiles from llama-profile-launcher.
2. Defining a base candidate and varying one or more tunable parameters.
3. N-dimensional grid search with constraints.
4. Candidate generation independent of workload generation.
5. llama-fit-params placement resolution against the production context size.
6. Frozen-placement llama-bench execution.
7. Prompt-processing and token-generation benchmarks.
8. Multiple context depths.
9. Multiple repetitions and preservation of every individual sample.
10. CPU and GPU telemetry during runs.
11. SQLite persistence of all experiment metadata, plans, execution records, raw results, normalized metrics, failures, and provenance.
12. Resumable experiments.
13. Comparison against a baseline candidate.
14. Pareto-style analysis across competing metrics.
15. Full llama-server validation for finalists.
16. SPEED-Bench support for end-to-end and speculative-decoding validation.
17. A CLI suitable for scripting and automation.
18. A local browser UI suitable for experiment creation and multidimensional result exploration.
19. Capability detection per llama.cpp binary.
20. Long-form llama.cpp arguments wherever a long form exists.

## 3. Non-goals for V1

V1 does not need:

- distributed execution across multiple hosts;
- remote multi-user authentication;
- cloud-hosted persistence;
- generic cluster scheduling;
- automatic modification of llama-profile-launcher profiles without an explicit user action;
- Bayesian optimization;
- a plugin system for arbitrary inference engines other than llama.cpp;
- a literal visualization of dimensions above three.

The architecture SHOULD leave room for these features without requiring a redesign of the persistence model.

## 4. Core conceptual model

The system models benchmarking as observations over three independent spaces:

~~~text
Candidate space
    ×
Workload space
    ×
Execution environment
    ↓
Observation vector
~~~

A candidate is one concrete point in the tunable-parameter space.

A workload case is one concrete point in the workload space.

An execution environment includes the host, llama.cpp binary/build, resolved placement, measurement policy, and runtime state.

An observation may contain multiple values:

~~~text
tokens/s
elapsed time
CPU utilization
GPU utilization
VRAM
RAM
power
temperature
clock rates
speculative acceptance
latency
...
~~~

The persistence layer MUST store observed points, not dense matrices or tensors. Matrix, cube, tensor, heatmap, slice, and projection views are derived from stored observations.

This makes sparse search spaces natural and allows higher-order searches without schema changes.

## 5. Architectural principles

### 5.1 Planning and execution are separate

The planner is pure application logic.

It SHALL:

- resolve a source profile;
- apply experiment dimensions;
- validate constraints;
- create immutable candidates;
- expand workload suites into concrete workload cases;
- determine whether placement resolution is required;
- create planned `benchmark_case` rows for llama-bench-compatible microbench workloads;
- retain server-only workloads such as `speed-bench` in `experiment_workload` without converting them into llama-bench cases;
- persist the plan before execution begins.

The planner SHALL NOT launch subprocesses.

The microbenchmark executor SHALL consume planned `benchmark_case` rows from SQLite. Server validation SHALL independently consume persisted server-workload links and create `server_run` / `server_benchmark` observations.

### 5.2 Experiments are immutable after planning

Before planning, a draft experiment may be edited.

A successful plan operation atomically persists the generated Candidates, concrete workloads, and benchmark cases, changes the experiment state to `planned`, and records `frozen_at`.

Once planned, changing the experiment definition means cloning it into a new experiment. Execution therefore consumes a stable persisted plan rather than re-expanding a mutable definition.

### 5.3 Historical observations are append-only

Completed benchmark runs SHALL NOT be overwritten.

A rerun creates a new run.

Failures are retained as data.

### 5.4 Raw measurements are authoritative

Optimization objectives are analysis policies, not properties of a run.

The system MUST preserve the underlying observations so results can later be reevaluated using different objectives.

For example, the same historical data may later be used to optimize for:

- highest prefill throughput;
- highest decode throughput;
- request latency;
- CPU usage below a threshold;
- lowest power;
- best long-context behavior;
- a Pareto frontier across multiple metrics.

### 5.5 Requested placement and resolved placement are different entities

A candidate stores the requested placement policy.

llama-fit-params produces a resolved placement for a particular host/build/configuration.

The resolved placement is stored separately and referenced by benchmark runs.

### 5.6 Production context and active depth are different concepts

Candidate context size represents production maximum context.

Workload depth represents active context at the point being benchmarked.

Production context SHALL be used when resolving memory placement.

Active depth SHALL be used for llama-bench workload execution.

### 5.7 SQLite is the canonical experiment record

JSON, JSONL, CSV, logs, and reports are import/export or artifact formats.

SQLite is the source of truth for experiment history.

## 6. Technology decisions

### 6.1 Backend

- Python 3.12+
- uv for Python selection, virtual-environment management, dependency resolution, locking, and development/CI command execution
- Pydantic v2 for public schemas and validation
- sqlite3 from the standard library for core persistence
- FastAPI for the local HTTP API
- subprocess-based process execution with shell disabled
- standard logging with structured context
- pathlib for paths
- hashlib for content identities
- JSON canonicalization for content-addressed objects

`pyproject.toml` SHALL remain the canonical project/dependency declaration. Development-only tooling SHALL use the standardized `dev` dependency group.

`uv.lock` SHALL be committed. Local development and CI SHALL use the lockfile in frozen mode so dependency resolution cannot silently drift between benchmark-tool revisions.

`.python-version` SHALL pin the repository's default Python line to 3.12. uv may provision the requested interpreter when it is not already installed.

setuptools remains the V1 Python build backend. Adopting uv does not require changing the build backend.

SQLAlchemy is not required in V1.

A thin repository layer SHALL isolate raw SQL from domain logic.

### 6.2 Frontend

- React
- TypeScript
- a small charting library selected during implementation
- API served by the local Python process
- local-only binding to 127.0.0.1 by default

### 6.3 CLI

Executable name:

~~~text
llprof
~~~

The CLI and web UI MUST use the same service/domain layer.

Neither surface may contain independent experiment logic.

## 7. Repository structure

Recommended initial structure:

~~~text
llama-profile-lab/
  pyproject.toml
  uv.lock
  .python-version
  README.md

  docs/
    technical-spec-v1.md

  src/
    llama_profile_lab/
      domain/
        candidate.py
        workload.py
        experiment.py
        search_space.py
        placement.py
        measurement.py
        run.py
        parameters.py

      planning/
        planner.py
        expand.py
        constraints.py
        strategies.py

      execution/
        process.py
        executor.py
        scheduler.py
        lock.py
        telemetry.py

      llama/
        capabilities.py
        args.py
        bench.py
        fit_params.py
        server.py
        speed_bench.py
        launcher.py

      db/
        connection.py
        repositories.py
        migrations.py

      analysis/
        metrics.py
        statistics.py
        pareto.py
        latency.py
        projections.py

      api/
        app.py
        experiments.py
        runs.py
        results.py
        profiles.py

      cli/
        main.py

  migrations/
    001_initial.sql
    002_telemetry.sql
    ...

  frontend/
    ...

  tests/
    unit/
    integration/
    fixtures/
~~~

## 8. Canonical JSON and content identities

Candidates, workload cases, measurement policies, and other reusable immutable definitions SHALL have canonical JSON representations.

Canonicalization:

~~~python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
~~~

Identity:

~~~text
sha256(canonical_json_utf8)
~~~

A short display ID MAY use a prefix of the SHA-256 hash, but the full hash SHALL be stored.

Content hashes SHALL exclude labels, descriptions, display order, experiment names, timestamps, and other non-semantic metadata.

Run identity is different: the same condition may be executed many times, so each run receives a unique event ID.

## 9. Candidate schema

A Candidate is an immutable performance-relevant configuration.

Candidate schema version 1:

~~~json
{
  "schema": "llama-profile-candidate",
  "version": 1,

  "model": {
    "target_model_id": "model:sha256:...",
    "draft_model_id": null
  },

  "context": {
    "size": 131072,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "kv_offload": true,
    "kv_unified": true
  },

  "compute": {
    "flash_attn": "on",
    "batch_size": 4096,
    "ubatch_size": 2048,
    "threads": 16,
    "load_mode": "mmap",
    "lazy_mode": "on",
    "repack": true,
    "no_host": false,
    "no_op_offload": false
  },

  "placement": {
    "mode": "fit",

    "fit": {
      "target_mib": 256,
      "min_context": 4096
    },

    "constraints": {
      "n_gpu_layers": "auto",
      "n_cpu_moe": 0,
      "split_mode": "layer",
      "main_gpu": 0,
      "devices": "auto",
      "tensor_split": null,
      "override_tensor": []
    }
  },

  "server": {
    "parallel": 1
  },

  "speculative": {
    "enabled": false,
    "type": null,
    "draft_n_max": null
  },

  "extra_args": {}
}
~~~

### 9.1 Candidate validation

At minimum:

- context.size > 0
- batch_size > 0
- ubatch_size > 0
- ubatch_size <= batch_size
- threads > 0 when set
- server.parallel >= 1
- speculative.draft_n_max is null when speculative is disabled
- placement.fit is non-null when placement.mode is fit
- fixed-placement-only fields are rejected when incompatible with the selected mode
- parameter registry constraints are satisfied

### 9.2 Effective defaults

Performance-relevant defaults SHALL be resolved before Candidate creation whenever feasible.

The system MUST distinguish:

- source profile did not specify a value;
- launcher inheritance supplied a value;
- llama.cpp binary default supplied a value;
- the user explicitly overrode a value;
- the system derived a value.

Provenance MAY be stored separately from Candidate identity.

### 9.3 extra_args

extra_args is an escape hatch for supported llama.cpp options not yet represented by typed Candidate fields.

Requirements:

- values participate in Candidate hashing;
- shell strings are forbidden;
- each item is represented as an argument name plus typed/string value;
- capability checks still apply;
- the UI places these values in an Advanced section.

## 10. Parameter registry

Performance parameters SHALL be described centrally.

Example conceptual definition:

~~~python
ParameterDefinition(
    path="compute.ubatch_size",
    value_type=int,
    cli_argument="--ubatch-size",
    minimum=1,
    affects_placement=True,
    supported_by={
        "llama-bench",
        "llama-server",
        "llama-fit-params",
    },
    category="Compute",
    label="Physical batch size",
)
~~~

A speculative parameter:

~~~python
ParameterDefinition(
    path="speculative.draft_n_max",
    value_type=int,
    cli_argument="--spec-draft-n-max",
    affects_placement=False,
    supported_by={"llama-server"},
    category="Speculative decoding",
    condition="speculative.enabled == true",
)
~~~

The registry is the source for:

- backend validation;
- search-space validation;
- CLI argument generation;
- capability matching;
- frontend controls;
- descriptions and categories;
- whether changing a value requires refitting;
- applicability conditions.

## 11. Search-space schema

A SearchSpace represents an N-dimensional tunable parameter space.

~~~json
{
  "schema": "llama-search-space",
  "version": 1,

  "dimensions": [
    {
      "path": "compute.batch_size",
      "values": [2048, 4096, 8192]
    },
    {
      "path": "compute.ubatch_size",
      "values": [512, 1024, 2048, 4096]
    }
  ],

  "constraints": [
    "compute.ubatch_size <= compute.batch_size"
  ],

  "strategy": {
    "type": "grid"
  }
}
~~~

V1 SHALL implement deterministic grid expansion.

The architecture SHALL allow future strategies such as:

- random sampling;
- coarse-to-fine search;
- successive halving;
- Latin hypercube sampling;
- Bayesian optimization.

Search strategy changes SHALL NOT alter Candidate identity or persistence.

### 11.1 Conditional dimensions

The schema SHALL support dimensions that only apply when a condition is true.

Example:

~~~json
{
  "path": "speculative.draft_n_max",
  "values": [1, 2, 3, 4],
  "condition": "speculative.enabled == true"
}
~~~

Grid dimensions are expanded in declared order. A conditional dimension is evaluated against the base Candidate plus assignments from earlier dimensions. If its condition is false, that dimension is skipped and does not multiply the Cartesian product.

This makes dimension ordering semantically relevant when one dimension controls the applicability of a later dimension.

### 11.2 Constraint language

V1 MAY implement constraints using an internal typed expression model rather than evaluating arbitrary Python.

Arbitrary eval SHALL NOT be used.

The V1 implementation supports:

- equality and inequality;
- numeric and string ordering comparisons;
- boolean conjunction/disjunction and boolean `not`;
- membership and non-membership;
- list/tuple/set literals;
- references to Candidate paths;
- lowercase `true`, `false`, and `null`/ `none` literals.

Expressions are parsed with Python's AST module and interpreted by an explicit whitelist. Calls, arithmetic, indexing, comprehensions, and arbitrary evaluation are rejected.

## 12. WorkloadSuite schema

A WorkloadSuite is human-authored and may contain relative depth expressions.

Example:

~~~json
{
  "schema": "llama-workload-suite",
  "version": 1,

  "id": "batch-ubatch-screen-v1",
  "description": "Screen batch and ubatch tradeoffs",

  "cases": [
    {
      "label": "pp-2k",
      "kind": "microbench-prefill",
      "prompt_tokens": 2048,
      "depth": {
        "type": "absolute",
        "tokens": 0
      }
    },
    {
      "label": "pp-8k",
      "kind": "microbench-prefill",
      "prompt_tokens": 8192,
      "depth": {
        "type": "absolute",
        "tokens": 0
      }
    },
    {
      "label": "tg-short",
      "kind": "microbench-decode",
      "generate_tokens": 256,
      "depth": {
        "type": "absolute",
        "tokens": 4096
      }
    },
    {
      "label": "tg-mid",
      "kind": "microbench-decode",
      "generate_tokens": 256,
      "depth": {
        "type": "fraction",
        "value": 0.50
      }
    }
  ]
}
~~~

## 13. WorkloadCase schema

WorkloadSuite entries are expanded against a Candidate into immutable concrete WorkloadCases.

No percentages, symbolic values, or implicit defaults remain after expansion.

### 13.1 Prefill case

~~~json
{
  "schema": "llama-workload-case",
  "version": 1,
  "kind": "microbench-prefill",
  "prompt_tokens": 2048,
  "generate_tokens": 0,
  "depth_tokens": 32768
}
~~~

### 13.2 Decode case

~~~json
{
  "schema": "llama-workload-case",
  "version": 1,
  "kind": "microbench-decode",
  "prompt_tokens": 0,
  "generate_tokens": 256,
  "depth_tokens": 65408
}
~~~

### 13.3 Combined case

~~~json
{
  "schema": "llama-workload-case",
  "version": 1,
  "kind": "microbench-combined",
  "prompt_tokens": 8192,
  "generate_tokens": 512,
  "depth_tokens": 16384
}
~~~

### 13.4 SPEED-Bench case

~~~json
{
  "schema": "llama-workload-case",
  "version": 1,
  "kind": "speed-bench",

  "speed_bench": {
    "bench": "throughput_32k",
    "categories": ["all"],
    "output_tokens": 1024,
    "concurrency": 1,
    "limit": 16,
    "request": {
      "temperature": 0
    }
  }
}
~~~

## 14. Relative-depth expansion

For a microbenchmark suite entry with a fractional depth:

~~~text
available_depth =
    candidate.context.size
    - prompt_tokens
    - generate_tokens
    - safety_margin_tokens

depth_tokens =
    floor(available_depth × fraction)
~~~

Every concrete workload MUST satisfy:

~~~text
depth_tokens
+ prompt_tokens
+ generate_tokens
<= candidate.context.size
~~~

Expansion provenance SHALL be recorded separately from workload identity.

Example provenance:

~~~json
{
  "suite_id": "batch-ubatch-screen-v1",
  "suite_case_index": 3,
  "candidate_context_size": 131072,
  "original_depth": {
    "type": "fraction",
    "value": 0.5
  }
}
~~~

Labels and expansion metadata do not participate in WorkloadCase hashing.

## 15. MeasurementPolicy schema

Measurement policy is deliberately separate from workload identity.

~~~json
{
  "schema": "llama-measurement-policy",
  "version": 1,

  "warmup": true,
  "repetitions": 3,
  "delay_seconds": 0,

  "adaptive": null
}
~~~

Future adaptive policy example:

~~~json
{
  "warmup": true,
  "repetitions": null,
  "delay_seconds": 0,

  "adaptive": {
    "minimum_repetitions": 3,
    "maximum_repetitions": 10,
    "target_relative_error": 0.01
  }
}
~~~

Fixed repetitions are sufficient for initial V1 execution, but the domain model SHOULD permit adaptive stopping.

## 16. Experiment schema

An Experiment connects a base Candidate, a SearchSpace, a WorkloadSuite, a MeasurementPolicy, and execution policy.

~~~json
{
  "schema": "llama-tuning-experiment",
  "version": 1,

  "name": "Flash Next 128K batch/ubatch sweep",

  "base_candidate_id": "cand_...",

  "search_space_id": "space_...",
  "workload_suite_id": "suite_...",
  "measurement_policy_id": "measure_...",

  "placement_policy": {
    "type": "per-candidate"
  },

  "baseline": {
    "type": "base-candidate"
  }
}
~~~

Supported placement policies in V1:

### per-candidate

Every candidate that changes placement-sensitive parameters is resolved independently against the production context.

This is the normal deployment-optimization mode.

### fixed

All candidates use a specified resolved placement.

This is useful for controlled parameter-isolation experiments.

~~~json
{
  "placement_policy": {
    "type": "fixed",
    "placement_id": "place_..."
  }
}
~~~

## 17. Batch/ubatch example

For Flash Next 128K:

~~~text
batch_size  = 2048, 4096, 8192
ubatch_size = 512, 1024, 2048, 4096

constraint:
ubatch_size <= batch_size
~~~

This yields eleven valid Candidate points.

With four screening workloads:

~~~text
PP2048 @ d0
PP8192 @ d0
TG256  @ d4096
TG256  @ d50%
~~~

the plan contains:

~~~text
11 candidates × 4 workloads = 44 benchmark cases
~~~

With three repetitions:

~~~text
44 cases × 3 samples = 132 timed samples
~~~

The 2D matrix is a presentation of these observations, not their storage form.

Adding KV type produces a 3D space. Adding flash attention, fit target, context, or another parameter produces higher orders without changing the observation model.

## 18. Model identity

A model record SHALL be content-based where practical.

Store:

- logical model ID;
- architecture;
- parameter count;
- quantization;
- metadata JSON;
- total size;
- file records.

For split GGUFs, each shard SHALL be represented in model_file.

A model identity MUST reflect all required shards.

A speculative draft model is another model record referenced by model ID rather than an arbitrary path.

## 19. Binary/build identity and capability detection

Each llama.cpp executable SHALL have a binary record containing:

- path;
- SHA-256;
- size;
- mtime;
- tool kind;
- git commit if discoverable;
- build number if discoverable;
- branch if discoverable;
- dirty state if discoverable;
- compiler/build information when advertised;
- backend information when advertised;
- captured help/version output.

The exact executable SHA-256 is the durable identity. Re-registering the same hash MAY refresh its current path and probe metadata without creating a second binary identity. A modified or custom executable receives a different binary ID even when its filename/tool kind is the same.

Capabilities SHALL be detected per binary rather than assumed globally. The V1 discovery probe invokes metadata commands with argv arrays and `shell=False`, captures both stdout and stderr, and applies a bounded timeout.

At minimum inspect:

- `llama-bench --help`
- `llama-server --help`
- `llama-fit-params --help` where available
- `--version` for build metadata

If `--help` exits unsuccessfully without producing output, discovery MAY retry `-h`. A non-zero help exit code with useful output does not by itself invalidate the capability surface.

Supported arguments are parsed from option declarations in the captured help text. Both short aliases and long-form names are retained. Raw help/version output remains persisted so future parsers can reinterpret historical discoveries.

The UI MUST expose unsupported parameters as unavailable for the selected binary rather than silently dropping them.

Custom Qwen branches and upstream builds may therefore coexist and can be compared by their advertised option sets.

## 20. llama.cpp adapters

llama.cpp behavior SHALL be encapsulated behind adapters:

~~~text
LlamaBenchAdapter
LlamaFitParamsAdapter
LlamaServerAdapter
SpeedBenchAdapter
LauncherProfileAdapter
~~~

The experiment engine SHALL NOT know llama.cpp CLI spellings.

Adapters consume typed domain objects and return argv arrays plus typed parsed results.

Long-form arguments SHALL be emitted wherever supported.

The M5 LlamaBenchAdapter executes one concrete WorkloadCase per process invocation. It explicitly supplies prompt, generation, depth, batch, ubatch, KV-cache, repetition, and JSON-output settings so a single invocation produces exactly one normalized result object.

M5 intentionally does not pass fit-target or otherwise invoke automatic placement fitting. Concrete placement resolution is introduced in M6. Explicit placement constraints already present on the Candidate may still be emitted where supported.

Until model-registry/launcher path resolution is wired into the executor, the M5 CLI requires an explicit --model-path. The Candidate model identity remains the semantic configuration reference; the actual process argv is stored for provenance.

llama-bench JSON output is preserved in benchmark_run.raw_result_json and stdout, while samples_ns/samples_ts are normalized into benchmark_sample rows and aggregate avg/stddev fields are stored as metrics.

Example llama-bench argv:

~~~text
llama-bench
--model /path/model.gguf
--flash-attn on
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 4096
--ubatch-size 1024
--n-gpu-layers 47
--n-prompt 0
--n-gen 256
--n-depth 32768
--repetitions 7
--output json
~~~

argv arrays, not shell strings, are authoritative.

## 21. Placement resolution

The normal V1 deployment workflow is:

~~~text
Candidate
   ↓
llama-fit-params using production context
   ↓
ResolvedPlacement
   ↓
llama-bench at selected active depths
~~~

Placement resolution is performed before the first benchmark workload for a Candidate. The fit invocation pins `--ctx-size` to the Candidate's full production context. A successful fit is reused for every shallow and deep microbenchmark workload for that Candidate.

### 21.1 Fit input

LlamaFitParamsAdapter SHALL derive fit input from the effective Candidate and the selected fit binary's advertised capabilities.

V1 includes, where supported:

- model path;
- production context size;
- batch and ubatch size;
- K/V cache types;
- KV offload;
- flash attention;
- load/lazy mode;
- operation offload;
- host-buffer behavior;
- repack setting;
- fit target;
- fit minimum context;
- device/split/main-GPU constraints;
- compatible extra arguments.

Boolean options for llama-fit-params use the common llama.cpp flag pairs advertised by the selected binary, for example `--kv-offload` / `--no-kv-offload`, `--op-offload` / `--no-op-offload`, and `--repack` / `--no-repack`. They are not rendered with llama-bench's numeric boolean convention.

Automatic fitting rejects pre-fixed GPU-layer/tensor placement constraints that would conflict with the fit result. Use fixed-placement experiment policy for controlled comparisons with an already resolved placement.

### 21.2 Fit output and attempts

Each actual llama-fit-params invocation is an append-only placement_attempt recording:

- placement hash;
- Candidate/host/binary references;
- model path;
- argv;
- start/finish timestamps;
- duration;
- terminal status;
- exit code;
- stdout/stderr;
- parsed raw result where available.

Failed, timed-out, interrupted, cancelled, and parser-failed attempts remain observable and do not populate the successful placement cache.

The current llama-fit-params adapter reads the final fitted CLI argument line from stdout using shell-compatible tokenization. V1 normalizes:

- context size;
- concrete n_gpu_layers;
- tensor split;
- one or more tensor-override arguments.

Raw output is retained so future parser versions can reinterpret historical attempts.

### 21.3 Resolved placement and cache identity

A successful ResolvedPlacement records:

- source Candidate ID;
- host ID;
- fit binary ID and fit-attempt ID;
- production context size;
- resolved n_gpu_layers;
- n_cpu_moe;
- split mode;
- main GPU;
- device selection;
- tensor split;
- tensor overrides;
- fit argv;
- stdout/stderr/exit status;
- raw parsed result;
- complete request provenance;
- creation timestamp.

The placement cache hash is derived only from fit-relevant identity:

- target model identity plus model artifact size/mtime safety metadata;
- production context;
- K/V cache types and KV offload;
- flash attention;
- batch and ubatch size;
- fit target/minimum context;
- load/lazy/repack/host/op-offload settings;
- placement constraints and fit-relevant extra args;
- host hardware fingerprint;
- exact llama-fit-params executable SHA-256.

Non-fit Candidate settings such as server parallelism and speculative-decoding configuration do not invalidate the placement cache. The mounted model path is preserved as provenance but is not itself part of the cache hash.

The Linux host hardware fingerprint includes stable DRM/PCI GPU identity when available. Transient DRM card numbering is retained as metadata but excluded from the fingerprint identity.

### 21.4 Per-candidate and fixed policies

For `per-candidate` policy, the executor lazily resolves placement when it reaches the first incomplete workload for a Candidate. This avoids fitting Candidates that are not reached by a limited/smoke run. Persistent cache lookup occurs before running the fit process.

For `fixed` policy, the experiment references an existing ResolvedPlacement. V1 requires that fixed placement to belong to the current host hardware identity and to match the Candidate production context. It may intentionally originate from another Candidate so controlled placement comparisons are possible.

After resolution, the placement is bound to all benchmark cases for that Candidate. Concrete placement values are authoritative in llama-bench argv generation. Candidate extra_args that would re-enable fitting or override resolved placement are rejected.

A shallow llama-bench invocation SHALL NOT silently refit placement for a production-context experiment.

## 22. Process execution

All subprocesses SHALL use argv arrays and shell=False.

A single ProcessRunner abstraction SHALL own:

- start timestamp;
- PID;
- stdout capture;
- stderr capture;
- exit code;
- timeout;
- process-group handling;
- graceful termination;
- forced termination;
- cancellation;
- cleanup;
- final status.

The same abstraction SHALL be used for bench, fit, server, and helper processes.

M5 launches every process in a new process group. Timeout, cancellation, and keyboard interruption first send SIGTERM to that group and escalate to SIGKILL after the configured grace period.

A subprocess launch failure is persisted as benchmark_failed rather than leaving a running row behind.

Immediately before benchmark execution, the selected registered llama-bench path SHALL be re-hashed and MUST still match its stored binary SHA-256. If it changed in place, execution stops and the user must register the new executable identity.

M5 executes one benchmark case at a time and holds a shared host-execution lock across the execution session. The lock is shared across databases for the same local user so two databases cannot independently launch benchmark work and contaminate each other's measurements.

## 23. Host ownership and interference

By default, a performance experiment owns the benchmark host.

The scheduler SHALL acquire a host-level lock before running performance work.

V1 SHALL execute only one benchmark case at a time by default.

The scheduler SHOULD detect:

- an existing llama-server;
- significant external GPU utilization;
- significant external CPU utilization;
- insufficient free memory.

Policy may be configured to:

- warn;
- wait;
- proceed and mark the run noisy;
- fail before execution.

## 24. Run states

A run SHALL have a typed state.

Minimum states:

~~~text
planned
running
completed
oom
timeout
invalid
fit_failed
load_failed
benchmark_failed
parser_failed
interrupted
cancelled
~~~

Failures are persisted and never discarded.

An orphaned running row found after application restart SHALL be converted to interrupted unless the process can be positively reattached.

## 25. Experiment states

Recommended experiment state machine:

~~~text
draft
  ↓
planned
  ↓
running
  ├── paused
  │     ↓
  │   running
  ↓
completed

running → cancelled
running → failed
~~~

“failed” means the orchestration itself could not continue, not merely that one candidate was OOM.

Experiments may complete with failed individual cases.

## 26. Telemetry

Telemetry is first-class because Flash Next and other architectures may use both CPU and GPU substantially.

Telemetry observations belong to a BenchmarkRun. They do not participate in Candidate, WorkloadCase, or placement identity.

Sampling is independent of benchmark computation and SQLite writes: the sampler collects in memory on a background thread and persists the complete observation set after the subprocess finishes. This avoids sharing a SQLite connection across threads.

Default sampling interval:

~~~text
1000 ms
~~~

V1 supports a minimum interval of:

~~~text
500 ms
~~~

The CLI exposes `--telemetry-interval-ms` on experiment run/resume.

### 26.1 Sampler lifecycle

For every executable benchmark run:

~~~text
before host snapshot
        ↓
Popen llama-bench
        ↓
benchmark PID handed to telemetry sampler
        ↓
immediate process-aware sample
        ↓
periodic process-aware samples
        ↓
process exits
        ↓
after host snapshot
        ↓
persist raw samples + summary + quality
~~~

Telemetry errors SHALL NOT convert a successful benchmark into a failed benchmark. They are recorded in quality details and may produce `telemetry_incomplete`.

### 26.2 CPU and process telemetry

LinuxTelemetryProvider uses `/proc` and `/sys` without requiring psutil.

Capture when available:

- total system CPU utilization percent;
- CPU user percent;
- CPU system percent;
- CPU iowait percent;
- process CPU percent normalized to the whole machine;
- process CPU percent in raw per-core convention;
- process user time;
- process system time;
- process thread count;
- per-core utilization;
- average/min/max CPU frequency;
- CPU/package temperature;
- load averages;
- CPU package power through Intel RAPL when reliably exposed.

Normalized process CPU semantics:

~~~text
0–100% represents fraction of total machine CPU capacity
~~~

Raw process CPU semantics follow the common per-core convention:

~~~text
100% ≈ one fully occupied logical CPU
800% ≈ eight fully occupied logical CPUs
~~~

The initial external-CPU estimate is:

~~~text
external_cpu_pct =
    max(0, system_cpu_pct - process_cpu_pct_normalized)
~~~

This is a quality heuristic, not exact process attribution.

### 26.3 Memory telemetry

Capture when available:

- RAM used;
- RAM available;
- swap used;
- benchmark-process RSS.

### 26.4 GPU telemetry

GPU collection is provider-based.

The automatic Linux provider prefers NVIDIA management telemetry through:

~~~text
nvidia-smi -q -x
~~~

It captures when available:

- GPU identity/name/UUID;
- utilization;
- VRAM used/total;
- temperature;
- power draw;
- graphics clock;
- memory clock;
- clock-event/throttle reasons.

Explicit thermal clock-event reasons are normalized to a thermal-throttle signal.

When NVIDIA management telemetry is unavailable, the generic DRM/sysfs provider captures available Linux GPU fields such as:

- `gpu_busy_percent`;
- `mem_info_vram_used`;
- `mem_info_vram_total`;
- hwmon temperature;
- hwmon power.

Per-GPU observations remain in `gpu_json` so vendor-specific fields can evolve without a schema migration.

### 26.5 Persistence and summaries

Each telemetry_sample stores normalized CPU/process/memory columns plus:

- per-core CPU percentages as JSON;
- per-GPU observations as JSON;
- optional provider-specific fields in extra JSON;
- sampling phase (`before`, `during`, or `after`) in extra JSON.

After a run, normalized `telemetry.*` metrics are written to the generic metric table. V1 summaries include:

- sample counts;
- average/peak system CPU;
- average/peak normalized process CPU;
- average/peak raw process CPU;
- CPU frequency range;
- peak CPU temperature;
- process user/system CPU time delta when enough samples exist;
- peak RAM/swap/process RSS;
- average/peak GPU utilization;
- peak GPU temperature;
- peak VRAM used;
- average/peak GPU power;
- estimated peak external CPU load;
- peak out-of-run GPU baseline utilization.

### 26.6 Run-quality classification

Initial primary labels:

~~~text
clean
noisy
external_cpu_load
external_gpu_load
thermal_throttle
telemetry_incomplete
~~~

The primary label does not replace raw signals. All detected reasons and the full summary are retained in `quality_details_json`.

Default V1 thresholds are intentionally conservative and configurable in code:

~~~text
external CPU       >= 20% of total machine capacity
CPU noise          >= 10%
external GPU       >= 15% baseline utilization
GPU noise          >= 5% baseline utilization
CPU thermal signal >= 95 C
GPU thermal signal >= 90 C
~~~

Explicit GPU thermal-throttle reasons also trigger `thermal_throttle`.

Out-of-run GPU baseline is measured from before/after snapshots. Because GPU process-utilization attribution is not portable across vendors, V1 does not subtract benchmark GPU utilization from total GPU utilization.

Quality priority when several conditions are observed:

~~~text
thermal_throttle
external_cpu_load
external_gpu_load
telemetry_incomplete
noisy
clean
~~~

A quality label never silently deletes a run and never changes `completed` into a failure state.

## 27. Derived run summaries and analysis

Analysis is read-only application logic over the canonical SQLite observation set. It SHALL NOT mutate Candidate identity, workload identity, placement, raw benchmark samples, telemetry, or historical run status.

For each completed run, the analysis layer can derive or expose:

Performance:

- mean tokens/s;
- median tokens/s;
- sample standard deviation;
- coefficient of variation;
- min;
- max;
- sample count.

CPU:

- mean/peak system CPU;
- mean/peak benchmark-process CPU;
- CPU user time;
- CPU system time;
- CPU seconds per 1000 measured tokens where applicable;
- CPU temperature and frequency metrics already normalized by telemetry.

GPU and memory:

- mean/peak utilization;
- peak VRAM;
- peak process RSS and RAM;
- temperature summaries;
- average/peak power;
- an energy estimate where run duration and average GPU power are available.

These values are derived from preserved raw samples and normalized run metrics and can be recomputed.

### 27.1 Metric registry and repetition semantics

MetricDefinition is the central analysis registry for human-facing metric name, label, unit, source metric, reduction policy, and optional derived calculation.

For throughput statistics, individual timed benchmark repetitions are authoritative. When an identical Candidate/workload coordinate has multiple successful runs, their individual benchmark samples are pooled before calculating mean, median, standard deviation, coefficient of variation, min, max, and sample count.

Resource metrics already normalized per run are combined according to metric semantics. Typical utilization averages use a mean across matching runs, while peak temperature, peak CPU, peak VRAM, and similar maxima retain a maximum reduction.

A persisted scalar metric not yet present in the built-in registry remains analyzable through a generic mean-aggregation fallback. Adding a new normalized telemetry metric therefore does not require a new database schema or block basic analysis.

### 27.2 Analysis coordinates and filters

An analysis observation combines:

~~~text
Candidate coordinates
Workload coordinates
Run/environment coordinates
Metric values
~~~

V1 exact-match filters may reference scalar paths such as:

~~~text
compute.batch_size
compute.ubatch_size
context.cache_type_k
workload.kind
workload.prompt_tokens
workload.generate_tokens
workload.depth_tokens
workload.suite_case_index
run.host_id
run.binary_id
run.quality
~~~

Candidate paths may also use an explicit `candidate.` prefix.

Run quality is an explicit filter rather than an implicit deletion rule. For example, a caller may analyze only `clean` runs or may deliberately include noisy historical data.

### 27.3 Sparse matrix, tensor slices, and facets

The database stores observed points; it never stores dense matrices or tensors.

A two-dimensional projection chooses:

~~~text
X Candidate path
Y Candidate path
metric
filters
optional facet path
~~~

and returns a sparse set of cells. Missing invalid, failed, pruned, or unmeasured coordinates remain missing.

The projector SHALL NOT silently average across hidden Candidate or workload dimensions. If an X/Y cell still contains more than one distinct Candidate configuration or more than one workload-suite case, the projection is ambiguous and SHALL fail with an instruction to add a filter or facet.

Repeated successful runs of the same Candidate/workload coordinate may be pooled according to the selected metric's reduction semantics.

Higher-dimensional exploration is therefore represented as:

~~~text
X × Y
+ exact filters
+ optional facet
~~~

rather than an attempt to visualize an arbitrary N-dimensional tensor directly.

### 27.4 Baseline comparison

An experiment baseline is either:

- the configured base Candidate; or
- an explicitly selected Candidate.

Candidate comparison aligns observations by `suite_case_index`, not by concrete WorkloadCase hash. This matters for Candidate-dependent workloads such as fractional context depth, whose concrete token depth can differ when Candidate context size differs.

For each requested metric, comparison returns:

~~~text
baseline value
candidate value
signed absolute delta
signed percent delta
~~~

The comparison layer does not select a winner.

### 27.5 Pareto frontier

Pareto analysis accepts one or more explicit user objectives. Every objective defines:

~~~text
key
maximize | minimize
metric
objective-specific filters
~~~

Example:

~~~text
PP8K throughput     maximize
TG @ 4K throughput maximize
process CPU         minimize
process RSS         minimize
~~~

An objective SHALL resolve to one workload-suite coordinate per Candidate. A broad objective that mixes PP2K and PP8K, or multiple decode depths, is rejected as ambiguous rather than averaged.

A Candidate enters the evaluated set only when all objective values are available. Planned Candidates with incomplete observations remain visible in the result as excluded with a reason.

The frontier contains exactly the non-dominated Candidates under the supplied objectives. The service SHALL NOT produce an overall score, ranking, or universal best Candidate unless a separate future optimization policy explicitly defines such a scalar objective.

### 27.6 Request-latency estimation

V1 estimates compute-only request latency from measured prefill and decode curves.

For prompt length `P`:

~~~text
T_pp(P) = P / interpolated_PP(P)
~~~

For `G` generated tokens beginning at active depth `D`:

~~~text
T_tg(D, G) =
    sum(i = 0 .. G - 1)
        1 / interpolated_TG(D + i)
~~~

Total estimated compute time is:

~~~text
T_total = T_pp + T_tg
~~~

Prefill interpolation uses measured d0 prefill points indexed by prompt tokens.

Decode interpolation uses measured decode points indexed by active depth. Linear interpolation is used between measured points. Outside the measured range, V1 clamps to the nearest measured endpoint rather than extrapolating an unbounded trend.

The estimator validates that the requested decode interval fits within the Candidate production context.

This is intentionally a compute model. It excludes tokenization, sampling, HTTP/network overhead, queueing, server concurrency effects, speculative-decoding server behavior, and other end-to-end costs. M9 server validation measures those separately.

### 27.7 Export

Analysis export produces one summarized row per Candidate × workload-suite coordinate, including:

- experiment/Candidate/workload identifiers;
- Candidate ordinal;
- suite-case index;
- concrete prompt/generation/depth fields;
- run count;
- observed quality labels;
- experiment search-space dimensions;
- requested summary metrics.

V1 supports deterministic JSON and CSV serialization without introducing pandas as a core dependency.

### 27.8 CLI

M8 exposes:

~~~text
llprof results metrics

llprof results matrix EXPERIMENT \
  --x compute.batch_size \
  --y compute.ubatch_size \
  --metric throughput.median \
  --filter workload.kind=microbench-prefill \
  --filter workload.prompt_tokens=8192

llprof results compare EXPERIMENT CANDIDATE \
  --metric throughput.median

llprof results pareto EXPERIMENT \
  --objective 'pp8k:max:throughput.median@workload.kind=microbench-prefill;workload.prompt_tokens=8192' \
  --objective 'tg4k:max:throughput.median@workload.kind=microbench-decode;workload.depth_tokens=4096'

llprof results latency EXPERIMENT \
  --candidate CANDIDATE \
  --prompt-tokens 4096 \
  --generate-tokens 256

llprof results export EXPERIMENT --format csv --output results.csv
~~~

All result commands operate on completed persisted observations and share the same analysis services intended for the later HTTP API and browser UI.

## 28. SQLite design principles

SQLite SHALL run with:

~~~sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
~~~

Schema changes SHALL use numbered migrations.

Applied migrations SHALL be recorded with their version, name, and SHA-256 checksum. An already-applied migration file is immutable; checksum drift SHALL be treated as an error and corrected with a new forward migration.

Migration SQL SHALL be packaged with installed distributions as well as available from source checkouts.

CREATE TABLE IF NOT EXISTS is not a migration strategy.

One database SHOULD contain all experiments for a host/project installation.

Recommended default location is configurable; a user-data location is preferable to placing runtime data in the source repository.

## 29. Core SQLite tables

The exact DDL belongs in implementation migrations, but V1 SHALL represent at least the following entities.

### host

~~~text
id
hostname
hardware_fingerprint
cpu_json
ram_bytes
gpu_json
os_json
created_at
~~~

### binary

~~~text
id
sha256
kind
path
size_bytes
mtime_ns
git_commit
git_branch
git_dirty
build_number
build_info_json
capabilities_json
created_at
~~~

### model

~~~text
id
identity_hash
architecture
parameter_count
quantization
size_bytes
metadata_json
created_at
~~~

### model_file

~~~text
id
model_id
part_index
path
sha256
size_bytes
~~~

### candidate

~~~text
id
config_hash UNIQUE
target_model_id
draft_model_id
context_size
batch_size
ubatch_size
cache_type_k
cache_type_v
flash_attn
fit_target_mib
config_json
created_at
~~~

Frequently queried fields are normalized into typed columns.

The complete canonical Candidate remains in config_json.

### search_space

~~~text
id
definition_hash UNIQUE
definition_json
created_at
~~~

SearchSpace is a first-class immutable content-addressed entity. Experiments reference it by ID so the same N-dimensional definition can be reused and compared across experiment events.

### workload_suite

~~~text
id
definition_hash
name
definition_json
created_at
~~~

### workload_case

~~~text
id
workload_hash UNIQUE
kind
prompt_tokens
generate_tokens
depth_tokens
definition_json
created_at
~~~

### measurement_policy

~~~text
id
policy_hash UNIQUE
definition_json
created_at
~~~

### experiment

~~~text
id
name
status
base_candidate_id
search_space_id
workload_suite_id
measurement_policy_id
placement_policy_json
baseline_json
definition_json
created_at
frozen_at
completed_at
~~~

### experiment_candidate

~~~text
experiment_id
candidate_id
ordinal
generation_metadata_json
~~~

### experiment_workload

~~~text
experiment_id
candidate_id
workload_case_id
suite_case_index
expansion_provenance_json
~~~

The Candidate reference is required because relative-depth workload expansion may produce different concrete WorkloadCases for different Candidate context sizes.

### placement_attempt

~~~text
id
placement_hash
candidate_id
host_id
binary_id
model_path
started_at
finished_at
duration_ns
status
exit_code
argv_json
stdout
stderr
raw_result_json
~~~

Fit attempts are append-only execution observations. Only successful parsed attempts produce or reuse a resolved_placement.

### resolved_placement

~~~text
id
placement_hash
candidate_id
host_id
binary_id
production_context_size
n_gpu_layers
n_cpu_moe
split_mode
main_gpu
tensor_split_json
override_tensor_json
argv_json
stdout
stderr
exit_code
raw_result_json
created_at
~~~

### benchmark_case

~~~text
id
experiment_id
candidate_id
workload_case_id
placement_id
case_hash
status
ordinal
~~~

### benchmark_run

~~~text
id
benchmark_case_id
host_id
binary_id
measurement_policy_id
started_at
finished_at
duration_ns
status
exit_code
argv_json
environment_json
stdout
stderr
raw_result_json
quality
quality_details_json
~~~

### benchmark_sample

~~~text
run_id
sample_index
elapsed_ns
tokens_per_second
~~~

### telemetry_sample

~~~text
run_id
timestamp_ns

cpu_system_pct
cpu_user_pct
cpu_system_mode_pct
cpu_iowait_pct
process_cpu_pct_normalized
process_cpu_pct_raw
process_user_time_ns
process_system_time_ns
process_threads
cpu_freq_avg_hz
cpu_freq_min_hz
cpu_freq_max_hz
cpu_temperature_c
load_avg_1m
load_avg_5m

ram_used_bytes
ram_available_bytes
swap_used_bytes
process_rss_bytes

gpu_json
cpu_per_core_json
extra_json
~~~

### metric

A generic fact table MAY be used for secondary or server metrics:

~~~text
run_id
metric_name
value_real
value_integer
unit
dimensions_json
~~~

High-value frequently queried metrics SHOULD remain normalized where appropriate.

### server_run

~~~text
id
experiment_id
candidate_id
placement_id
host_id
server_binary_id

target_model_id
draft_model_id
target_model_path
draft_model_path

spec_type
spec_draft_n_max

bind_host
bind_port

argv_json
environment_json

started_at
ready_at
finished_at
duration_ns

status
exit_code
stdout
stderr
~~~

V1 server-run states are:

~~~text
starting
ready
completed
start_failed
readiness_failed
benchmark_failed
interrupted
cancelled
~~~

### server_benchmark

~~~text
id
server_run_id
workload_case_id
speed_bench_binary_id
category
status

argv_json
duration_ns
exit_code
stdout
stderr

requests
failed
turns

avg_prompt_ts
avg_pred_ts
avg_latency_ms

draft_n
accepted_n
accept_rate

raw_json
created_at
~~~

V1 server-benchmark states are:

~~~text
running
completed
benchmark_failed
timeout
parser_failed
interrupted
cancelled
~~~

The complete SPEED-Bench JSON remains authoritative in `raw_json`; normalized columns exist for frequent comparisons and UI queries.

### candidate_evaluation

~~~text
id
experiment_id
candidate_id
stage
decision
reason
metrics_json
created_at
~~~

Candidate evaluation rows are append-only workflow/provenance events. M9 records a `finalist / validate` event before launching the server and a terminal `server-validated / completed|failed` event afterward. They do not replace raw server-run or server-benchmark status.

## 30. SQLite indexes

At minimum:

~~~sql
CREATE INDEX idx_run_case
ON benchmark_run(benchmark_case_id);

CREATE INDEX idx_run_started
ON benchmark_run(started_at);

CREATE INDEX idx_sample_run
ON benchmark_sample(run_id);

CREATE INDEX idx_telemetry_run
ON telemetry_sample(run_id, timestamp_ns);

CREATE INDEX idx_case_candidate
ON benchmark_case(candidate_id);

CREATE INDEX idx_placement_candidate
ON resolved_placement(candidate_id);

CREATE UNIQUE INDEX idx_candidate_hash
ON candidate(config_hash);

CREATE UNIQUE INDEX idx_workload_hash
ON workload_case(workload_hash);
~~~

Additional indexes SHALL be added based on observed query patterns rather than speculative over-indexing.

## 31. Raw data retention

The database SHALL preserve:

- complete candidate JSON;
- complete workload JSON;
- complete measurement-policy JSON;
- argv arrays;
- environment snapshots;
- raw llama-bench JSON;
- raw SPEED-Bench JSON;
- stdout;
- stderr;
- individual samples;
- telemetry.

This allows future parsers and analysis code to reinterpret historical runs.

Large future artifacts such as profiles or traces SHOULD live outside SQLite and be referenced by:

- path;
- SHA-256;
- media/type;
- size;
- run ID.

## 32. Time and units

All timestamps SHALL be stored in UTC.

External timestamp form SHOULD be RFC3339 with subsecond precision where useful.

Durations SHOULD use integer nanoseconds when directly measured.

Storage units:

- bytes, not formatted GiB strings;
- tokens;
- nanoseconds;
- Hz;
- Celsius;
- watts when available.

Human-readable unit conversion belongs in presentation code.

## 33. Scheduling and ordering

The planner SHALL generate a deterministic candidate order for a deterministic search strategy.

The scheduler MAY support reordered execution for environmental-bias reduction.

If randomized ordering is used:

- the seed MUST be stored;
- generated order MUST be persisted.

A useful finalist-validation option is reverse-order rerunning to identify thermal/order bias.

## 34. Warmup and thermal policy

llama-bench warmup SHALL remain enabled by default.

MeasurementPolicy may additionally define:

- fixed delay between cases;
- fixed delay between candidates;
- future temperature-threshold cooldown.

V1 MAY initially implement fixed delay while preserving a policy model that can later support thermal stabilization.

## 35. Resumability

SQLite is the checkpoint.

On restart, the executor queries benchmark cases for which no successful completed run exists and resumes only those cases.

Before resuming, stale benchmark_run rows still marked running for that experiment are finalized as interrupted. Their case returns to a retryable planned state unless a successful run already exists.

A successfully completed case is not automatically rerun unless:

- the experiment policy explicitly requests another run;
- the user asks for a rerun;
- quality policy marks the previous observation insufficient.

Interrupted and failed attempts remain visible as append-only run history. An individual OOM, timeout, parser failure, or benchmark failure leaves the experiment resumable rather than converting the whole experiment into orchestration-level failed state.

## 36. Screening and validation stages

The system SHALL distinguish microbenchmark screening from server validation.

Typical workflow:

~~~text
Generate candidates
    ↓
Smoke test / fit
    ↓
Microbenchmark screening
    ↓
Full PP/TG depth measurements
    ↓
Pareto/frontier analysis
    ↓
Finalists
    ↓
llama-server + SPEED-Bench
    ↓
Candidate promotion
~~~

Server validation is append-only. It SHALL NOT rewrite the original experiment plan, microbenchmark observations, or Candidate identity.

A server-validation session SHALL reuse a concrete M6 ResolvedPlacement for the Candidate. The validation layer does not independently run automatic fitting because doing so would make server measurements incomparable with the placement that survived screening.

The host-level benchmark lock also protects server validation so a managed llama-server session cannot overlap another performance benchmark owned by llama-profile-lab.

The UI SHOULD expose statuses such as:

~~~text
planned
screened
finalist
server-validated
promoted
~~~

These are experiment/evaluation concepts, not replacements for raw run status.

### 36.1 Managed server lifecycle

For one finalist:

~~~text
verify exact binary hashes
        ↓
verify Candidate + host + resolved placement
        ↓
persist finalist evaluation
        ↓
persist server_run(starting)
        ↓
Popen llama-server in its own process group
        ↓
poll GET /health
        ↓
server_run(ready)
        ↓
run persisted SPEED-Bench workloads
        ↓
SIGTERM server process group
        ↓
SIGKILL after grace period if necessary
        ↓
persist server logs/status
        ↓
persist server-validated evaluation
~~~

The readiness endpoint is:

~~~text
GET http://HOST:PORT/health
~~~

HTTP 200 is ready. Connection failures and non-200 responses are treated as not-yet-ready until the readiness timeout. A process that exits before readiness is a `start_failed` server run.

Server stdout and stderr are captured independently from SPEED-Bench stdout and stderr. Startup, readiness, benchmark, parser, timeout, cancellation, and interruption failures remain persisted observations.

### 36.2 Exact executable identity

Both `llama-server` and SPEED-Bench are registered binaries with:

- path;
- SHA-256;
- size/mtime;
- help output;
- parsed option capabilities;
- build/version metadata where available.

Before validation, executable SHA-256 values are rechecked. A changed executable must be registered again rather than silently reusing stale provenance.

SPEED-Bench is recognized as binary kind `speed-bench`, including the upstream `speed_bench.py` filename.

### 36.3 Server argv

LlamaServerAdapter maps the Candidate plus concrete ResolvedPlacement to the selected server's advertised capability surface.

Long-form options are preferred. Relevant configuration includes:

- model path;
- host and port;
- production context;
- batch/ubatch;
- KV cache types;
- server parallelism;
- threads where explicit;
- flash attention;
- load/lazy settings;
- concrete GPU layers / split / device / tensor overrides;
- KV offload/unified behavior;
- op offload;
- repack;
- Candidate extra args;
- optional model alias;
- speculative settings.

Placement-fit arguments SHALL NOT be supplied during server validation.

Boolean defaults may be omitted only when the requested Candidate value equals the tool's known default and the selected binary does not advertise an explicit flag for that default. A non-default value that the binary cannot represent is an error.

## 37. Speculative decoding and SPEED-Bench

MTP and other speculative-decoding parameters are server-validation concerns unless a future benchmark tool directly supports them.

llama-bench SHALL be used for raw target-model PP/TG screening.

llama-server + SPEED-Bench SHALL be used to measure:

- actual delivered decode throughput;
- prompt throughput;
- request latency;
- draft acceptance;
- speculative speedup.

Candidate configuration still contains speculative settings so full deployment configurations remain reproducible.

### 37.1 SPEED-Bench invocation

The M9 adapter uses the selected binary's detected support for the current long-form surface:

~~~text
--url
--model
--bench
--category
--osl
--extra-inputs
--concurrency
--limit
--timeout
--output
~~~

Each persisted `speed-bench` WorkloadCase may produce one invocation per configured category. The subprocess writes raw JSON to a temporary output path; the database retains that parsed raw object plus subprocess stdout/stderr and argv.

### 37.2 Normalized SPEED-Bench metrics

From the invocation's `overall` summary row, V1 normalizes:

~~~text
requests
turns
failed
avg_prompt_t_s       → avg_prompt_ts
avg_pred_t_s         → avg_pred_ts
avg_latency seconds  → avg_latency_ms
draft_n
accepted             → accepted_n
accept_rate
~~~

Acceptance rate must be within 0–1 and accepted draft count cannot exceed drafted token count.

The persisted invocation category is the requested workload category such as `all`; normalized values come from that invocation's `overall` aggregate row.

### 37.3 Speculative Candidate validation

Candidate dimensions such as:

~~~text
speculative.enabled
speculative.type
speculative.draft_n_max
~~~

remain ordinary Candidate coordinates and may therefore be swept independently of workloads.

When the Candidate references a draft model, validation requires an explicit draft-model path. The server argv freezes the Candidate's speculative type and `draft_n_max` along with target/draft model provenance.

### 37.4 Baseline comparison

Server comparison aligns completed observations by persisted WorkloadCase and requested category.

For a non-speculative baseline and speculative Candidate it exposes, without selecting a winner:

~~~text
baseline/spec prompt throughput
baseline/spec delivered decode throughput
baseline/spec average latency
decode speedup = spec_pred_ts / baseline_pred_ts
latency speedup = baseline_latency_ms / spec_latency_ms
draft_n
accepted_n
accept_rate
~~~

Exactly one common workload/category is required for a scalar CLI comparison. Broader multi-workload visualization belongs to the analysis/API/UI layers.

## 38. Analysis model

Analysis consumes immutable observations and SHALL NOT mutate raw data.

V1 analysis should support:

- absolute metrics;
- relative-to-baseline metrics;
- matrix projections;
- N-dimensional filtering;
- 2D slices of higher-order spaces;
- Pareto frontier calculation;
- stability filtering;
- historical comparison by build;
- request-latency estimation.

### 38.1 Request latency model

For prompt P and generation G:

~~~text
prefill_time ≈ P / PP(P, initial_depth)

decode_time ≈ sum over generated tokens of:
    1 / TG(current_depth)

request_time =
    prefill_time + decode_time
~~~

Measured depth curves may be interpolated.

This allows analysis of PP/TG tradeoffs without inventing an arbitrary weighted score.

## 39. Pareto analysis

Raw candidates may be compared across multiple objectives.

Examples:

Maximize:

- PP tokens/s;
- TG tokens/s.

Minimize:

- CPU utilization;
- VRAM;
- RAM;
- power;
- latency.

The application SHOULD identify non-dominated candidates rather than presenting one universal “best” configuration.

A separate user-selected objective policy may choose among frontier candidates.

## 40. Frontend UX

### 40.1 Experiment list

Show:

- name;
- source profile/model;
- status;
- completed/total cases;
- created time;
- baseline;
- validation state.

### 40.2 New experiment

Primary form:

1. Select launcher profile.
2. Show resolved production settings.
3. Select parameters to vary.
4. Enter/select values.
5. Show constraints and valid candidate count.
6. Select workloads.
7. Select measurement policy.
8. Select placement mode.
9. Preview case/repetition count.
10. Save or run.

Advanced llama.cpp details stay collapsed by default.

### 40.3 Search-space editor

For each selected dimension display:

- friendly label;
- candidate values;
- source/current value;
- validation constraints;
- whether variation triggers refitting.

Example:

~~~text
Batch size
2048, 4096, 8192

UBatch size
512, 1024, 2048, 4096

11 valid candidates
1 invalid combination removed
~~~

### 40.4 Workload editor

Present workload concepts, not raw flags:

~~~text
Prefill
[x] 2K prompt at empty context
[x] 8K prompt at empty context

Decode
[x] 256 tokens at 4K context
[x] 256 tokens at 50% context
~~~

### 40.5 Placement choice

Expose:

~~~text
Re-fit each candidate
Measures deployable configurations.

Keep placement fixed
Isolates parameter effects.
~~~

Per-candidate fitting is the default optimization mode.

### 40.6 Experiment preview

Before execution show:

- base profile;
- dimensions;
- constraints;
- candidate count;
- workload count;
- benchmark case count;
- repetition count;
- placement policy;
- measurement policy.

### 40.7 Live execution page

Show:

- progress;
- current candidate;
- current workload;
- latest result;
- CPU process/system load;
- GPU load;
- VRAM;
- RAM;
- temperatures;
- power where available;
- completed/failed/pending counts.

Raw logs live under an Advanced section.

### 40.8 Matrix result view

For a 2D search:

~~~text
Metric: PP8192 tokens/s

             UBatch
Batch      512    1024    2048    4096
2048        ...     ...     ...      —
4096        ...     ...     ...     ...
8192        ...     ...     ...     ...
~~~

Metric selection may switch among:

- PP/TG throughput;
- CPU utilization;
- process CPU;
- GPU utilization;
- VRAM;
- RAM;
- temperature;
- power;
- estimated request latency.

### 40.9 Higher-dimensional exploration

Use:

- X dimension;
- Y dimension;
- metric;
- filters;
- facets.

Example:

~~~text
X axis: batch_size
Y axis: ubatch_size
Metric: PP8192

KV K: q8_0
KV V: q8_0
Fit target: 256

Facet by: cache_type_k
~~~

There is no requirement to draw literal 4D+ objects.

### 40.10 Candidate comparison

Allow selecting several candidates and comparing:

- configuration differences;
- PP workloads;
- TG depth curve;
- CPU;
- GPU;
- memory;
- stability;
- request-latency estimates;
- server validation.

### 40.11 Baseline comparison

Every experiment may designate a baseline, usually the source launcher profile.

Result views SHOULD support relative display:

~~~text
PP8K       +7.4%
TG65K      -1.8%
CPU        +14.0%
VRAM       +0.8 GiB
~~~

### 40.12 Promotion

A validated candidate may be promoted toward llama-profile-launcher.

V1 SHOULD first generate a patch/diff rather than silently changing launcher configuration.

Promotion records:

- experiment ID;
- candidate ID;
- source profile;
- old profile snapshot;
- proposed new profile snapshot;
- supporting validation results;
- timestamp.

## 41. CLI UX

Representative commands:

~~~text
llprof profile list
llprof profile show qwen-flash-gsq-rco-iq3-128k

llprof binary discover \
  --search-dir /path/to/llama.cpp/build/bin \
  --database data/benchmarks.db
llprof binary inspect /path/to/llama-server /path/to/speed_bench.py \
  --database data/benchmarks.db
llprof binary list --database data/benchmarks.db
llprof binary compare BIN_LEFT BIN_RIGHT --database data/benchmarks.db

llprof experiment create
llprof experiment plan EXPERIMENT --database data/benchmarks.db
llprof experiment run EXPERIMENT \
  --binary BENCH_BIN_ID \
  --fit-binary FIT_BIN_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db

llprof experiment resume EXPERIMENT \
  --binary BENCH_BIN_ID \
  --fit-binary FIT_BIN_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db

llprof placement list --database data/benchmarks.db
llprof placement show PLACEMENT_ID --database data/benchmarks.db

llprof run show RUN --database data/benchmarks.db

llprof results matrix EXPERIMENT
llprof results compare EXPERIMENT
llprof run show RUN

llprof server validate EXPERIMENT CANDIDATE \
  --server-binary SERVER_BIN_ID \
  --speed-bench-binary SPEED_BIN_ID \
  --model-path /path/to/target.gguf \
  --draft-model-path /path/to/draft.gguf \
  --placement PLACEMENT_ID \
  --model-name finalist

llprof server compare EXPERIMENT BASELINE_CANDIDATE SPEC_CANDIDATE \
  --category all

llprof ui
llprof archive
~~~

A direct non-interactive creation command SHOULD eventually support:

~~~text
llprof experiment create \
  --profile qwen-flash-gsq-rco-iq3-128k \
  --vary compute.batch_size=2048,4096,8192 \
  --vary compute.ubatch_size=512,1024,2048,4096 \
  --suite batch-ubatch-screen-v1
~~~

The CLI may expose IDs, but user-facing listing and lookup should accept friendly names where unambiguous.

## 42. Local HTTP API

M10 exposes a stable FastAPI DTO boundary over the same application/domain services used by the CLI. The frontend SHALL NOT read SQLite directly and HTTP handlers SHALL NOT duplicate planner, executor, analysis, or llama.cpp adapter logic.

### 42.1 Process and binding model

The local server is started with:

~~~text
llprof api \
  --host 127.0.0.1 \
  --port 8000 \
  --database data/benchmarks.db \
  --launcher-config /path/to/launcher/config/hosts/workstation.json
~~~

The default bind address is loopback-only:

~~~text
127.0.0.1
~~~

Remote binding requires an explicit `--host` override. M10 does not add remote-user authentication; exposing the API outside a trusted local environment is therefore outside the default security posture.

FastAPI documentation is available at:

~~~text
/api/docs
/api/openapi.json
~~~

Environment-based factory configuration is also supported:

~~~text
LLPROF_DATABASE
LLPROF_LAUNCHER_CONFIG
~~~

### 42.2 Stable DTO boundary

API responses use explicit Pydantic DTOs. They do not expose arbitrary SQLite rows.

DTOs cover:

- launcher profiles;
- exact binary identities/capabilities;
- model/model-file registry entries;
- experiments and definitions;
- planning summaries;
- execution operations/progress;
- Candidates and completion counts;
- benchmark runs, samples, metrics, and logs;
- telemetry samples;
- analysis/export rows;
- matrix projections;
- server-validation results.

Domain objects such as Candidate, SearchSpace, WorkloadSuite, MeasurementPolicy, and ExperimentDefinition remain the typed semantic source for nested experiment creation.

### 42.3 Profiles

M10 does not introduce launcher promotion or a second persisted profile registry.

When `--launcher-config` is supplied, these endpoints read the existing llama-profile-launcher host JSON directly:

~~~text
GET /api/profiles
GET /api/profiles/{id}
~~~

Effective arguments are resolved in deterministic precedence order:

~~~text
defaults.args
    ↓
listed profile args, in order
    ↓
model args
~~~

The response also includes the selected binary key/path, target model path, optional draft-model path, profile chain, and server alias.

The provider is read-only. Profile mutation and promotion remain M12 work.

If no launcher config is supplied, the profile list reports `configured=false` with no entries rather than inventing profile state.

### 42.4 Binary and model resources

Implemented resources:

~~~text
GET  /api/binaries
POST /api/binaries/inspect
GET  /api/models
~~~

Binary inspection reuses the exact M4 probing path: executable SHA-256, path, size/mtime, build metadata, help-derived capabilities, and tool kind are persisted before being returned.

The model endpoint is read-only in M10 and exposes existing `model` / `model_file` registry records. Model-registration workflow remains separate from this milestone.

### 42.5 Experiment creation and inspection

Implemented resources:

~~~text
POST /api/experiments
GET  /api/experiments
GET  /api/experiments/{id}
POST /api/experiments/{id}/clone
~~~

Experiment creation accepts one typed request containing:

~~~text
name
base_candidate
search_space
workload_suite
measurement_policy
placement_policy
baseline
~~~

The service persists content-addressed Candidate/SearchSpace/WorkloadSuite/MeasurementPolicy records and then creates the draft Experiment in one SQLite transaction.

A fixed placement or explicitly selected Candidate baseline must already exist.

Clone creates a new draft ExperimentDefinition referring to the same immutable component identities. It does not copy run history.

### 42.6 Planning and benchmark execution

Implemented resources:

~~~text
POST /api/experiments/{id}/plan
POST /api/experiments/{id}/run
POST /api/experiments/{id}/pause
POST /api/experiments/{id}/resume
POST /api/experiments/{id}/cancel

GET  /api/experiments/{id}/progress
~~~

Planning calls the existing pure `plan_experiment` application function.

Run/resume requests provide:

~~~text
binary_id
model_path
fit_binary_id
timeout_seconds
fit_timeout_seconds
limit
telemetry_interval_ms
~~~

The API operation manager starts the existing `ExperimentExecutor` in an in-process worker thread and returns HTTP 202 immediately. It does not implement a second benchmark scheduler.

Only one API-owned benchmark operation may be active on the host at a time. The executor's existing host lock remains the final cross-process exclusion mechanism.

Pause and cancel are cooperative:

- pause sets the executor cancellation event; the current subprocess is stopped through the existing ProcessRunner behavior and the Experiment ends in `paused`;
- cancel uses the same cooperative stop, then records the Experiment as `cancelled`;
- the executor checks the event before starting the next planned case so no additional case begins after a pause/cancel request.

The operation manager's thread/status object is intentionally ephemeral presentation/control state. Durable experiment state, run attempts, failures, samples, placement, and telemetry remain in SQLite.

If the API process disappears during execution, normal resume/orphan recovery semantics apply on the next run.

### 42.7 Progress and Server-Sent Events

Polling:

~~~text
GET /api/experiments/{id}/progress
~~~

returns:

- Experiment status;
- total/completed/incomplete case counts;
- current case-status counts;
- current or most recent API operation snapshot when available.

Live progress uses SSE:

~~~text
GET /api/experiments/{id}/events
Content-Type: text/event-stream
~~~

Each changed snapshot is emitted as:

~~~text
event: progress
data: <ExperimentProgressDTO JSON>
~~~

Unchanged snapshots are suppressed. The stream ends when the API operation is absent or terminal:

~~~text
completed
paused
cancelled
failed
~~~

SSE is preferred over WebSocket in V1 because progress delivery is one-way.

### 42.8 Candidate, run, result, and telemetry resources

Implemented resources:

~~~text
GET /api/experiments/{id}/candidates
GET /api/experiments/{id}/runs
GET /api/experiments/{id}/results
GET /api/experiments/{id}/matrix

GET /api/runs/{id}
GET /api/runs/{id}/telemetry
~~~

Candidate responses combine the immutable Candidate with generation metadata and counts for workloads, microbenchmark cases, completed cases, and completed server validations.

Run detail returns normalized provenance plus individual benchmark samples, normalized metrics, stdout, and stderr. Telemetry remains a separate endpoint because its sample volume can be substantially larger.

The result endpoint delegates to M8 `AnalysisService.export_rows`. It supports repeated query parameters:

~~~text
filter=PATH=VALUE
quality=clean
metric=throughput.median
~~~

Filter values use the same JSON-scalar parsing semantics as the CLI.

The matrix endpoint delegates to the existing sparse projection service:

~~~text
x=compute.batch_size
y=compute.ubatch_size
metric=throughput.median
facet=context.cache_type_k
filter=workload.kind=microbench-prefill
~~~

The API does not reinterpret or silently aggregate coordinates differently from CLI analysis.

### 42.9 Server validation

Implemented resource:

~~~text
POST /api/candidates/{id}/validate
~~~

The request supplies the Experiment context, registered server/SPEED-Bench binary IDs, target/draft model paths, optional concrete placement, bind host/port, timeouts, and optional workload selection.

The route delegates to M9 `ServerValidationService`; the HTTP layer does not construct llama-server or SPEED-Bench argv itself.

Candidate promotion is intentionally not implemented in M10:

~~~text
POST /api/candidates/{id}/promote
~~~

remains M12 work because promotion requires launcher provenance and patch/diff semantics.

### 42.10 HTTP status semantics

V1 uses:

~~~text
200 successful reads/actions
201 created resource / registered binary
202 accepted asynchronous benchmark-control action

400 malformed analysis/binary request
404 missing resource
409 invalid state transition / host-operation conflict
422 Pydantic request validation failure
503 configured launcher profile source unavailable/malformed
~~~

Subprocess benchmark failures themselves remain persisted experiment/run data rather than being rewritten as generic HTTP failures.

### 42.11 Browser client (M11)

The V1 browser client is implemented with React, TypeScript, and Vite. It consumes only the local HTTP/SSE API and SHALL NOT read SQLite directly.

The production-style local command is:

~~~text
llprof ui \
  --host 127.0.0.1 \
  --port 8000 \
  --launcher-config /path/to/launcher/config/hosts/workstation.json \
  --database data/benchmarks.db
~~~

The built frontend is mounted after the /api routes so the browser and API share one local origin. Development uses the Vite dev server on 127.0.0.1:5173 with /api proxied to the FastAPI process on 127.0.0.1:8000.

Frontend dependencies SHALL be reproduced from frontend/package-lock.json using npm ci. Generated frontend/dist artifacts SHALL NOT be committed.

#### 42.11.1 Route model

V1 uses hash routing so static hosting does not require server-side SPA rewrites:

~~~text
#/
#/new
#/experiments/{experiment_id}
#/experiments/{experiment_id}/compare
#/experiments/{experiment_id}/candidates/{candidate_id}
~~~

#### 42.11.2 Experiment editor

The editor SHALL begin from the read-only launcher profile resource and construct an immutable Candidate snapshot.

Search dimensions SHALL be populated from backend ParameterDefinition metadata rather than a second hard-coded parameter catalog. The UI SHALL expose parameter label/category/path, comma-separated discrete values, backend string choices and numeric constraints where present, placement-affecting status, and exact binary capability support.

When a registered llama-bench binary is selected, a parameter whose required CLI option is absent SHALL be visibly disabled.

The reference editor defaults SHALL reproduce:

~~~text
compute.batch_size = {2048,4096,8192}
compute.ubatch_size = {512,1024,2048,4096}
constraint: compute.ubatch_size <= compute.batch_size
~~~

which previews 12 raw combinations, 11 valid Candidates, and one constrained rejection.

The default workload suite SHALL provide:

~~~text
PP2K d0
PP8K d0
TG256 d4096
TG256 d50%-available-context
~~~

which yields 44 microbenchmark cases for the 11 valid Candidates.

SPEED-Bench workload definitions MAY be added in the same editor. They SHALL remain server-only workloads and SHALL NOT be included in the llama-bench case count.

#### 42.11.3 Live execution

The execution screen SHALL use the existing run/resume/pause/cancel endpoints. SSE is the primary live-update channel, with polling used as a resilience/fallback mechanism.

The live screen SHALL make visible, when available, completed/total/incomplete cases, current Candidate and WorkloadCase, latest tokens/s, process and system CPU, GPU utilization, RAM/process RSS/VRAM, CPU/GPU temperature, GPU power, recent run status/quality, and persisted failure counts.

No browser-only execution state is authoritative.

#### 42.11.4 Result exploration

For a two-dimensional view, the browser SHALL request the existing sparse matrix projection and render table/heatmap semantics from the returned coordinates.

For spaces with more than two dimensions, each hidden dimension SHALL be either fixed by an exact filter or selected as an explicit facet. The UI SHALL NOT silently average hidden coordinates.

Pareto visualization SHALL use the existing caller-defined M8 objectives and display only the non-dominated observed Candidate set. It SHALL NOT synthesize an overall benchmark score.

#### 42.11.5 Candidate detail and comparison

Candidate detail SHALL include immutable configuration, resolved production placement, PP/TG observations, measured decode throughput versus active context depth, CPU/GPU/memory/stability metrics, signed baseline deltas, compute-only request-latency estimation, and M9 server-validation history/controls.

Advanced hashes/raw JSON SHALL be collapsed by default.

Comparison SHALL support 2–5 selected Candidates. Values and signed deltas SHALL remain workload- and metric-specific. The UI SHALL NOT rank Candidates or declare a universal winner.

#### 42.11.6 Frontend quality gates

CI SHALL run:

~~~text
npm ci
npm run typecheck
npm run test
npm run build
~~~

in addition to the Python lockfile, Ruff, strict mypy, and pytest gates.

## 43. Source launcher integration

llama-profile-launcher is treated as an external source of production profiles.

The LauncherProfileAdapter SHALL:

- load the configured launcher JSON;
- resolve defaults;
- resolve profile inheritance;
- resolve model-specific args;
- preserve the source snapshot;
- map performance-relevant settings into Candidate fields;
- retain non-benchmark profile data for provenance;
- report unsupported/unknown fields.

The benchmark database MUST not depend on launcher files remaining unchanged after import.

## 44. Reproducibility metadata

Each run SHOULD be reconstructable from stored data.

At minimum retain:

- source experiment;
- Candidate canonical JSON;
- WorkloadCase canonical JSON;
- MeasurementPolicy;
- ResolvedPlacement;
- model file hashes;
- binary hashes/build information;
- host fingerprint;
- argv;
- relevant environment variables;
- timestamps;
- telemetry;
- raw output.

A path change alone should not change model or candidate identity.

## 45. Environmental variables and secrets

Environment variables passed to subprocesses SHALL be captured selectively.

Known secrets such as Hugging Face tokens MUST NOT be persisted.

Environment capture must use an allowlist or redaction policy.

## 46. Security model

V1 is local-first.

Defaults:

- API binds only to 127.0.0.1;
- no remote network exposure;
- no shell=True;
- no arbitrary eval;
- subprocess paths are explicit;
- experiment expressions are parsed by a restricted evaluator;
- destructive launcher-profile updates require explicit user action.

If remote access is added later, authentication/authorization becomes a separate explicit mode.

## 47. Logging

Use Python logging with structured context.

Useful fields:

- experiment ID;
- candidate ID;
- workload ID;
- benchmark case ID;
- run ID;
- process ID.

Console logs are diagnostic.

SQLite remains the authoritative experiment record.

## 48. Artifacts and database size

Small raw results and logs may live directly in SQLite.

Large future artifacts should live in an artifact directory and have a database record containing:

- run ID;
- path;
- SHA-256;
- size;
- media/type;
- creation time.

An archive operation SHALL checkpoint WAL and produce a consistent database snapshot.

## 49. Testing strategy

### 49.1 Unit tests

Cover:

- canonical hashing;
- Candidate validation;
- SearchSpace expansion;
- constraints;
- conditional dimensions;
- workload depth expansion;
- parameter registry;
- llama.cpp argv generation;
- output parsers;
- Pareto calculations;
- latency calculations;
- state transitions.

### 49.2 Golden parser fixtures

Store representative llama.cpp outputs as fixtures for:

- llama-bench JSON;
- fit-params output;
- server startup output;
- SPEED-Bench JSON;
- failures/OOM.

Parsers SHALL be testable without invoking llama.cpp.

### 49.3 Integration tests

Use fake executables/scripts to test:

- process management;
- timeout;
- interrupt;
- stdout/stderr capture;
- status persistence;
- resume behavior;
- telemetry lifecycle.

Real llama.cpp integration tests may be optional and host-dependent.

### 49.4 Migration tests

A migration test SHALL create an old schema, migrate forward, and verify data remains readable.

## 50. Initial implementation sequence

Implementation should proceed in this order:

### Phase 1 — Core domain

- Pydantic schemas
- canonical serialization/hashing
- ParameterRegistry
- Candidate
- SearchSpace
- WorkloadSuite
- WorkloadCase
- MeasurementPolicy

### Phase 2 — Persistence

- SQLite connection configuration
- migration framework
- initial schema
- repositories
- immutable object persistence
- experiment planning persistence

### Phase 3 — Planning

- launcher-profile adapter
- grid expansion
- constraint evaluation
- workload expansion
- benchmark-case generation
- dry-run/plan output

At this point a batch/ubatch experiment can be completely planned without launching anything.

### Phase 4 — llama-bench execution

- binary discovery/capability detection
- LlamaBenchAdapter
- ProcessRunner
- JSON parser
- benchmark samples
- run lifecycle
- resume

At this point the system is a useful microbenchmark runner.

### Phase 5 — Placement

- llama-fit-params adapter
- placement hashing/cache
- per-candidate placement resolution
- fixed-placement mode

### Phase 6 — Telemetry

- Linux process/system CPU
- RAM
- GPU provider
- summary metrics
- run-quality classification

### Phase 7 — Analysis

- matrix projections
- baseline deltas
- Pareto frontier
- request-latency model
- CSV/JSON exports

### Phase 8 — Server validation

- llama-server lifecycle
- readiness detection
- SPEED-Bench adapter
- MTP results
- acceptance rate

### Phase 9 — Local UI

- FastAPI endpoints
- React experiment editor
- plan preview
- execution progress
- matrix/heatmap
- higher-dimensional slicing
- candidate comparison
- promotion diff

## 51. V1 acceptance criteria

V1 is complete when the following end-to-end scenario works without manual database editing:

1. User selects the existing Flash Next 128K launcher profile.
2. User chooses to vary batch_size over 2048/4096/8192.
3. User chooses to vary ubatch_size over 512/1024/2048/4096.
4. The planner removes combinations where ubatch > batch.
5. The UI reports eleven valid candidates.
6. User selects PP2K, PP8K, TG256@4K, and TG256@50%.
7. The UI reports 44 benchmark cases.
8. Each candidate is fit against the 128K production context when per-candidate fitting is selected.
9. The concrete placement is frozen for that candidate's llama-bench workloads.
10. llama-bench uses long-form arguments wherever available.
11. Three or more repetitions are persisted individually.
12. CPU system load and benchmark-process CPU load are captured.
13. GPU telemetry is captured when available.
14. Failed/OOM cases remain queryable.
15. The run can be interrupted and resumed without losing completed work.
16. Results can be displayed as a batch × ubatch matrix for any selected metric.
17. Results can be filtered by workload.
18. Candidate comparison shows PP, TG, CPU, GPU, memory, and stability.
19. A Pareto view identifies non-dominated candidates without forcing a universal winner.
20. Selected finalists can be validated via llama-server/SPEED-Bench.
21. A validated candidate can generate a proposed llama-profile-launcher configuration diff.
22. Six months later, the database contains enough information to understand exactly what was run, with which model/build/hardware/configuration, and to recompute analysis using new objectives.

## 52. Design invariant summary

The implementation should protect these invariants:

1. Candidate configuration is immutable.
2. Workload identity is independent of measurement repetitions.
3. Experiment definitions freeze when a complete plan is persisted.
4. Completed runs are append-only.
5. Placement requested by a Candidate is distinct from placement resolved on a host.
6. Production context determines fitting; active depth determines workload.
7. N-dimensional search spaces are stored as observed points, not dense tensors.
8. Raw observations are preserved independently of optimization policy.
9. SQLite is the canonical scientific record.
10. CLI and web UI share one domain/service layer.
11. llama.cpp argument spelling is isolated behind adapters.
12. Long-form llama.cpp arguments are preferred wherever supported.
13. CPU is a first-class resource metric alongside GPU.
14. Failures are data.
15. Every result should be explainable and reproducible from stored provenance.
