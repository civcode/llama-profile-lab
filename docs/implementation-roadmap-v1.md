# llama-profile-lab — V1 Implementation Roadmap

Status: Implementation roadmap  
Repository: civcode/llama-profile-lab  
Companion specification: docs/technical-spec-v1.md

## 1. Purpose

This roadmap converts the V1 technical specification into an implementation sequence with clear milestones, dependencies, acceptance gates, and suggested work items.

The roadmap is intentionally structured around vertical capability milestones rather than individual files. Each milestone should leave the repository in a usable, testable state.

The preferred development principle is:

> Build the smallest trustworthy experiment engine first, then add placement, telemetry, analysis, server validation, and UI on top of the same stable domain and persistence model.

## 2. Guiding implementation rules

The following rules apply throughout V1 implementation:

1. **SQLite is the source of truth.**
2. **Experiments are planned before they execute.**
3. **Candidates and workload cases are immutable.**
4. **Completed runs are append-only.**
5. **Failures are persisted as observations.**
6. **Raw tool output is retained alongside parsed fields.**
7. **Candidate search and workload expansion remain independent.**
8. **llama.cpp CLI details stay behind adapters.**
9. **Long-form llama.cpp arguments are used wherever supported.**
10. **Production context determines fitting; active context depth determines benchmark workload.**
11. **CPU and GPU telemetry are first-class measurements.**
12. **The UI and CLI use the same backend/domain services.**
13. **Optimization policy never overwrites raw observations.**
14. **Every milestone must include tests before the next milestone begins.**
15. **Development and CI dependency resolution use the committed uv.lock in frozen mode.**

## 3. Milestone overview

~~~text
M0  Repository bootstrap
 ↓
M1  Domain model + canonical hashing
 ↓
M2  SQLite persistence + migrations
 ↓
M3  Experiment planning + N-D grid expansion
 ↓
M4  llama.cpp binary discovery + capability detection
 ↓
M5  llama-bench execution engine
 ↓
M6  Placement resolution with llama-fit-params
 ↓
M7  CPU/GPU telemetry + run quality
 ↓
M8  Analysis layer + matrix/tensor projections
 ↓
M9  llama-server + SPEED-Bench validation
 ↓
M10 Local HTTP API
 ↓
M11 React UI
 ↓
M12 Launcher promotion + archive/export
 ↓
M13 V1 hardening and release
~~~

A milestone is complete only when its acceptance gate passes.

## 4. M0 — Repository bootstrap\n\n**Status: Complete**

### Objective

Create the project skeleton, development environment, test runner, linting, and CI foundation.

### Deliverables

- pyproject.toml
- committed uv.lock
- .python-version pinned to Python 3.12
- Python package under src/llama_profile_lab
- test structure
- migration directory
- frontend directory placeholder
- CLI entry point placeholder
- README with development setup
- basic CI workflow
- formatting/linting configuration

### Recommended dependencies

Backend:

- uv
- Python 3.12+
- Pydantic v2
- FastAPI
- Uvicorn
- pytest

Optional development tooling:

- Ruff
- mypy or pyright
- pytest-cov

Frontend is not required to build yet.

### Suggested structure

~~~text
src/llama_profile_lab/
  domain/
  planning/
  execution/
  llama/
  db/
  analysis/
  api/
  cli/
~~~

### Work items

- [x] Initialize pyproject.toml
- [x] Configure uv project workflow
- [x] Add .python-version
- [x] Generate and commit uv.lock
- [x] Move development tools to the dev dependency group
- [x] Add package metadata
- [x] Add llprof console entry point
- [x] Add pytest configuration
- [x] Add lint/type-check configuration
- [x] Add .gitignore
- [x] Add initial README
- [x] Add CI for lint + unit tests
- [x] Add empty migration framework directory

### Acceptance gate

The following must succeed from a clean checkout:

~~~bash
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest
uv run --frozen llprof --help
~~~

CI SHALL use the same frozen lockfile rather than independently resolving dependency ranges.

No llama.cpp installation is required yet.

---

## 5. M1 — Domain model and canonical identities\n\n**Status: Complete**

### Objective

Implement the immutable typed objects that define candidates, search spaces, workloads, measurement policies, and experiments.

This milestone establishes the vocabulary used by the entire project.

### Deliverables

Pydantic/domain models for:

- Candidate
- SearchSpace
- SearchDimension
- SearchConstraint
- WorkloadSuite
- WorkloadCase
- MeasurementPolicy
- ExperimentDefinition
- PlacementPolicy
- model references
- binary references where needed
- canonical JSON serialization
- SHA-256 content identity

### Candidate scope

Implement V1 typed fields for:

- model
- context
- KV cache types
- KV offload/unified settings
- flash attention
- batch size
- ubatch size
- threads
- load mode
- lazy mode
- repack
- host/op offload
- fit policy
- GPU placement constraints
- server parallelism
- speculative decoding
- extra_args

### Search-space scope

Support:

- arbitrary N-dimensional discrete dimensions
- path-based parameter references
- deterministic grid expansion input
- conditional dimensions
- constraint definitions

### Workload scope

Support:

- microbench-prefill
- microbench-decode
- microbench-combined
- speed-bench
- absolute depth
- fractional depth

### Work items

- [x] Candidate schema
- [x] Candidate validators
- [x] WorkloadSuite schema
- [x] WorkloadCase schema
- [x] MeasurementPolicy schema
- [x] SearchSpace schema
- [x] ExperimentDefinition schema
- [x] canonical serializer
- [x] SHA-256 identity functions
- [x] semantic metadata exclusion from hashes
- [x] unit tests for stable hashes
- [x] unit tests for validation failures

### Important tests

- candidate hash does not change when display metadata changes
- candidate hash changes when batch size changes
- workload hash is independent of repetition count
- ubatch_size greater than batch_size is rejected
- speculative draft count is rejected when speculative mode is disabled
- canonical JSON is deterministic

### Acceptance gate

Given two independently created equivalent Candidate objects, they produce byte-identical canonical JSON and identical hashes.

No subprocess execution exists yet.

---

## 6. M2 — SQLite persistence and migrations\n\n**Status: Complete**

### Objective

Create the durable experiment record before implementing execution.

### Deliverables

- SQLite connection layer
- WAL configuration
- numbered migrations
- schema-version tracking
- repository layer
- persistence for immutable domain objects
- experiment state storage
- benchmark planning tables
- transaction utilities

### Initial tables

At minimum:

- schema_migration
- host
- binary
- model
- model_file
- candidate
- workload_suite
- workload_case
- measurement_policy
- experiment
- experiment_candidate
- experiment_workload
- resolved_placement
- benchmark_case
- benchmark_run
- benchmark_sample
- telemetry_sample
- server_run
- server_benchmark
- candidate_evaluation

### Persistence rules

- immutable entities use content hashes and unique constraints
- repeated insertion of the same immutable entity returns the existing row
- runs are never deduplicated
- all foreign keys are enforced
- raw JSON is preserved

### Work items

- [x] connection factory
- [x] SQLite PRAGMA initialization
- [x] migration runner
- [x] 001_initial.sql
- [x] repository interfaces
- [x] candidate repository
- [x] workload repository
- [x] measurement-policy repository
- [x] experiment repository
- [x] benchmark-case repository
- [x] run repository
- [x] transaction tests
- [x] migration tests

### Acceptance gate

A fresh database can be created, populated with a Candidate + WorkloadCase + Experiment, closed, reopened, and read back without semantic differences.

Migration tests run in CI.

---

## 7. M3 — Planning engine and N-dimensional expansion

**Status: Complete**

### Objective

Turn experiment definitions into a complete deterministic benchmark plan without running llama.cpp.

This is the first milestone where the user can define a real batch/ubatch experiment.

### Deliverables

- parameter registry
- path resolution
- grid expansion
- constraint engine
- conditional dimensions
- workload expansion
- relative-depth expansion
- benchmark-case generation
- plan persistence
- plan summary

### Parameter registry

Implement typed metadata for at least:

- compute.batch_size
- compute.ubatch_size
- compute.flash_attn
- context.cache_type_k
- context.cache_type_v
- context.size
- placement.fit.target_mib
- placement.constraints.n_gpu_layers
- speculative.draft_n_max

Registry metadata should include:

- type
- label
- category
- CLI argument where applicable
- minimum/maximum where applicable
- allowed values
- whether it affects placement
- applicable tools
- conditions

### Constraint engine

V1 constraint support:

- ==
- !=
- <
- <=
- >
- >=
- membership
- boolean AND/OR
- candidate-path references

No arbitrary eval.

### Relative depth

Implement:

~~~text
available_depth =
    candidate.context.size
    - prompt_tokens
    - generate_tokens
    - safety_margin_tokens

depth_tokens =
    floor(available_depth × fraction)
~~~

### First reference experiment

Use the Flash Next batch/ubatch sweep:

~~~text
batch_size:
2048, 4096, 8192

ubatch_size:
512, 1024, 2048, 4096

constraint:
ubatch <= batch
~~~

Expected result:

~~~text
11 candidates
~~~

Reference workloads:

~~~text
PP2048 @ d0
PP8192 @ d0
TG256  @ d4096
TG256  @ d50%
~~~

Expected result:

~~~text
44 benchmark cases
~~~

### Work items

- [x] ParameterRegistry
- [x] typed parameter-path access
- [x] dimension expansion
- [x] conditional dimensions
- [x] safe constraint parser/evaluator
- [x] Candidate mutation from base Candidate
- [x] candidate deduplication
- [x] WorkloadSuite expansion
- [x] relative-depth handling
- [x] benchmark-case generation
- [x] deterministic ordering
- [x] persisted plan
- [x] llprof experiment plan
- [x] human-readable plan summary

### Acceptance gate

From a stored base Candidate and SearchSpace, llprof experiment plan creates exactly the expected eleven batch/ubatch candidates and forty-four benchmark cases, deterministically, without launching any external process.

---

## 8. M4 — llama.cpp discovery and capability detection

**Status: Complete**

### Objective

Make llama-profile-lab aware of the actual llama.cpp binaries present on the machine, including custom branches.

### Deliverables

Binary discovery/configuration for:

- llama-bench
- llama-fit-params
- llama-server

Capture:

- file SHA-256
- path
- size
- mtime
- build/version output
- git/build metadata where discoverable
- help output
- parsed supported options

### Capability model

The selected binary determines whether a parameter is available.

Examples:

- a custom Qwen build may support capabilities absent upstream
- server speculative flags do not imply llama-bench support
- unsupported UI controls are disabled rather than silently ignored

### Work items

- [x] binary registration
- [x] executable hashing
- [x] help/version invocation
- [x] capability parser
- [x] capability persistence
- [x] binary comparison logic
- [x] CLI command to inspect binaries
- [x] tests using captured help fixtures

### Acceptance gate

The application can register both the native llama.cpp build and the custom Qwen build, identify them separately by executable hash, and report their supported arguments.

---

## 9. M5 — llama-bench execution engine

**Status: Implementation Complete — Real-host acceptance pending**

### Objective

Execute planned microbenchmarks safely and persist complete results.

### Deliverables

- ProcessRunner
- LlamaBenchAdapter
- argv generation
- result JSON parsing
- run lifecycle
- individual sample persistence
- stdout/stderr persistence
- failure classification
- resume support
- host lock

### ProcessRunner requirements

- shell=False
- argv arrays
- stdout capture
- stderr capture
- process group handling
- timeout
- graceful termination
- forced termination
- cancellation
- timestamps
- exit code

### llama-bench arguments

Prefer long-form arguments:

- --model
- --n-prompt
- --n-gen
- --n-depth
- --batch-size
- --ubatch-size
- --cache-type-k
- --cache-type-v
- --n-gpu-layers
- --flash-attn
- --repetitions
- --output
- etc.

Short-only arguments may be special-cased only when required.

### Run states

Implement:

- planned
- running
- completed
- oom
- timeout
- invalid
- benchmark_failed
- parser_failed
- interrupted
- cancelled

### Work items

- [x] ProcessRunner
- [x] host lock
- [x] LlamaBenchAdapter
- [x] candidate/workload to argv mapping
- [x] JSON parser
- [x] raw JSON preservation
- [x] individual sample insertion
- [x] aggregate metric extraction
- [x] failure classification
- [x] orphaned-run recovery
- [x] llprof experiment run
- [x] llprof experiment resume
- [x] llprof run show

### Acceptance gate

Automated acceptance is complete: the batch/ubatch reference plan executes through the real ProcessRunner against a llama-bench-compatible test executable, deliberately pauses after five cases, resumes the remaining thirty-nine cases, and verifies exactly forty-four successful runs and 132 individually persisted timed samples with no duplicate reruns.

Real-host acceptance remains pending: run the same flow against a registered native/custom llama-bench executable and a real GGUF on the benchmark workstation, without placement fitting. That validation cannot be performed from GitHub CI because the workstation binaries and models are local.

---

## 10. M6 — Placement resolution

### Objective

Make benchmark placement representative of the actual production context rather than allowing shallow benchmark cases to refit independently.

### Deliverables

- LlamaFitParamsAdapter
- placement-cache key
- per-candidate fit
- fixed-placement mode
- resolved placement persistence
- concrete placement applied to llama-bench

### Placement cache identity

At minimum include:

- model identity
- production context
- KV types
- flash attention
- batch size
- ubatch size
- fit target
- load/lazy settings where relevant
- relevant placement constraints
- host fingerprint
- fit binary identity

### Required behavior

For per-candidate placement:

~~~text
Candidate
  ↓
llama-fit-params at production context
  ↓
ResolvedPlacement
  ↓
all microbench workloads use that placement
~~~

For fixed placement:

~~~text
ResolvedPlacement X
  ↓
all selected Candidates use X
~~~

### Work items

- [ ] fit argv generation
- [ ] fit output parser
- [ ] resolved-placement model
- [ ] placement hash
- [ ] placement caching
- [ ] per-candidate fit scheduling
- [ ] fixed-placement experiment mode
- [ ] placement failure handling
- [ ] concrete placement injection into bench argv

### Acceptance gate

A 128K Flash Next Candidate is fit once against 131072 production context, and all its shallow and deep llama-bench cases use the same resolved placement.

---

## 11. M7 — CPU/GPU telemetry and run-quality classification

### Objective

Capture enough runtime state to distinguish real parameter effects from CPU/GPU contention, throttling, or memory pressure.

### Deliverables

- TelemetryProvider abstraction
- Linux CPU/process telemetry
- RAM telemetry
- GPU telemetry provider
- periodic sampler
- before/after snapshots
- run summary
- run-quality classification

### CPU measurements

Capture where available:

- total system CPU %
- user/system/iowait %
- benchmark-process CPU %
- process CPU raw per-core %
- process user/system CPU time
- process thread count
- per-core utilization
- CPU frequency
- temperature
- load average
- package power if reliably available

### Memory measurements

- RAM used
- RAM available
- swap
- process RSS

### GPU measurements

For each relevant GPU:

- utilization
- VRAM used/total
- temperature
- power
- clocks
- throttling indicators when available

### Sampling

Default:

~~~text
1000 ms
~~~

Optional:

~~~text
500 ms
~~~

### Run quality

Implement initial classifications:

- clean
- noisy
- external_cpu_load
- external_gpu_load
- thermal_throttle
- telemetry_incomplete

### Work items

- [ ] /proc CPU/process provider
- [ ] /sys frequency/temperature provider
- [ ] memory provider
- [ ] NVIDIA or generic GPU provider
- [ ] sampler lifecycle
- [ ] benchmark PID association
- [ ] telemetry persistence
- [ ] run summary metrics
- [ ] quality rules
- [ ] UI/API-ready telemetry DTO

### Acceptance gate

A Flash Next decode run records both process CPU load and system CPU load, and a deliberately CPU-loaded test machine produces a noisy/external-load classification without discarding the run.

---

## 12. M8 — Analysis and multidimensional projections

### Objective

Turn the stored observation set into useful comparisons without changing raw measurements.

### Deliverables

- metric layer
- result projections
- matrix generation
- filtering
- baseline deltas
- stability metrics
- Pareto frontier
- request-latency estimation
- export

### Required metrics

At minimum:

Performance:

- mean t/s
- median t/s
- standard deviation
- coefficient of variation
- min/max

Resources:

- average/peak process CPU
- average/peak system CPU
- peak VRAM
- average GPU utilization
- peak RAM/process RSS
- temperature summaries
- power where available

### Matrix projection

Given:

~~~text
X = compute.batch_size
Y = compute.ubatch_size
metric = PP8192 median t/s
filters = remaining dimensions/workload/build/etc.
~~~

produce a matrix suitable for CLI and UI.

### Higher dimensions

Support:

- arbitrary filters
- X/Y selection
- facet dimension
- sparse missing cells

### Baseline deltas

Example:

~~~text
PP8K    +7.4%
TG65K   -1.8%
CPU     +14%
~~~

### Pareto analysis

Support user-selected maximize/minimize objectives.

Do not force a universal winner.

### Request-latency analysis

Use PP and TG curves to estimate representative workloads.

### Work items

- [ ] metric registry
- [ ] statistical summaries
- [ ] matrix projector
- [ ] filter engine
- [ ] baseline comparison
- [ ] Pareto frontier
- [ ] latency interpolation/model
- [ ] CSV export
- [ ] JSON export
- [ ] llprof results matrix
- [ ] llprof results compare

### Acceptance gate

The batch/ubatch reference experiment can be viewed as:

- PP2K matrix
- PP8K matrix
- TG4K matrix
- TG50% matrix
- process-CPU matrix

and candidates can be filtered to the Pareto frontier using PP, TG, CPU, and memory objectives.

---

## 13. M9 — llama-server and SPEED-Bench validation

### Objective

Validate finalists under actual server execution, including MTP/speculative decoding.

### Deliverables

- LlamaServerAdapter
- server lifecycle management
- readiness detection
- SpeedBenchAdapter
- server-run persistence
- speculative metrics
- finalist validation workflow

### Server lifecycle

Support:

- start
- readiness wait
- log capture
- benchmark
- clean stop
- forced stop
- failure handling

### SPEED-Bench metrics

Persist:

- prompt throughput
- predicted/decode throughput
- latency
- draft count
- accepted draft count
- acceptance rate
- raw result JSON

### Speculative validation

Support candidate variation such as:

~~~text
draft_n_max = 1, 2, 3, 4
~~~

where supported by the selected server binary.

### Work items

- [ ] server argv generator
- [ ] server process lifecycle
- [ ] health/readiness detection
- [ ] SPEED-Bench runner
- [ ] raw/normalized parser
- [ ] server benchmark tables
- [ ] speculative acceptance metrics
- [ ] baseline vs speculative comparison
- [ ] finalist stage transitions

### Acceptance gate

A selected Qwen MTP Candidate can be launched with llama-server, tested with SPEED-Bench, and compared against a non-speculative baseline with acceptance rate and delivered decode throughput stored in SQLite.

---

## 14. M10 — Local HTTP API

### Objective

Expose the domain/service layer for the browser UI without leaking raw database structure.

### Deliverables

FastAPI service with stable DTOs.

### Initial endpoints

~~~text
GET    /api/health

GET    /api/profiles
GET    /api/profiles/{id}

GET    /api/binaries
GET    /api/models

POST   /api/experiments
GET    /api/experiments
GET    /api/experiments/{id}

POST   /api/experiments/{id}/plan
POST   /api/experiments/{id}/run
POST   /api/experiments/{id}/pause
POST   /api/experiments/{id}/resume
POST   /api/experiments/{id}/cancel
POST   /api/experiments/{id}/clone

GET    /api/experiments/{id}/candidates
GET    /api/experiments/{id}/runs
GET    /api/experiments/{id}/results
GET    /api/experiments/{id}/matrix

GET    /api/runs/{id}
GET    /api/runs/{id}/telemetry
~~~

### Live updates

Prefer Server-Sent Events first unless bidirectional WebSocket behavior is actually needed.

### Work items

- [ ] FastAPI app
- [ ] DTO layer
- [ ] experiment endpoints
- [ ] result endpoints
- [ ] matrix endpoint
- [ ] telemetry endpoint
- [ ] SSE progress endpoint
- [ ] API tests

### Acceptance gate

All core experiment actions can be performed using HTTP only, while the CLI continues to call the same application service layer.

---

## 15. M11 — React frontend

### Objective

Implement the agreed simple-form UX over the complete backend.

### Screen 1 — Experiment list

Display:

- name
- model/profile
- status
- completed/total cases
- baseline
- validation state

### Screen 2 — New experiment

Flow:

1. select launcher profile
2. inspect resolved settings
3. select parameter dimensions
4. enter values
5. inspect constraints/candidate count
6. select workloads
7. select measurement policy
8. select placement mode
9. preview plan
10. save/run

### Screen 3 — Live execution

Display:

- progress
- current Candidate
- current WorkloadCase
- latest t/s
- process/system CPU
- GPU utilization
- VRAM/RAM
- temperatures
- power where available
- recent runs
- failure count

### Screen 4 — Results

For 2D spaces:

- matrix
- heatmap
- metric dropdown

For higher dimensions:

- X dimension
- Y dimension
- metric
- filters
- facets

### Screen 5 — Candidate detail/comparison

Display:

- configuration
- resolved placement
- PP results
- TG depth curve
- CPU
- GPU
- memory
- stability
- baseline delta
- request-latency estimates
- server validation

### UX rules

- advanced details collapsed by default
- unsupported binary parameters visibly disabled
- hashes hidden unless Advanced is opened
- invalid search combinations explained
- case-count preview before execution
- failures are visible, not silently omitted

### Work items

- [ ] frontend scaffold
- [ ] API client
- [ ] experiment list
- [ ] experiment editor
- [ ] parameter-dimension editor
- [ ] workload editor
- [ ] plan preview
- [ ] live run view
- [ ] matrix/heatmap
- [ ] higher-dimensional slicing
- [ ] candidate detail
- [ ] comparison screen
- [ ] baseline display
- [ ] Pareto view

### Acceptance gate

The full V1 reference experiment can be created, planned, started, monitored, and analyzed from the browser without editing JSON or using SQL.

---

## 16. M12 — Launcher promotion, archive, and export

### Objective

Complete the lifecycle from production profile to experiment to validated Candidate to proposed launcher update.

### Deliverables

- launcher-profile diff generation
- explicit promotion record
- archive command
- database snapshot
- export support

### Promotion behavior

V1 should produce a diff/patch first.

Example:

~~~text
Profile:
qwen-flash-gsq-rco-iq3-128k

--batch-size
4096 → 8192

--ubatch-size
2048 → 1024
~~~

Promotion metadata:

- experiment
- Candidate
- source profile
- source profile snapshot
- proposed profile
- validation result
- timestamp

### Archive behavior

llprof archive should:

1. checkpoint WAL
2. create a consistent SQLite snapshot
3. optionally include artifact files
4. emit manifest with hashes

### Work items

- [ ] launcher diff generator
- [ ] promotion persistence
- [ ] explicit apply workflow or patch output
- [ ] archive snapshot
- [ ] archive manifest
- [ ] experiment export

### Acceptance gate

A server-validated Candidate can generate a reproducible launcher-profile patch and the complete experiment database can be archived consistently.

---

## 17. M13 — V1 hardening and release

### Objective

Validate the complete system against the V1 acceptance scenario and prepare the first tagged release.

### Hardening work

- run full migration suite
- test interrupted runs
- test OOM cases
- test unsupported arguments
- test custom Qwen binary capabilities
- test native binary capabilities
- test malformed llama-bench JSON
- test server startup failure
- test stale/orphaned running state
- test noisy telemetry classification
- test profile changes after experiment creation
- test archive restore
- review SQL indexes using real query plans
- measure database growth
- document troubleshooting

### Reference V1 acceptance experiment

Base:

~~~text
Qwen3.8 Flash Next 128K profile
~~~

Search:

~~~text
batch_size:
2048, 4096, 8192

ubatch_size:
512, 1024, 2048, 4096

constraint:
ubatch <= batch
~~~

Expected candidates:

~~~text
11
~~~

Workloads:

~~~text
PP2048 @ d0
PP8192 @ d0
TG256  @ d4096
TG256  @ d50%
~~~

Expected benchmark cases:

~~~text
44
~~~

Requirements:

- per-candidate fit against 128K context
- fixed placement within each Candidate
- at least three repetitions
- CPU and GPU telemetry
- resumability
- matrix result views
- Pareto analysis
- finalist server validation
- launcher-profile diff

### Release deliverables

- tagged V1 release
- README installation/setup
- architecture summary
- benchmark workflow tutorial
- troubleshooting guide
- schema migration documentation
- known limitations

### Acceptance gate

All acceptance criteria in docs/technical-spec-v1.md pass on the primary workstation with both relevant Qwen models available.

---

## 18. Suggested issue/epic breakdown

The milestones above can map directly to GitHub issues or epics.

Recommended epics:

~~~text
EPIC-01  Project bootstrap
EPIC-02  Domain schemas and hashing
EPIC-03  SQLite persistence
EPIC-04  Experiment planner
EPIC-05  llama.cpp capability discovery
EPIC-06  llama-bench runner
EPIC-07  Placement resolver
EPIC-08  Telemetry
EPIC-09  Analysis and projections
EPIC-10  Server validation
EPIC-11  HTTP API
EPIC-12  Frontend
EPIC-13  Launcher promotion and archive
EPIC-14  V1 hardening
~~~

Each epic should have an explicit acceptance checklist copied from its milestone.

## 19. Dependency graph

~~~text
M0
└── M1
    └── M2
        └── M3
            ├── M4
            │   └── M5
            │       └── M6
            │           └── M7
            │               └── M8
            │                   └── M9
            │
            └──────────────────────────┐
                                       ↓
                                      M10
                                       ↓
                                      M11
                                       ↓
                                      M12
                                       ↓
                                      M13
~~~

M10 can begin once core services are stable, but it should not become the source of business logic.

M11 should wait until the API shape and analysis projections are reasonably stable.

## 20. Parallelizable work

After M3, some work can proceed in parallel.

Possible split:

### Track A — Execution

~~~text
M4 → M5 → M6 → M7
~~~

### Track B — Analysis foundations

Can begin once benchmark persistence is stable:

~~~text
statistics
matrix projection
Pareto logic
latency model
~~~

### Track C — Frontend design

Can prototype using mock API data after M3, but production integration should wait for M10.

### Track D — Fixtures/tests

Captured llama.cpp outputs, fake executables, and parser fixtures can be assembled continuously.

## 21. Recommended implementation checkpoints

Instead of waiting until V1 is complete, create usable checkpoints.

### Checkpoint A — Planner

User can run:

~~~text
llprof experiment plan
~~~

and inspect Candidates/WorkloadCases in SQLite.

No benchmarking yet.

### Checkpoint B — Microbenchmark runner

User can run deterministic llama-bench sweeps and resume them.

This is already useful.

### Checkpoint C — Accurate placement + telemetry

Results are production-context-aware and environmentally diagnosable.

This should be considered the first trustworthy tuning release.

### Checkpoint D — Analysis

Batch/ubatch and higher-order searches can be explored without SQL.

### Checkpoint E — Server validation

MTP and actual serving throughput are covered.

### Checkpoint F — Web UI

Normal experiments no longer require CLI configuration.

## 22. Definition of done for a feature

A feature is complete when:

1. Domain/schema behavior is defined.
2. Validation exists.
3. Persistence is implemented where applicable.
4. Failure behavior is defined.
5. Unit tests pass.
6. Integration tests exist where subprocess/database behavior is involved.
7. CLI/API behavior is exposed where appropriate.
8. Raw provenance remains recoverable.
9. Documentation is updated.
10. Existing experiment data remains migration-compatible.

## 23. V1 scope control

To keep V1 achievable, defer the following unless required by implementation discoveries:

- distributed workers
- multiple simultaneous benchmark executors
- remote authentication
- Bayesian search
- automatic cloud sync
- arbitrary user scripting inside experiments
- generalized support for engines other than llama.cpp
- automatic tuning that selects and applies a winner without review
- custom dashboard builders
- high-frequency sub-100 ms telemetry
- literal N-dimensional visualization

If a proposed feature does not directly improve correctness, reproducibility, benchmark coverage, or the agreed UX, it should generally wait until after V1.

## 24. First implementation target

The first concrete end-to-end target should be:

~~~text
Base profile:
qwen-flash-gsq-rco-iq3-128k

Search:
batch_size × ubatch_size

Candidates:
11 valid combinations

Workloads:
PP2048 @ d0
PP8192 @ d0
TG256 @ d4096
TG256 @ d50%

Output:
SQLite experiment plan
~~~

This target exercises the core abstractions without requiring process execution.

The next target adds llama-bench execution to the exact same stored plan.

This sequencing validates the architecture before hardware/process complexity is introduced.
