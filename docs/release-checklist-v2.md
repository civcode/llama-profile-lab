# V2 release checklist

Use this checklist after V2-M10 implementation changes and before declaring workstation acceptance.

## Source and migration

- [ ] `main` contains the intended release commit set.
- [ ] No required work exists only on an abandoned feature branch.
- [ ] Schema migration count/version is 15 or the documented newer version.
- [ ] Historical migration-prefix upgrade tests pass.
- [ ] SQLite integrity and foreign-key checks pass.
- [ ] `llprof database check` reports indexed deployment plan/run/promotion queries.

## Locked backend gates

```bash
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest
```

- [ ] All commands pass from a clean checkout.
- [ ] No GitHub Actions workflow was made automatic; workflows remain manual-only unless deliberately changed.

## Frontend gates

```bash
cd frontend
npm ci
npm run typecheck
npm run test
npm run build
```

- [ ] All commands pass.
- [ ] Deployment authoring/planning/run/results/Pareto/promotion pages are keyboard usable.
- [ ] Production bundle is served by `llprof ui`.

## Hardening

- [ ] Malformed memory-estimator output fails closed.
- [ ] Helper/server binary SHA drift fails closed.
- [ ] Device disappearance/remapping after planning fails before server startup.
- [ ] Runtime memory-margin violations are persisted.
- [ ] Startup OOM and partial startup tear down all members.
- [ ] Cleanup errors are persisted.
- [ ] Stale deployment state recovers safely.
- [ ] Corrupted concurrent output becomes `output_validation_failed`.
- [ ] Missing/ambiguous standalone baselines remain explicit.
- [ ] Partial telemetry does not masquerade as complete evidence.
- [ ] Launcher source drift refuses promotion.
- [ ] Partial coordinated promotion refuses promotion.

## Archive and provenance

- [ ] `llprof deployment export DEPLOYMENT_ID` contains immutable, planning, execution, environment, and promotion sections.
- [ ] Full archive restore preserves V2 deployment/promotion tables.
- [ ] Restored archive passes SQLite integrity and foreign-key checks.
- [ ] Acceptance export and archive hashes are recorded.

## Acceptance report

Generate the persisted-evidence report after the workstation campaign:

```bash
llprof deployment acceptance-report DEPLOYMENT_ID \
  --minimum-devices 2 \
  --minimum-phase-repetitions 3 \
  --format json \
  --output workstation-acceptance.json \
  --database data/benchmarks.db
```

- [ ] Machine-checkable evidence reports `machine_checks_passed: true`.
- [ ] Any `fail` item is resolved or explicitly blocks release.
- [ ] `pareto_finalist_selection` is completed in the acceptance notes.
- [ ] `target_hardware_model_identity` is confirmed against the intended workstation/artifacts.
- [ ] `automated_clean_checkout_gates` is completed separately; the report does not infer CI/test status from SQLite.

## Target workstation

Record exact hardware/build identities before benchmarking.

- [ ] RTX 4070 Ti Super identity and 16 GiB capacity recorded.
- [ ] AMD GPU identity/backend/capacity recorded from the actual build.
- [ ] Driver versions and stable physical IDs recorded.
- [ ] llama.cpp server/helper commit/build identity and SHA-256 recorded.
- [ ] Qwen3.8 27B and Flash Next model artifact identities recorded.
- [ ] Mixed-vendor telemetry shows both GPUs concurrently.

### Memory truth

- [ ] At least two context/KV settings per model compared projected vs runtime memory.
- [ ] Signed peak-use/free-headroom deltas recorded.
- [ ] Any systematic overhead policy is explicit and justified.

### Topologies and phases

- [ ] One-model-per-GPU baseline completed.
- [ ] At least one both-models-on-both-GPUs feasible topology completed.
- [ ] DD completed.
- [ ] PP completed.
- [ ] PD completed.
- [ ] DP completed.
- [ ] Per-instance standalone/concurrent retention available.
- [ ] No hidden swap or unintended CPU fallback.
- [ ] Output correctness valid.

### Search and finalists

- [ ] Bounded search includes multiple Qwen and Flash allocations/context/KV coordinates.
- [ ] Known overcommit points are pruned.
- [ ] Pareto frontier contains only valid evidence.
- [ ] Non-dominated finalists repeated at least three times per canonical phase.
- [ ] Finalist telemetry meets the chosen clean/thermal policy.

## Promotion and release artifacts

- [ ] One coordinated proposal covers every affected profile.
- [ ] Proposal includes exact model, binary/helper, placement, memory, runtime, and correctness provenance.
- [ ] Unified patch reviewed; launcher source was not mutated by proposal generation.
- [ ] Complete deployment export saved.
- [ ] Full archive saved.
- [ ] Known limitations updated from measured workstation behavior.
- [ ] Release notes include exact acceptance commit and artifact identifiers.
