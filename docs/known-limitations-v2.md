# V2 known limitations

These are current product/acceptance boundaries, not silent assumptions.

## Target workstation acceptance is still required

Synthetic and repository fixtures cannot prove the actual NVIDIA + AMD workstation, model builds, drivers, thermal behavior, or estimator accuracy. V2 is not workstation-accepted until the phases in the V2 roadmap/release checklist have been executed on the target hardware.

## Full automated gates must be run from a real checkout

The implementation includes backend and frontend test coverage, but release acceptance still requires the repository-wide locked Python gates and the frontend `npm ci` / typecheck / test / build gates from a runnable checkout. GitHub Actions are intentionally manual-only.

## Coordinated promotion is review-only

V2 produces and persists one unified launcher proposal but does not edit launcher configuration.

Resolved device/tensor/override placement arguments are encoded in the review patch as launcher argument values. The older V1 launcher-to-Candidate parser does not claim to round-trip every V2 placement-list argument back into a V1 Candidate; the persisted V2 placement and promotion evidence remain authoritative.

## Exact device inventory is required for execution

Deployment execution now re-discovers exact-binary device inventory. A binary that cannot expose the planned selectors fails closed rather than falling back to an unverified topology.

## Runtime projection deltas include the whole observed device state

Runtime GPU used/free telemetry is device-wide. External processes can therefore make an estimator appear pessimistic or optimistic. Workstation acceptance requires clean telemetry and an otherwise controlled host.

## Retention depends on exact baselines

V2 does not interpolate or guess standalone throughput. Missing/ambiguous exact baselines remain visible quality states.

## Pareto results are evidence-sensitive

Failed and correctness-invalid runs are retained in raw results but excluded from valid frontier evidence. Exact hidden coordinates must be filtered/faceted rather than silently averaged.

## Search remains grid-based

The persisted V2 search strategy is deterministic grid expansion. Adaptive/Bayesian joint placement search is outside the current milestone.

## Workstation-specific overhead policy is not yet frozen

The implementation reports runtime-versus-projected memory deltas but does not automatically inflate future estimates. If target-hardware acceptance finds a repeatable systematic overhead, record an explicit policy and its measured basis before changing planner margins.
