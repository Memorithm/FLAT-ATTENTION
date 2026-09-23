# MAA-13e physical sparse-routing qualification preregistration

Status: preregistered physical-qualification protocol. No latency, throughput, traffic, or speedup observation is made by this commit.

## Entry evidence

MAA-13e follows the MAA-13d `CONFIRMATORY_PASS` preserved on main. The candidate is frozen:

- policy: `tiered_recall`;
- signature width: 128 bits;
- projector seed: `0x514b_5349_474e_3133`;
- admit Hamming distance <=68;
- for 68<distance<=72, admit only key norm >=1.5;
- otherwise reject.

No threshold retuning is permitted in MAA-13e.

## Question

Does the frozen sparse-routing candidate reduce physical end-to-end attention cost on declared hardware after including routing, candidate materialization, synchronization and sparse numerical execution, while preserving the numerical/quality diagnostics required by the MAA-13d contract?

Logical candidate density is an explanatory variable, not a speedup measurement.

## Required execution arms

For identical input geometry and data, measure:

1. dense FLAT numerical reference/execution;
2. structural sparse execution with a precomputed candidate set, isolating sparse-kernel potential;
3. end-to-end tiered-recall routing plus candidate materialization plus sparse execution;
4. density-matched random routing plus sparse execution;
5. all-accept routing plus sparse execution as routing-overhead control.

Where V888 structural routing is used as the sparse carrier, its candidate semantics MUST remain identical to the merged FA-V888-1 oracle.

## Measurement boundary

Each accepted timing record MUST bind:

- exact source commit;
- device, backend, driver/runtime identity when available;
- batch, heads, query/KV lengths, head dimension and dtype;
- warm-up count and measured iteration count;
- candidate density and executed pair count;
- routing time;
- candidate materialization time where separately measurable;
- sparse numerical execution time;
- synchronization/readback time if included;
- total end-to-end time;
- dense total time under the same timing boundary;
- p50, p95 and p99 latency plus throughput where meaningful.

Software Vulkan/lavapipe/WARP may validate correctness and timing plumbing but MUST NOT be generalized as physical-device performance.

## Correctness and quality gate

Timing is accepted only after:

- sparse execution matches the authoritative masked numerical oracle within the declared tolerance;
- no empty effective row is silently replaced by dense or zero output;
- candidate identity/order accounting is exact;
- retained-mass and O/LSE diagnostics remain available for the measured selection;
- all-accept control reproduces dense semantics.

## Physical promotion rule

No speedup threshold is assumed in advance. MAA-13e may report measured ratios only for the exact qualified device/workload.

A physical candidate is eligible for further integration only if end-to-end tiered-recall latency is strictly lower than dense under the same boundary and its correctness/quality gates pass. Precomputed-mask speedup without end-to-end speedup is retained as kernel-potential evidence only.

## Non-claims

This protocol does not establish cross-device generalization, real-model quality, production readiness, novelty or energy savings unless independently measured. It does not authorize default routing.
