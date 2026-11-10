# CYCLOID-FLAT-3 — multiple cost-regime qualification

Research-only Rust module:
\`crates/flat-algebraic-attention/src/cycloid_scenarios.rs\`.

CYCLOID-FLAT-1 screens independently generated sparse-key sets against the
complete dense numerical Q/K score vector and bounded softmax mass/error.
CYCLOID-FLAT-2 actually runs selected sparse keys through the existing
canonical numerical oracle. Neither establishes a physical speedup.

The new \`choose_scenario_robust_attention\` addresses a missing systems
gate: a candidate that wins in one *estimated* access-cost regime may lose
in another due to memory/cache/backend conditions. The module requires
2–8 explicitly identified, complete and nonduplicated cost scenarios.
Every scenario reports the dense baseline and costs keyed by exact
candidate ID. No positional remapping of numerical key identities is
permitted.

For each candidate, the existing exact score oracle first determines
whether retained probability mass, max output-error bound and read
budget qualify it in the baseline. Then **every** additional scenario
must satisfy \`candidate_total_ns < dense_total_ns\` and the same explicit
read-byte budget. Invalid/unmapped/duplicated IDs, zero/overflowed
costs and malformed scenario cardinality fail closed.

Among candidates surviving all scenarios, the function minimizes
the worst *relative* estimated cost (candidate/dense) across the baseline
and scenarios, using exact u128 cross-multiplication rather than
floating-point ties. Baseline time, read bytes and candidate ID provide
stable deterministic tie-breaks. The complete original key positions
are retained. If no sparse choice passes, the original dense numerical
key set is returned.

This can select a different sparse candidate from the single-regime
baseline without weakening numerical eligibility. It does NOT run
dense Q/K scoring faster, avoid the dense-reference diagnostic cost,
or alter the default FLAT WGPU dispatch. It does not establish a
physical GPU memory-traffic reduction, model quality, KV-cache
speedup or causal correctness automatically; caller-derived candidates
must still honor original shape, mask and key-epoch identities.

The design transfers the **independent regime robustness** principle
from SML-GENIUS Cycloid-6 to FLAT without linking the repositories.
Unlike SML, the numeric attention unit retains its own softmax error
bounds and dense reference; SML's native Boolean MOR pages, graph
planner and Rust RAM lifecycle are not imported.

Qualification requires full FLAT exact-head CI, including Rust,
semver, MAA host and WGPU-device jobs where applicable. Physical
promotion requires pinned hardware and real-device paired experiments
under RemoteOps, with memory, latency, and retained numerical/model
quality measured together.
