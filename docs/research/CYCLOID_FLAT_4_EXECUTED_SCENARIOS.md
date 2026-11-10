# CYCLOID-FLAT-4 — executed multi-scenario sparse attention

**Status:** research-only host numerical diagnostic; not a GPU acceleration
claim, an online prefilter or a default inference route.

FLAT-CYCLOID-1 introduced candidate scoring against the dense Q/K and
softmax mass/output error controls; FLAT-CYCLOID-2 added canonical sparse
numerical execution. FLAT-CYCLOID-3 introduced a strict, multi-scenario
cost selector. The fourth slice closes the **functional execution gap**
by consuming a multi-scenario winner as an actual numerical K/V selection.

\`cycloid_scenario_execution::execute_cycloid_scenario_reference\`
evaluates one \`CycloidScenarioExecutionRow\` for every original
batch/head/query row:

1. Executes the existing full baseline \`execute_cycloid_reference\`.
   This validates Q/K/V shape, finiteness, all original candidate key
   identities, and future-key exclusion in causal mode; it also
   retains a fully independent dense reference output/LSE.
2. Recomputes each query's full eligible dense Q/K scores from the
   exact original data and the authoritative attention scale, and uses
   the maximum actual absolute value in the source head.
3. Calls \`choose_scenario_robust_attention\`, which rejects candidates
   failing either the original retained-softmax-mass/numerical-error
   tests or **any** of the 2–8 declared cost regimes. It preserves
   the original candidate ID and key positions, or selects dense.
4. Reconstructs a canonical numerical row containing **only** its
   qualified original sparse candidate, or no candidates for dense.
   Executes it through the pre-existing FLAT structural attention
   oracle. The baseline-only and robust numerical outcomes are
   independently inspectable, including output and LSE differences.
5. Requires both independent dense oracles to agree and verifies the
   actually executed candidate ID and key-set identity against the
   multi-scenario decision. Any drift fails closed.

Four regression tests exercise a different robust winner actually
executed through the numerical reference, dense execution after an
unfavorable scenario, causal future-key rejection, and malformed
candidate IDs. All whole-workspace CI gates are required.

## Evidence limits

This adapter intentionally recomputes dense Q/K scores, runs dense
numerical reference execution twice, and validates every route before
calling the canonical sparse executor. **Its total work is larger
than the original dense-only baseline.** It qualifies semantics and
cost/quality control logic, not inference speed.

The 2–8 scenario costs are externally declared estimates, not
attested GPU-device observations. A scenario may represent cache state,
read order or device/backend, but the value is only as good as the
external measurement and versioned provenance. A reduction in logical
sparse pairs does not prove reduced KV physical traffic, bandwidth,
wall-clock latency or end-to-end language-model quality.

No new FLAT stable API, shader, GPU default dispatch, model quality
claim, SML-GENIUS dependency or vendor-specific GPU SDK is introduced.
Runtime promotion would require a separately qualified cheap router,
whole-workload paired real-device measurements through RemoteOps,
numeric/model parity and explicit net gain after all routing overhead.
