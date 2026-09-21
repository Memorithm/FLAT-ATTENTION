# BANC V888 sparse-routing bootstrap for FLAT-ATTENTION

Status: research track. Existing qualified softmax attention remains the correctness baseline.

## Scope

This track uses only FlyWire BANC v888 as an external source of structural hypotheses. No BANC data is vendored into FLAT-ATTENTION. V888 is used to derive graph statistics and controlled sparse routing patterns through SciRust-generated artifacts.

Codex currently identifies BANC v888 as Female Adult Fly Brain and Nerve Cord, snapshot 2026-05-20, 158,262 neurons and 3,037,361 aggregated connections.

Sources:
- https://codex.flywire.ai/?dataset=banc
- https://codex.flywire.ai/faq

## Why FLAT-ATTENTION is involved

FLAT-ATTENTION already contains:
- a deterministic streaming-softmax oracle;
- portable WGPU kernels;
- Boolean admission and F2/ANF research in MAA;
- evidence gates separating host semantics from hardware claims.

V888 provides a large real sparse directed graph from which to ask a precise question:

Can structural pre-routing reduce the key/value candidate set while preserving task quality better than density-matched and topology-matched synthetic controls?

This does not imply that biological connectivity is attention.

## Bootstrap sequence

### FA-V888-0 — frozen research contract

Define:
- V888-only source;
- graph-to-attention mapping is an experimental transformation, not a biological claim;
- raw V888 data remains external;
- exact candidate density must be reported;
- all comparisons retain dense FLAT softmax as reference;
- topology controls are mandatory.

### FA-V888-1 — host structural-mask oracle

Add a research-only host representation for sparse admissibility:
- CSR/segment candidate lists per query;
- deterministic mask canonicalization;
- explicit empty-set semantics;
- no duplicate key admission;
- causal-mask intersection;
- exact admitted-key counters.

Validate against dense masked softmax.

### FA-V888-2 — matched mask families

Produce mask families at identical density:
- uniform random;
- local-window;
- block/modular;
- degree-matched;
- reciprocity/motif-informed;
- V888-derived structural prior.

No V888 result is interpretable without these controls.

### FA-V888-3 — MAA structural admission

Extend the existing Boolean-front-end research boundary:
- structural edge predicate;
- existing Boolean/F2 predicates;
- temporal/causal readiness;
- optional module/hub predicate.

The admitted set is the conjunction/disjunction defined by a frozen expression, then normal FLAT numerical attention operates only on that set.

Keep Boolean-only and softmax-only arms.

### FA-V888-4 — sparse streaming-softmax oracle

Implement a scalar Rust oracle that streams only admitted keys while preserving:
- numerically stable online max;
- online normalization;
- LSE;
- exact causal semantics;
- deterministic ordering.

Compare to dense masked reference within the existing numerical policy.

### FA-V888-5 — portable WGPU sparse kernel

After host correctness:
- compact candidate index buffers;
- segment offsets;
- workgroup-friendly indirect or segmented traversal;
- bounded scratch memory;
- no N x N matrix;
- no CUDA/nvcc/vendor SDK dependency.

Benchmark candidate density, memory traffic proxy and latency separately.

### FA-V888-6 — graph recurrent / attention hybrid

Create a research adapter for states produced by the SciRust sparse recurrent engine:
- recurrent graph proposes candidate modules/keys;
- FLAT performs restricted numerical attention when authorized;
- attention output may feed the next recurrent graph step;
- switching and reconstruction costs are measured.

This is a hybrid baseline for SML/TDI, not a change to the stable API.

### FA-V888-7 — hub-aware routing experiment

Test whether hub-like graph structures are useful for attention routing:
- direct local candidates;
- hub summary candidates;
- local + hub union;
- degree-matched pseudo-hubs.

Measure quality versus candidate density. Do not label hubs as biologically meaningful in the artificial model unless source annotations are explicitly used.

### FA-V888-8 — learned structural router

Only after fixed masks:
- learn sparse admission logits or Boolean predicates;
- freeze to a hard bounded candidate structure for inference;
- constrain total admitted edges;
- compare learned-from-random initialization with learned-from-V888 prior;
- detect collapse to dense admission.

### FA-V888-9 — NNIS runtime handoff

Publish a backend-neutral execution contract:
- sparse candidate buffers;
- mask metadata;
- tensor layout;
- LSE/output contract;
- capability requirements;
- exact counters.

NNIS owns runtime scheduling and device integration. FLAT owns attention semantics.

### FA-V888-10 — SML-GENIUS comparison gate

Provide external baselines for SML:
- dense FLAT;
- conventional sparse controls;
- V888-derived pre-routing + FLAT;
- recurrent-graph + restricted FLAT.

SML candidate code remains non-attention. These arms exist to determine whether SML gains come from eliminating attention or merely from sparse routing.

## Performance evidence

Every V888-related benchmark must record:
- exact commit;
- source topology fingerprint;
- query/key dimensions;
- candidate density distribution, not only mean;
- batch/heads/sequence/head dimension;
- CPU/WGPU adapter;
- warmup and measured iterations;
- median/p95 where practical;
- exact logical candidate loads;
- output error versus dense oracle.

No NVIDIA-specific path is required or authorized by this bootstrap.

## Exit gate

The track may graduate from research-only status only if a sparse structural arm:
1. is numerically correct against dense masked attention;
2. produces reproducible quality evidence on a declared task;
3. is compared against density- and structure-matched controls;
4. demonstrates a measured resource/latency advantage on at least one non-NVIDIA target;
5. preserves a fully portable fallback and deterministic Rust oracle.
