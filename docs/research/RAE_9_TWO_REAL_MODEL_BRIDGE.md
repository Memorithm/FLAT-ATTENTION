# RAE-9 — real-model Q/K/V integration (GPT-2 XL and SmolLM2-135M)

Status: **research-only, gated by RemoteOps real-checkpoint evidence**. The Rust code is an executable connector between live model attention activations and the RAE-5 → RAE-7 → RAE-8 numerical pipeline. CI fixture tests are deterministic **synthetic inputs**, NOT trained-weight validation.

## Selected model families and deployment boundary

1. **GPT-2 XL** — \`openai-community/gpt2-xl\`, existing local CPU-only safetensors install and checksum gate in \`Memorithm/RemoteOps/ops/models/gpt2-xl-dell.sh\`. MHA: one KV head per query head. Check model weight SHA-256 on Dell before any capture.
2. **SmolLM2-135M-Instruct** — \`HuggingFaceTB/SmolLM2-135M-Instruct\`, Llama-family RoPE and grouped heads (nine Q heads, three KV heads in the published configuration). Extract the *post-RoPE* head-specific Q/K/V actually sent to SDPA after each model's own transformations. Use a pinned trained weight revision and CPU-only operation, controlled exclusively by \`Memorithm/RemoteOps\`.

These are *external, real pretrained models used by Memorithm's model research and control workflows*; neither is a newly trained proprietary SML-GENIUS checkpoint. Boolean-only SML/V888 state is NOT silently reinterpreted as Transformer QKV.

## Snapshot format (RAE9QKV1)

One bounded, ephemeral UTF-8 whitespace-delimited record (not JSON):

\`\`\`text
RAE9QKV1 <gpt2-xl|smollm2-135m-instruct> <kv_rows> <qk_dim> <value_dim> <page_size> <score_scale> <generation> <query_heads> <kv_heads> <q_head> <kv_head>
<previous_query[qk_dim]>
<last_query[qk_dim]>
<post-transform_keys[kv_rows*qk_dim]>
<post-transform_values[kv_rows*value_dim]>
\`\`\`

- K/V are the exact original, **unmasked last-query causal** key sequence for the chosen head. No left padding, additive bias, dropout, position-specific extra score terms, or cached future rows are admitted in this initial contract. Full-prefill, batch-one, nonpadded last token only.
- \`kv_head\` must equal \`q_head / (q_heads / kv_heads)\` and respect original K/V sequence order. If the runtime has already repeated grouped K/V heads, use the repeated effective head count for this capture.
- Query/K/V samples are finite and in the model's **post-transformation SDPA space**; score scale is exactly what SDPA receives (default \`1/sqrt(d)\`). Use an independent comparison against the native framework's attention output before allowing an injected research head.
- The producer creates one generation for a single immutable in-process snapshot. The Rust connector derives all K-norm upper bounds, exact current per-page key counts and V absolute bounds by **scanning the full snapshot**, so neither physical memory savings nor a safe incremental metadata lifecycle is established.
- RAE-5 sign flips between the two consecutive model query vectors are mapped by a deliberately untrained coordinate-modulo-page control. The resulting wakes propose pages but do not certify relevance.
- RAE-7 may promote pages; RAE-8 computes actual selected QK and V accumulation, or a **full numerical dense fallback**. An empty event, invalid metadata, malformed grouped-head mapping or failed error budget must never create an unqualified sparse output.
- Maximum capture size 16 MiB, up to 4,096 causal K/V rows, 256 QK and value dimensions, 512 pages. Invalid input exits nonzero.

Example:

\`\`\`bash
cargo test -p flat-algebraic-attention --example rae9_real_model_bridge
cargo run -p flat-algebraic-attention --example rae9_real_model_bridge -- /path/to/ephemeral-snapshot.txt
\`\`\`

The executable outputs \`RAE9_RESULT\` metadata and \`RAE9_OUTPUT\` components, in original value-head coordinates. \`dense_abs_difference\` is compared to the RAE-7 output-error bound. These are *host f64 numerical counters and discrepancies*, not GPU throughput, physical bandwidth, formal intervals or model-quality measurements.

## Required real-model gate (RemoteOps)

At a pinned FLAT commit, RemoteOps must verify host load, current model identity, model weight revisions, CPU-only HF runtime, and compile the adapter from a controlled exact source. Execute an unmodified model forward and a model forward with one model attention head instrumented. Capture native post-transform Q/K/V in memory, run the adapter through a short-lived child process, verify the independent original framework attention result against the Rust dense counterpart, and optionally inject the RAE output on a strictly sparse selected head. Report both model last-token logits, max component discrepancy, original and candidate top token, selected/fallback rate, bounded CPU time and per-row logical score counts. Keep raw QKV and model inputs ephemeral and local.

Gate fails closed for unsupported attention implementation or mask, dropout, GQA mapping, nonfinite activations, unsupported precision, missing model, unexpected replay divergence, different model revisions, or unverifiable native-vs-Rust attention parity.

**No user-visible trained-model quality, decode speedup, energy, memory or real GPU statement until both model runs complete and produce inspectable evidence.** The RemoteOps run is a host-side controlled shadow/one-head intervention, NOT an optimized end-to-end attention backend, GPU kernel or continuous training adapter. Raw checkpoint files must never be uploaded as CI artifacts.

## RAE-9 SciAgent extension

The third explicitly tagged model identity, `sciagent`, is reserved for the
actual Rust-native GQA model in `Memorithm/scirust/scirust-sciagent`.
The SciRust producer reads **post-RoPE** Q/K and projected V from its
existing differentiable attention tape, preserving original (not repeated)
GQA head mapping. The snapshot protocol, native dense oracle, strict parsing,
maximum dimensions and failure/fallback semantics are unchanged.
This extension does not promote RAE-9 into SciAgent's resident WGPU M32/M33
attention, does not provide backward gradients for injected numerical output,
and makes no physical I/O, performance or model quality claim.
