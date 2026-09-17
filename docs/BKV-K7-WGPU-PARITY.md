# FLAT BKV-4 / KVLab BKV-K7 portable WGPU candidate-set parity

Status: research infrastructure. This is a correctness gate, not a performance result.

KVLab BKV-K7 requires the portable WGPU Boolean router to produce exactly the
same candidate set as the deterministic CPU Hamming oracle for identical frozen
signatures and policy before any CPU-vs-WGPU timing comparison is admissible.
FLAT now exposes the research schema `flat.boolean-kv-wgpu-parity.v1` through
`api::wgpu_boolean_router::BooleanKvWgpuParityEvidence`.

The parity record is constructed only from the actual u32 admission-buffer
readback for the exact `BooleanWgpuRouterPlan` used to encode the WGPU dispatch.
It retains the signature width, threshold, exact packed query/key words, CPU
candidate IDs and WGPU candidate IDs. Malformed readback fails closed on a length mismatch or non-binary flag. A valid
but different WGPU candidate set is retained as explicit negative evidence with
`exact_candidate_set_match=false`; `require_exact_match()` then rejects that
record before any CPU-vs-WGPU performance comparison.

The canonical JSON includes an FNV-1a checksum only for deterministic
corruption detection. It is not cryptographic attestation and does not prove
adapter identity, physical-GPU execution or trustworthy remote execution.

The mandatory WGPU CI gate executes the router on its WGPU adapter and verifies
candidate parity. GitHub CI currently uses a software Vulkan device for this
matrix; success there is portable backend correctness evidence, not real-GPU
performance evidence. A KVLab BKV-K7 / FLAT BKV-4 performance claim remains blocked on separately
preregistered real-device measurements with exact hardware/runtime provenance.

This slice contains no latency, bandwidth, energy, TTFT, TPOT, tokens/s, model
quality or numerical-KV traffic claim. Dense numerical FLAT remains the fallback
and quality authority.
