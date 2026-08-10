# M15 — pre-rotated K resident decode interoperability

SciRust's resident decode cache stores **RoPE-rotated K** and raw V at append time. The general FLAT M11 rectangular kernel normally receives raw projected K and rotates K internally. Feeding SciRust's cache to that default path would therefore rotate K twice.

M15 adds an explicit interoperability encoding mode to the already-qualified M11 pipeline:

```text
encode(...)               => raw Q, raw K, raw V; fused Q+K RoPE
encode_pre_rotated_k(...) => raw Q, RoPE K, raw V; fused Q RoPE only
```

The default `encode` behavior is unchanged.

## Why this mode exists

A second raw-K shadow cache would duplicate resident memory and cache append traffic. Un-rotating K before attention would add work and a temporary buffer. Rebuilding K from host data would violate the zero-copy resident contract.

Instead, the kernel carries one uniform `rotate_k` flag. When disabled, K tiles are loaded directly from the resident cache and enter Q·K without another rotary transform. Q is still rotated in workgroup memory at its absolute decode position. V is never rotated.

The mode keeps:

- four storage bindings (Q, K, V, O|LSE);
- native GQA/MQA KV cardinality;
- FP32 score accumulation and online-softmax state;
- no materialised score/probability matrix;
- caller-owned command encoder and submission;
- the same portable WGPU/WGSL stack.

## Qualification

Device tests construct raw K, produce an independently pre-rotated projection-layout cache on the CPU, run `encode_pre_rotated_k`, and compare O/LSE against the existing scalar M11 oracle operating on the original raw K.

Coverage includes:

- GQA decode, D=64, KV length 17;
- MQA decode, D=80, KV length 65 (crossing multiple KV tiles).

Existing M11 tests continue to exercise `encode` with raw K, proving its behavior remains qualified.

## Performance honesty

This change establishes correctness and zero-copy cache compatibility. It does not claim lower decode latency until a paired same-adapter benchmark against SciRust's existing incremental attention path is recorded.

No project-authored C/C++, C ABI bridge, CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or mandatory vendor SDK is introduced.
