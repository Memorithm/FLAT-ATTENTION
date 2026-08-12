# M38 — Property and stress qualification

M38 adds deterministic property-style coverage around the scalar oracle, portable grouped WGPU execution, and the resident KV-cache lifecycle.

## Deterministic corpus

The property suite uses an internal fixed-seed linear-congruential generator rather than an external randomness source. Every CI run therefore exercises the same 64 grouped-attention cases and can reproduce a failure from the repository alone.

The generated matrix varies batch size, MHA/GQA/MQA head grouping, sequence lengths across tile boundaries, head dimensions through the portable maximum, and causal/non-causal mode. Inputs remain finite and bounded.

## Invariants

The suite verifies:

- all scalar O/LSE outputs stay finite for every generated case;
- attention normalization by replacing V with all ones and requiring O to remain one;
- for causal attention, query position zero observes exactly the first physical KV row for its mapped KV head;
- extreme-but-finite score fixtures remain finite under online softmax and preserve normalization;
- one resident WGPU Q/K/V set can be dispatched and read back repeatedly without numerical drift from the scalar oracle;
- resident K/V cache reset changes logical length only and permits subsequent rows to reuse the same fixed-capacity storage contract.

When `FLAT_REQUIRE_WGPU=1`, inability to obtain a WGPU device is a test failure. Otherwise device tests may skip on hosts without a GPU adapter while the scalar properties still run.

## Evidence boundary

M38 is correctness and robustness evidence. Repeated-dispatch execution is not a throughput benchmark and introduces no performance claim.

## Sovereignty

The tests and runtime remain Rust-native plus WGPU/WGSL. No project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK is introduced.
