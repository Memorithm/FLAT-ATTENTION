# M8 qualification history: portable binary16 on WGPU 0.20

This note preserves the qualification result that led FLAT-ATTENTION M8 away from native WGSL `f16` syntax and toward the packed-binary16 design merged in PR #10.

## Context

The first M8 experiment attempted to expose a native shader-f16 path on top of the SciRust-aligned WGPU 0.20 / Naga 0.20 stack.

The isolated qualification gate demonstrated that the intended native WGSL `f16` route was not a reliable contract for the qualified stack. FLAT therefore did not weaken shader validation, upgrade the runtime solely for this feature, or introduce CUDA/C++/WMMA/vendor tooling.

## Retained design

The accepted M8 path stores two IEEE-754 binary16 scalars in each `u32` storage word and uses baseline WGSL storage conversion at the kernel boundary.

The numerical contract remains:

- Q/K/V/O use binary16 representation at the storage boundary;
- Q/K/V values widen to `f32` before attention arithmetic;
- Q·K accumulation remains `f32`;
- online-softmax max, exponentials and normalization remain `f32`;
- P·V accumulation remains `f32`;
- O is rounded back to binary16 only at final writeback;
- LSE remains `f32`;
- no N×N score or probability matrix is materialized.

## Why the original PR was superseded

PR #9 explored the same milestone while its public test and benchmark vocabulary still referred to `NativeF16`, `native_f16_supported`, and `RequiredF16Unavailable`.

The implementation itself had already moved to the portable packed representation, so those names no longer matched the actual contract and caused CI compilation failures.

PR #10 completed M8 with the final packed-binary16 API and was merged before PR #9. Subsequent milestones were built on that merged implementation. Reapplying the old PR #9 code would therefore regress the current API and conflict with later work.

## Permanent engineering rule

A reduced-width storage path must never be described as native arithmetic unless the selected open shader/runtime stack actually exposes, validates and executes that arithmetic capability.

Representation-size facts are not performance claims. Binary16 storage uses half the bytes per scalar of `f32`; runtime speedup, bandwidth improvement or energy reduction require reproducible physical-device measurements.

## Sovereignty

This decision preserves FLAT-ATTENTION's project constraints:

- no CUDA C++;
- no `nvcc`;
- no WMMA/WGMMA API dependency;
- no CUTLASS/cuDNN dependency;
- no project-authored C ABI bridge;
- no vendor SDK required by the FLAT core;
- portable WGPU/WGSL remains the qualified execution path.

The active precision contract is documented in `docs/m8-precision.md`.
