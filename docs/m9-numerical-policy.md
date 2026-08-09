# M9 — Numerical policy layer

M9 makes the numerical behavior of FLAT-ATTENTION an explicit public contract. It does not replace the M1–M8 kernels and it does not weaken their validation.

## Modes

### `ExactReference`

Purpose: project oracle and debugging baseline.

- backend: scalar Rust CPU reference;
- Q·K accumulation: serial `f32` in increasing head-dimension order;
- key traversal: serial increasing key position;
- P·V accumulation: serial `f32`;
- softmax: stable online max/sum update;
- subgroup: never used;
- expected repeatability: identical FP32 bit patterns for repeated identical calls under the same Rust build/runtime/platform contract.

`ExactReference` means **exactly the FLAT-ATTENTION reference evaluation order**. It does not mean real-number exact arithmetic or cross-platform transcendental bit identity.

### `FastPortable`

Purpose: normal qualified f32 GPU execution.

- backend: WGPU only;
- runtime may select the qualified M5 subgroup reduction when exposed;
- otherwise it uses the qualified Q4 fixed-tree path;
- M6 vectorized f32 storage may be selected for D64/D128;
- M7 remains disabled by default, as required by its physical-benchmark gate;
- all input, shape and scale validation remains active;
- no same-device bit-repeatability promise is made because the reduction family is capability-dependent.

A WGPU construction failure is returned as an error. `FastPortable` never falls back to the CPU reference.

### `DeterministicPortable`

Purpose: reproducible f32 GPU evaluation on one backend/device contract.

- backend: WGPU only;
- subgroup reduction is forcibly disabled;
- M7 double buffering is disabled;
- D64/D128 may retain M6 vectorized storage because it does not change the 64-lane reduction topology;
- Q·K first stage: fixed mapping of at most two dimensions per lane;
- Q·K reduction: fixed 64-lane shared-memory binary tree;
- online-softmax state: FP32;
- P·V accumulation: FP32;
- expected repeatability: exact FP32 bit equality for repeated identical calls on the same context/backend/device.

The permanent device gate runs repeated executions and compares `f32::to_bits()` for every O and LSE element.

This guarantee is intentionally local. M9 does **not** promise identical bit patterns across GPU vendors, driver versions, shader compilers or different transcendental implementations.

## Shared stable-softmax policy

All modes retain the same algorithmic invariant:

1. compute a score in FP32;
2. update the running row maximum;
3. rescale prior softmax state by `exp(old_max - new_max)`;
4. compute the current numerator as `exp(score - new_max)`;
5. update the running normalization sum and P·V accumulator;
6. normalize O only after all eligible keys have been consumed;
7. emit `LSE = running_max + log(running_sum)`.

Because exponent arguments are measured relative to the running maximum, large finite positive scores do not require exponentiating the raw score itself.

## Validation invariants

Numerical policy never weakens validation silently.

For host-slice execution, every mode preserves:

- non-zero shape validation;
- checked shape/address arithmetic;
- exact Q/K/V length checks;
- finite Q/K/V requirement;
- finite positive softmax scale requirement;
- WGPU dispatch-limit checks for GPU modes.

Resident GPU APIs keep their pre-existing contract: the runtime validates ownership and lengths without downloading resident tensors merely to inspect their values.

## Numerical regression corpus

The permanent M9 corpus contains:

- equal-score rows;
- near-tied scores around zero;
- alternating-sign cancellation;
- large finite scores stressing online-max rescaling;
- causal and non-causal attention;
- default, small and large positive softmax scales;
- non-finite input rejection;
- invalid-scale rejection;
- D64, D80 and D128 deterministic GPU execution;
- sequence lengths crossing K/V and Q-tile boundaries.

The deterministic WGPU gate repeats each selected case multiple times on one context and requires exact O/LSE bit equality before the mode is accepted.

## Relation to M8 precision

M9 controls **accumulation/reduction/execution policy**. M8 controls **storage precision**.

The M9 executor intentionally starts from the qualified f32 WGPU path so deterministic guarantees are not conflated with input quantization. M8 packed-binary16 continues to promote values to FP32 before attention arithmetic and retains its independent precision-specific parity gates.

A future combined precision/numerical dispatcher may compose the two policies only when the guarantees of both layers can be stated without ambiguity.
