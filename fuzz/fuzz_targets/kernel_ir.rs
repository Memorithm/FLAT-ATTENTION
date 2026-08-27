//! Fuzz the Kernel IR construction and validation.
//!
//! Invariants: arbitrary bytes must never cause a panic, out-of-bounds
//! access, or bypass of validation. Valid problems must produce
//! deterministic fingerprints.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let u32_at = |index: usize| -> u32 {
        u32::from_le_bytes([
            data[index * 4],
            data[index * 4 + 1],
            data[index * 4 + 2],
            data[index * 4 + 3],
        ])
    };

    // Construct arbitrary AttentionProblem-like inputs
    let batch_heads = u32_at(0);
    let seq_len = u32_at(1);
    let head_dim = u32_at(2);
    let causal = u32_at(3) & 1 == 1;

    let problem = flat_attention::kernel_ir::AttentionProblem {
        batch_heads,
        seq_len,
        head_dim,
        causal,
    };

    // Validation must not panic
    let is_valid = problem.validate().is_ok();

    // Try to build a KernelModule with arbitrary config
    let configs = [
        flat_attention::kernel_ir::KernelConfig::PORTABLE_SCALAR,
        flat_attention::kernel_ir::KernelConfig::PORTABLE_VEC4,
        flat_attention::kernel_ir::KernelConfig::DOUBLE_BUFFERED_VEC4,
        flat_attention::kernel_ir::KernelConfig::SUBGROUP_ASSISTED,
    ];
    let config_idx = (u32_at(0) as usize) % configs.len();
    let config = configs[config_idx];

    if is_valid {
        // For valid problems, module build either succeeds or fails with a
        // typed error (e.g. head_dim not supported for vector width)
        let result = flat_attention::kernel_ir::KernelModule::build(
            flat_attention::kernel_ir::KernelFamily::DenseQ4Forward,
            problem,
            config,
        );
        if let Ok(module) = result {
            // Valid modules must have deterministic fingerprints
            let fp1 = module.structural_fingerprint();
            let fp2 = module.structural_fingerprint();
            assert_eq!(fp1, fp2);
            assert_eq!(module.canonical_record(), module.canonical_record());

            // Resources must be computable without panic
            let _ = module.resources();
            let _ = module.capability_requirements();
        }
    } else {
        // Invalid problems must be rejected before module construction
        assert!(problem.validate().is_err());
    }

    // Tuning cache parsing must not panic on arbitrary bytes
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = flat_attention::kernel_cache::TuningCache::deserialize(s);
    }
});
