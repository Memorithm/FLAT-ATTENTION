use flat_attention::pvp_prepared::{PvpPreparedArm as Arm, PvpPreparedSpec,
    FLAT_PVP_PACKED_ONE_STAGE_WGSL};

#[test]
fn checked_recipes_cover_fallbacks_and_all_prefix_boundaries() {
    for k in [1_usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 262144] {
        for g in [1_usize, 31, 128, 129, 2048] {
            let n = k.ilog2() as usize;
            for arm in Arm::ALL {
                let spec = PvpPreparedSpec::new(arm, k, g).unwrap();
                let dispatches = match arm {
                    Arm::Vec4 | Arm::PackedOneStage => n,
                    Arm::Fused2 => if n < 2 { n } else { n - 1 },
                    Arm::Tile8 => if n < 3 { n } else { n - 2 },
                    Arm::PackedSevenStage => if n == 0 { 0 } else { 1 + n.saturating_sub(7) },
                };
                assert_eq!(spec.dispatches(), dispatches, "K={k} G={g} arm={arm:?}");
                let vectors = if arm.is_packed() { g * k.div_ceil(128) } else { k * g.div_ceil(128) };
                assert_eq!(spec.storage_u32_words(), vectors * 4);
                assert_eq!(spec.storage_bytes(), (vectors * 16) as u64);
                assert_eq!((spec.arm(), spec.addresses(), spec.gates()), (arm, k, g));
            }
        }
    }
    for arm in Arm::ALL {
        for (k, g) in [(0, 1), (3, 1), (1, 0), (128, usize::MAX)] {
            assert!(PvpPreparedSpec::new(arm, k, g).is_err());
        }
        if usize::BITS > 32 {
            assert!(PvpPreparedSpec::new(arm, 1_usize << 32, 1).is_err());
        }
    }
}

#[test]
fn one_stage_control_wgsl_validates_without_optional_capabilities() {
    let module = naga::front::wgsl::parse_str(FLAT_PVP_PACKED_ONE_STAGE_WGSL).unwrap();
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module).unwrap();
}

#[cfg(feature = "wgpu")]
#[test]
fn capability_rejection_precedes_resource_creation_and_is_arm_specific() {
    use flat_attention::pvp_prepared::PvpPreparedError;
    let limits = wgpu::Limits::downlevel_defaults();
    for arm in Arm::ALL {
        let spec = PvpPreparedSpec::new(arm, 128, 129).unwrap();
        spec.validate_limits(&limits).unwrap();
        let mut low = limits.clone();
        low.max_storage_buffer_binding_size = spec.storage_bytes() - 1;
        assert!(matches!(spec.validate_limits(&low), Err(PvpPreparedError::DeviceLimit { name: "storage binding bytes", .. })));
        low = limits.clone(); low.max_compute_workgroups_per_dimension = 0;
        assert!(matches!(spec.validate_limits(&low), Err(PvpPreparedError::DeviceLimit { name: "workgroups x", .. })));
        low = limits.clone(); low.max_compute_workgroup_storage_size = 127;
        assert_eq!(spec.validate_limits(&low).is_err(), arm == Arm::Tile8);
        // Tile8 below its prefix threshold delegates to vec4, with no shared memory.
        PvpPreparedSpec::new(arm, 4, 1).unwrap().validate_limits(&low).unwrap();
    }
}
