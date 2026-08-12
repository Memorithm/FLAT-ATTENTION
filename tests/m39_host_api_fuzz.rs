use std::panic::{catch_unwind, AssertUnwindSafe};

use flat_attention::api::v1::{
    AttentionConfig, AttentionShape, BorrowedAttentionRequest, ResidentAttentionRequest,
};

#[derive(Clone, Copy)]
struct FuzzStream(u64);

impl FuzzStream {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn dimension(&mut self) -> usize {
        match self.next() & 15 {
            0 => 0,
            1 => 1,
            2 => usize::MAX,
            3 => usize::MAX / 2 + 1,
            4 => 128,
            5 => 129,
            _ => (self.next() as usize % 65) + 1,
        }
    }

    fn scale(&mut self) -> Option<f32> {
        match self.next() & 7 {
            0 => None,
            1 => Some(0.0),
            2 => Some(-1.0),
            3 => Some(f32::NAN),
            4 => Some(f32::INFINITY),
            _ => Some(((self.next() % 10_000) as f32 + 1.0) / 10_000.0),
        }
    }
}

fn fuzz_shape(stream: &mut FuzzStream) -> AttentionShape {
    AttentionShape {
        batch: stream.dimension(),
        q_heads: stream.dimension(),
        kv_heads: stream.dimension(),
        query_len: stream.dimension(),
        kv_len: stream.dimension(),
        head_dim: stream.dimension(),
        query_position_offset: stream.dimension(),
    }
}

#[test]
fn arbitrary_stable_contract_fields_never_panic() {
    let q = [0.0f32, 1.0, -1.0, f32::NAN];
    let k = [0.5f32, -0.5, f32::INFINITY];
    let v = [1.0f32, f32::NEG_INFINITY];
    let resident_token = 7u8;
    let mut stream = FuzzStream(0x6d39_f022_1bad_c0de);

    for case in 0..4_096 {
        let shape = fuzz_shape(&mut stream);
        let config = AttentionConfig {
            causal: stream.next() & 1 == 0,
            softmax_scale: stream.scale(),
        };
        let q_len = stream.next() as usize % (q.len() + 1);
        let k_len = stream.next() as usize % (k.len() + 1);
        let v_len = stream.next() as usize % (v.len() + 1);

        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _ = shape.validate();
            let _ = shape.q_elements();
            let _ = shape.kv_elements();
            let _ = shape.lse_elements();
            let _ = shape.group_size();
            let _ = shape.to_core_shape();
            let _ = config.validate(shape.head_dim);

            let borrowed = BorrowedAttentionRequest {
                shape,
                config,
                q: &q[..q_len],
                k: &k[..k_len],
                v: &v[..v_len],
            };
            let _ = borrowed.validate();
            let _ = borrowed.to_owned();

            let resident = ResidentAttentionRequest {
                shape,
                config,
                q: &resident_token,
                k: &resident_token,
                v: &resident_token,
            };
            let _ = resident.validate_contract();
        }));
        assert!(
            outcome.is_ok(),
            "stable host contract panicked on case {case}"
        );
    }
}

#[test]
fn overflow_position_and_length_fail_closed() {
    let valid = AttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 2,
        kv_len: 3,
        head_dim: 8,
        query_position_offset: 0,
    };
    let q = vec![0.0; valid.q_elements().unwrap()];
    let k = vec![0.0; valid.kv_elements().unwrap()];
    let v = vec![0.0; valid.kv_elements().unwrap()];

    for shape in [
        AttentionShape {
            batch: usize::MAX,
            ..valid
        },
        AttentionShape {
            q_heads: usize::MAX,
            kv_heads: 1,
            ..valid
        },
        AttentionShape {
            query_len: 2,
            query_position_offset: usize::MAX,
            ..valid
        },
    ] {
        assert!(shape.validate().is_err());
    }

    for (q_slice, k_slice, v_slice) in [
        (&q[..q.len() - 1], k.as_slice(), v.as_slice()),
        (q.as_slice(), &k[..k.len() - 1], v.as_slice()),
        (q.as_slice(), k.as_slice(), &v[..v.len() - 1]),
    ] {
        let request = BorrowedAttentionRequest {
            shape: valid,
            config: AttentionConfig::default(),
            q: q_slice,
            k: k_slice,
            v: v_slice,
        };
        assert!(request.validate().is_err());
    }
}

#[test]
fn every_nonfinite_input_class_is_rejected() {
    let shape = AttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        query_len: 1,
        kv_len: 1,
        head_dim: 1,
        query_position_offset: 0,
    };
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for tensor in 0..3 {
            let mut q = [0.0f32];
            let mut k = [0.0f32];
            let mut v = [0.0f32];
            match tensor {
                0 => q[0] = nonfinite,
                1 => k[0] = nonfinite,
                _ => v[0] = nonfinite,
            }
            let request = BorrowedAttentionRequest {
                shape,
                config: AttentionConfig::default(),
                q: &q,
                k: &k,
                v: &v,
            };
            assert!(request.validate().is_err());
        }
    }
}
