use flat_attention::{single_row_io_model, tiled_q4_io_model, AttentionShape};

fn main() {
    for seq_len in [128usize, 512, 2048] {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len,
            head_dim: 64,
        };
        for causal in [false, true] {
            let baseline = single_row_io_model(shape, causal).expect("valid baseline model");
            let tiled = tiled_q4_io_model(shape, causal).expect("valid Q4 model");
            let ratio = baseline.kv_storage_scalar_loads as f64
                / tiled.kv_storage_scalar_loads as f64;
            println!(
                "N={seq_len:>4} D=64 causal={causal:<5} baseline_wg={:>4} q4_wg={:>4} baseline_kv_loads={:>12} q4_kv_loads={:>12} logical_load_ratio={ratio:.3}x",
                baseline.query_workgroups,
                tiled.query_workgroups,
                baseline.kv_storage_scalar_loads,
                tiled.kv_storage_scalar_loads,
            );
        }
    }
}
