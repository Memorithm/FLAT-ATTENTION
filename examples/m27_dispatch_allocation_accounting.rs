const F32_BYTES: u64 = 4;
const GROUPED_FORWARD_PARAM_BYTES: u64 = 8 * 4;
const RESIDENT_DECODE_PARAM_BYTES: u64 = 12 * 4;

fn grouped_forward_row(q_heads: u64, kv_heads: u64, seq_len: u64, head_dim: u64) {
    let q_bytes = q_heads * seq_len * head_dim * F32_BYTES;
    let kv_bytes_each = kv_heads * seq_len * head_dim * F32_BYTES;
    let output_bytes = q_bytes + q_heads * seq_len * F32_BYTES;
    let resident_storage_bytes = q_bytes + 2 * kv_bytes_each + output_bytes;

    println!(
        "grouped_forward,{q_heads},{kv_heads},{seq_len},{head_dim},{resident_storage_bytes},{GROUPED_FORWARD_PARAM_BYTES},4,1,1,1,0,0"
    );
}

fn resident_decode_row(q_heads: u64, kv_heads: u64, kv_len: u64, head_dim: u64) {
    let q_bytes = q_heads * head_dim * F32_BYTES;
    let cache_bytes_each = kv_heads * kv_len * head_dim * F32_BYTES;
    let output_bytes = (q_heads * head_dim + q_heads) * F32_BYTES;
    let resident_storage_bytes = q_bytes + 2 * cache_bytes_each + output_bytes;

    println!(
        "resident_decode,{q_heads},{kv_heads},{kv_len},{head_dim},{resident_storage_bytes},{RESIDENT_DECODE_PARAM_BYTES},4,1,1,1,0,0"
    );
}

fn main() {
    println!("benchmark=m27_dispatch_allocation_accounting");
    println!("accounting_scope=source_contract_not_allocator_telemetry");
    println!("precision=f32");
    println!("timed_pipeline_creations=0");
    println!("materialized_score_probability_bytes=0");
    println!("performance_claim=none");
    println!(
        "path,q_heads,kv_heads,context_len,head_dim,resident_storage_bytes,timed_uniform_bytes,resident_storage_buffer_count,timed_uniform_buffer_allocations,timed_bind_group_creations,timed_dispatches,timed_pipeline_creations,materialized_score_probability_bytes"
    );

    for &(q_heads, kv_heads) in &[(4_u64, 4_u64), (4, 2), (4, 1)] {
        for &seq_len in &[32_u64, 128, 512] {
            for &head_dim in &[32_u64, 64, 80, 96, 128] {
                grouped_forward_row(q_heads, kv_heads, seq_len, head_dim);
            }
        }
    }

    for &(q_heads, kv_heads) in &[(4_u64, 4_u64), (4, 2), (4, 1)] {
        for &kv_len in &[32_u64, 128, 512, 2048] {
            for &head_dim in &[32_u64, 64, 80, 96, 128] {
                resident_decode_row(q_heads, kv_heads, kv_len, head_dim);
            }
        }
    }
}
