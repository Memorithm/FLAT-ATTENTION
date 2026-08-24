//! Fuzz the paged KV page-table state machine.
//!
//! Invariants: appends bounded by capacity succeed; telemetry accounting is
//! exact (mapped + free == physical pages); addresses are `None` outside the
//! live prefix and stable until reset advances the generation.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let page_size = (data[0] as usize % 8).max(1);
    let physical_pages = (data.len().saturating_sub(1) / 2).max(1);
    let mut table = match flat_attention::paged_kv::PagedKvTable::new(
        flat_attention::paged_kv::PagedKvConfig {
            page_size,
            physical_pages,
        },
    ) {
        Ok(table) => table,
        Err(_) => return,
    };

    for chunk in data[1..].chunks_exact(2) {
        let tokens = (chunk[0] as usize) % (page_size * 3 + 1);
        if table.append(tokens).is_ok() {
            let telemetry = table.telemetry().expect("live table must report");
            assert_eq!(
                telemetry.mapped_pages + telemetry.free_pages,
                physical_pages,
                "page accounting must be exact"
            );
            assert!(telemetry.live_tokens <= telemetry.capacity_tokens);
            let probe = (chunk[1] as usize).min(telemetry.live_tokens.max(1) - 1);
            if telemetry.live_tokens > 0 {
                let address = table.address(probe).expect("in-prefix address exists");
                assert!(address.physical_page < physical_pages);
                assert!(address.offset_in_page < page_size);
            }
        }
    }

    table.reset().expect("generation counter cannot overflow here");
    assert!(table.is_empty());
    assert_eq!(table.address(0), None);
});
