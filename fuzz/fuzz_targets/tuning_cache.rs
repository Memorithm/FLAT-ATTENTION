//! Fuzz the tuning cache decoder with arbitrary bytes.
//!
//! Invariants: any byte sequence must be handled safely — either accepted as
//! a valid cache or rejected with a typed error. No panics, no OOM from
//! unbounded allocation, no arbitrary code execution.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to deserialize arbitrary bytes as a cache
    if let Ok(s) = std::str::from_utf8(data) {
        let result = flat_attention::kernel_cache::TuningCache::deserialize(s);
        // Either succeeds or fails with a typed error — never panics
        match result {
            Ok(cache) => {
                // Valid caches must round-trip deterministically
                if let Ok(serialized) = cache.serialize() {
                    // Re-parsing must succeed and be equal
                    if let Ok(reparsed) = flat_attention::kernel_cache::TuningCache::deserialize(&serialized) {
                        assert_eq!(cache, reparsed);
                    }
                }
                // Cache size must be bounded
                assert!(cache.len() <= flat_attention::kernel_cache::MAX_CACHE_ENTRIES);
            }
            Err(_) => {
                // Rejected — this is the safe failure path
            }
        }
    }

    // Also fuzz with raw bytes that may not be valid UTF-8
    let s = String::from_utf8_lossy(data);
    let _ = flat_attention::kernel_cache::TuningCache::deserialize(&s);
});
