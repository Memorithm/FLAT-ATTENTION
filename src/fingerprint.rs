//! Shared non-cryptographic fingerprint primitive.
//!
//! FNV-1a-64 is project policy for deterministic structural fingerprints of
//! host-side records (device identity, capability limits, Kernel IR). These
//! values are cache-key/equality accelerators and provenance identifiers;
//! they are never authentication and never replace structural validation.

/// FNV-1a-64 over `bytes`.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_reference_vectors() {
        // Published FNV-1a 64-bit test vectors.
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
    }
}
