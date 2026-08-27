//! Safe persistent tuning cache for FLAT kernel selection evidence.
//!
//! This module provides an **advisory** file-backed cache for autotuning
//! results. It is not authoritative: every cached entry is re-validated
//! against the current device, IR version, and candidate registry before use.
//! Corruption or staleness invalidates entries; it never crashes the caller
//! or executes arbitrary content.
//!
//! Security properties:
//!
//! - All parsing is bounded (file size, entry count, string lengths).
//! - No `eval`, `serde` with arbitrary type tags, or shell construction.
//! - Atomic writes via temp file + rename prevent torn states.
//! - Fingerprints are cache keys, never authentication.

use crate::fingerprint::fnv1a64;
use std::fmt;
use std::path::{Path, PathBuf};

/// Schema version of the serialized cache format.
pub const CACHE_SCHEMA_VERSION: u32 = 1;

/// Hard limit on serialized cache size (bytes).
pub const MAX_CACHE_BYTES: usize = 1024 * 1024;

/// Hard limit on number of entries in one cache file.
pub const MAX_CACHE_ENTRIES: usize = 256;

/// Maximum length of any single line in the serialized format.
pub const MAX_LINE_BYTES: usize = 4096;

/// Deterministic cache key for one tuning result.
///
/// Combines device identity, problem identity, and code version. A change in
/// any component invalidates the cached result — this is the driver/codegen
/// invalidation required by the roadmap.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TuningCacheKey {
    /// FNV-1a-64 of the device's canonical record.
    pub device_fingerprint: u64,
    /// FNV-1a-64 of the problem's canonical record.
    pub problem_fingerprint: u64,
    /// IR schema version at tuning time.
    pub ir_version: crate::kernel_ir::KernelIrVersion,
    /// Codegen version at tuning time.
    pub codegen_version: crate::kernel_wgsl::CodegenVersion,
}

impl TuningCacheKey {
    /// Deterministic canonical record for the key.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        format!(
            "device={:016x};problem={:016x};ir=v{};cg=v{}",
            self.device_fingerprint,
            self.problem_fingerprint,
            self.ir_version,
            self.codegen_version
        )
    }

    /// Stable fingerprint of the canonical key record.
    #[must_use]
    pub fn stable_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_record().as_bytes())
    }
}

/// One cached tuning result.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheEntry {
    /// Key identifying the tuning context.
    pub key: TuningCacheKey,
    /// Stable candidate identity that was selected.
    pub candidate_id: u64,
    /// Median latency in microseconds at tuning time.
    pub median_us: f64,
    /// P95 latency in microseconds at tuning time.
    pub p95_us: f64,
}

impl CacheEntry {
    fn validate(&self) -> Result<(), CacheError> {
        if !self.median_us.is_finite() || self.median_us < 0.0 {
            return Err(CacheError::InvalidTiming {
                field: "median_us",
                value: self.median_us,
            });
        }
        if !self.p95_us.is_finite() || self.p95_us < 0.0 {
            return Err(CacheError::InvalidTiming {
                field: "p95_us",
                value: self.p95_us,
            });
        }
        if self.p95_us < self.median_us - f64::EPSILON {
            return Err(CacheError::InvalidTiming {
                field: "p95_us < median_us",
                value: self.p95_us,
            });
        }
        Ok(())
    }

    fn canonical_line(&self) -> String {
        format!(
            "entry device={:016x} problem={:016x} ir={}.{} cg={}.{} candidate={:016x} median={:.6} p95={:.6}",
            self.key.device_fingerprint,
            self.key.problem_fingerprint,
            self.key.ir_version.major,
            self.key.ir_version.minor,
            self.key.codegen_version.major,
            self.key.codegen_version.minor,
            self.candidate_id,
            self.median_us,
            self.p95_us
        )
    }
}

/// In-memory tuning cache.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningCache {
    version: u32,
    entries: Vec<CacheEntry>,
}

impl TuningCache {
    /// Create an empty cache at the current schema version.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: CACHE_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert or replace an entry. Returns an error if the cache is full or
    /// the entry is invalid (NaN/Inf/negative timing).
    pub fn insert(&mut self, entry: CacheEntry) -> Result<(), CacheError> {
        entry.validate()?;
        if let Some(pos) = self.entries.iter().position(|e| e.key == entry.key) {
            self.entries[pos] = entry;
            return Ok(());
        }
        if self.entries.len() >= MAX_CACHE_ENTRIES {
            return Err(CacheError::TooManyEntries {
                limit: MAX_CACHE_ENTRIES,
            });
        }
        self.entries.push(entry);
        Ok(())
    }

    /// Look up a cached result by key. Returns `None` if absent or if the
    /// cached candidate is no longer in the registry (caller must check).
    #[must_use]
    pub fn get(&self, key: &TuningCacheKey) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| &e.key == key)
    }

    /// Serialize the cache to a deterministic string.
    pub fn serialize(&self) -> Result<String, CacheError> {
        let mut out = String::new();
        out.push_str(&format!("FLAT-TUNING-CACHE v{}\n", self.version));
        // Deterministic order: sort by key fingerprint.
        let mut sorted = self.entries.clone();
        sorted.sort_by_key(|e| e.key.stable_fingerprint());
        for entry in &sorted {
            out.push_str(&entry.canonical_line());
            out.push('\n');
        }
        if out.len() > MAX_CACHE_BYTES {
            return Err(CacheError::TooLarge {
                bytes: out.len(),
                limit: MAX_CACHE_BYTES,
            });
        }
        Ok(out)
    }

    /// Deserialize a cache, validating every invariant.
    pub fn deserialize(s: &str) -> Result<Self, CacheError> {
        if s.len() > MAX_CACHE_BYTES {
            return Err(CacheError::TooLarge {
                bytes: s.len(),
                limit: MAX_CACHE_BYTES,
            });
        }
        let mut lines = s.lines();
        let header = lines.next().ok_or(CacheError::Truncated)?;
        if header.len() > MAX_LINE_BYTES {
            return Err(CacheError::LineTooLong);
        }
        let version = parse_header(header)?;

        let mut entries = Vec::new();
        let mut seen_keys = std::collections::HashSet::new();
        for line in lines {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.len() > MAX_LINE_BYTES {
                return Err(CacheError::LineTooLong);
            }
            if entries.len() >= MAX_CACHE_ENTRIES {
                return Err(CacheError::TooManyEntries {
                    limit: MAX_CACHE_ENTRIES,
                });
            }
            let entry = parse_entry_line(line)?;
            let key_fp = entry.key.stable_fingerprint();
            if !seen_keys.insert(key_fp) {
                return Err(CacheError::DuplicateEntry { key_fp });
            }
            entries.push(entry);
        }
        Ok(Self { version, entries })
    }

    /// Atomically write the cache to `path` (temp file + rename).
    pub fn save_atomic(&self, path: &Path) -> Result<(), CacheError> {
        let serialized = self.serialize()?;
        let tmp_path = tmp_path_for(path);
        std::fs::write(&tmp_path, &serialized).map_err(|e| CacheError::Io(e.to_string()))?;
        std::fs::rename(&tmp_path, path).map_err(|e| CacheError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load a cache from `path`, returning an empty cache if the file does
    /// not exist (cold cache is not an error).
    pub fn load(path: &Path) -> Result<Self, CacheError> {
        match std::fs::read_to_string(path) {
            Ok(s) => Self::deserialize(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new()),
            Err(e) => Err(CacheError::Io(e.to_string())),
        }
    }
}

impl Default for TuningCache {
    fn default() -> Self {
        Self::new()
    }
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

fn parse_header(line: &str) -> Result<u32, CacheError> {
    let prefix = "FLAT-TUNING-CACHE v";
    let version_str = line.strip_prefix(prefix).ok_or(CacheError::UnknownSchema)?;
    let version: u32 = version_str.parse().map_err(|_| CacheError::UnknownSchema)?;
    if version != CACHE_SCHEMA_VERSION {
        return Err(CacheError::UnsupportedVersion {
            found: version,
            expected: CACHE_SCHEMA_VERSION,
        });
    }
    Ok(version)
}

fn parse_entry_line(line: &str) -> Result<CacheEntry, CacheError> {
    if !line.starts_with("entry ") {
        return Err(CacheError::MalformedLine);
    }
    let rest = &line[6..];
    let mut device_fp = None;
    let mut problem_fp = None;
    let mut ir_major = None;
    let mut ir_minor = None;
    let mut cg_major = None;
    let mut cg_minor = None;
    let mut candidate_id = None;
    let mut median_us = None;
    let mut p95_us = None;

    for part in rest.split_whitespace() {
        let (k, v) = part.split_once('=').ok_or(CacheError::MalformedLine)?;
        match k {
            "device" => {
                if v.len() != 16 {
                    return Err(CacheError::MalformedLine);
                }
                device_fp =
                    Some(u64::from_str_radix(v, 16).map_err(|_| CacheError::MalformedLine)?);
            }
            "problem" => {
                if v.len() != 16 {
                    return Err(CacheError::MalformedLine);
                }
                problem_fp =
                    Some(u64::from_str_radix(v, 16).map_err(|_| CacheError::MalformedLine)?);
            }
            "ir" => {
                let (maj, min) = v.split_once('.').ok_or(CacheError::MalformedLine)?;
                ir_major = Some(maj.parse::<u32>().map_err(|_| CacheError::MalformedLine)?);
                ir_minor = Some(min.parse::<u32>().map_err(|_| CacheError::MalformedLine)?);
            }
            "cg" => {
                let (maj, min) = v.split_once('.').ok_or(CacheError::MalformedLine)?;
                cg_major = Some(maj.parse::<u32>().map_err(|_| CacheError::MalformedLine)?);
                cg_minor = Some(min.parse::<u32>().map_err(|_| CacheError::MalformedLine)?);
            }
            "candidate" => {
                if v.len() != 16 {
                    return Err(CacheError::MalformedLine);
                }
                candidate_id =
                    Some(u64::from_str_radix(v, 16).map_err(|_| CacheError::MalformedLine)?);
            }
            "median" => {
                let val: f64 = v.parse().map_err(|_| CacheError::MalformedLine)?;
                if !val.is_finite() {
                    return Err(CacheError::InvalidTiming {
                        field: "median",
                        value: val,
                    });
                }
                median_us = Some(val);
            }
            "p95" => {
                let val: f64 = v.parse().map_err(|_| CacheError::MalformedLine)?;
                if !val.is_finite() {
                    return Err(CacheError::InvalidTiming {
                        field: "p95",
                        value: val,
                    });
                }
                p95_us = Some(val);
            }
            _ => return Err(CacheError::MalformedLine),
        }
    }

    let entry = CacheEntry {
        key: TuningCacheKey {
            device_fingerprint: device_fp.ok_or(CacheError::MalformedLine)?,
            problem_fingerprint: problem_fp.ok_or(CacheError::MalformedLine)?,
            ir_version: crate::kernel_ir::KernelIrVersion {
                major: ir_major.ok_or(CacheError::MalformedLine)?,
                minor: ir_minor.ok_or(CacheError::MalformedLine)?,
            },
            codegen_version: crate::kernel_wgsl::CodegenVersion {
                major: cg_major.ok_or(CacheError::MalformedLine)?,
                minor: cg_minor.ok_or(CacheError::MalformedLine)?,
            },
        },
        candidate_id: candidate_id.ok_or(CacheError::MalformedLine)?,
        median_us: median_us.ok_or(CacheError::MalformedLine)?,
        p95_us: p95_us.ok_or(CacheError::MalformedLine)?,
    };
    entry.validate()?;
    Ok(entry)
}

/// Errors for cache operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CacheError {
    Truncated,
    UnknownSchema,
    UnsupportedVersion { found: u32, expected: u32 },
    MalformedLine,
    LineTooLong,
    TooLarge { bytes: usize, limit: usize },
    TooManyEntries { limit: usize },
    DuplicateEntry { key_fp: u64 },
    InvalidTiming { field: &'static str, value: f64 },
    Io(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "cache file is truncated"),
            Self::UnknownSchema => write!(f, "unknown cache schema"),
            Self::UnsupportedVersion { found, expected } => {
                write!(f, "unsupported cache version {found}, expected {expected}")
            }
            Self::MalformedLine => write!(f, "malformed cache entry line"),
            Self::LineTooLong => write!(f, "cache line exceeds maximum length"),
            Self::TooLarge { bytes, limit } => {
                write!(f, "cache size {bytes} exceeds limit {limit}")
            }
            Self::TooManyEntries { limit } => {
                write!(f, "too many cache entries (limit {limit})")
            }
            Self::DuplicateEntry { key_fp } => {
                write!(f, "duplicate cache entry for key {key_fp:016x}")
            }
            Self::InvalidTiming { field, value } => {
                write!(f, "invalid timing {field}: {value}")
            }
            Self::Io(msg) => write!(f, "cache I/O error: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::KernelIrVersion;
    use crate::kernel_wgsl::CodegenVersion;

    fn key() -> TuningCacheKey {
        TuningCacheKey {
            device_fingerprint: 0x1234567890abcdef,
            problem_fingerprint: 0xfedcba0987654321,
            ir_version: KernelIrVersion { major: 1, minor: 0 },
            codegen_version: CodegenVersion { major: 1, minor: 0 },
        }
    }

    fn entry() -> CacheEntry {
        CacheEntry {
            key: key(),
            candidate_id: 0xdeadbeefcafebabe,
            median_us: 1234.5,
            p95_us: 1500.0,
        }
    }

    #[test]
    fn round_trip() {
        let mut cache = TuningCache::new();
        cache.insert(entry()).unwrap();
        let serialized = cache.serialize().unwrap();
        let deserialized = TuningCache::deserialize(&serialized).unwrap();
        assert_eq!(cache, deserialized);
    }

    #[test]
    fn schema_mismatch_rejected() {
        let bad = "FLAT-TUNING-CACHE v99\n";
        assert!(matches!(
            TuningCache::deserialize(bad),
            Err(CacheError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn truncated_rejected() {
        assert_eq!(TuningCache::deserialize(""), Err(CacheError::Truncated));
    }

    #[test]
    fn oversized_rejected() {
        let big = "x".repeat(MAX_CACHE_BYTES + 1);
        assert!(matches!(
            TuningCache::deserialize(&big),
            Err(CacheError::TooLarge { .. })
        ));
    }

    #[test]
    fn duplicate_entry_rejected() {
        let mut cache = TuningCache::new();
        cache.insert(entry()).unwrap();
        let mut serialized = cache.serialize().unwrap();
        // Append same entry again
        serialized.push_str(&entry().canonical_line());
        serialized.push('\n');
        assert!(matches!(
            TuningCache::deserialize(&serialized),
            Err(CacheError::DuplicateEntry { .. })
        ));
    }

    #[test]
    fn nan_rejected() {
        let mut e = entry();
        e.median_us = f64::NAN;
        assert!(e.validate().is_err());
        let mut cache = TuningCache::new();
        assert!(cache.insert(e).is_err());
    }

    #[test]
    fn inf_rejected() {
        let mut e = entry();
        e.p95_us = f64::INFINITY;
        assert!(e.validate().is_err());
    }

    #[test]
    fn atomic_write_and_load() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("flat-test-cache-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(tmp_path_for(&path));
        let mut cache = TuningCache::new();
        cache.insert(entry()).unwrap();
        cache.save_atomic(&path).unwrap();
        let loaded = TuningCache::load(&path).unwrap();
        assert_eq!(cache, loaded);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_empty_cache() {
        let path = Path::new("/tmp/flat-cache-nonexistent-12345-xyz");
        let _ = std::fs::remove_file(path);
        let cache = TuningCache::load(path).unwrap();
        assert!(cache.is_empty());
    }

    #[test]
    fn corrupted_fingerprint_rejected() {
        let bad = "FLAT-TUNING-CACHE v1\nentry device=zzzz problem=fedcba0987654321 ir=1.0 cg=1.0 candidate=deadbeefcafebabe median=1.0 p95=2.0\n";
        assert!(TuningCache::deserialize(bad).is_err());
    }

    #[test]
    fn version_sensitivity() {
        let mut cache = TuningCache::new();
        cache.insert(entry()).unwrap();
        let mut different_key = key();
        different_key.ir_version = KernelIrVersion { major: 2, minor: 0 };
        let different_entry = CacheEntry {
            key: different_key,
            candidate_id: 0xdeadbeefcafebabe,
            median_us: 100.0,
            p95_us: 120.0,
        };
        cache.insert(different_entry).unwrap();
        assert_eq!(cache.len(), 2);
        // Different IR version must be a distinct key
        assert!(cache.get(&key()).is_some());
    }
}
