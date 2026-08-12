//! Machine-readable, deterministic benchmark provenance records.
//!
//! M40 deliberately keeps this schema dependency-free. It records benchmark
//! provenance and integer-valued measurements without performing timing itself.

use core::fmt;
use std::fmt::Write as _;

/// Version of the machine-readable benchmark manifest contract.
pub const BENCHMARK_MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Host/device identity required to interpret one benchmark result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkEnvironment {
    pub device: String,
    pub backend: String,
    pub driver: String,
    pub os: String,
    pub arch: String,
}

/// Logical attention workload recorded alongside a benchmark result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkProblem {
    pub precision: String,
    pub batch: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub causal: bool,
}

/// Integer-valued result fields avoid locale/float serialization ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkResult {
    pub median_latency_ns: u64,
    pub p95_latency_ns: u64,
    /// Effective throughput in thousandths of a token per second.
    pub tokens_per_second_milli: u64,
}

/// Reproducible benchmark record tied to one exact source revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkManifest {
    pub commit_sha: String,
    pub benchmark_id: String,
    pub command: String,
    pub environment: BenchmarkEnvironment,
    pub problem: BenchmarkProblem,
    pub warmup_iterations: u32,
    pub measured_iterations: u32,
    pub result: BenchmarkResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkManifestError {
    InvalidCommitSha,
    EmptyField(&'static str),
    ZeroDimension(&'static str),
    InvalidHeadGrouping,
    ZeroMeasuredIterations,
    InvalidPercentileOrdering,
}

impl fmt::Display for BenchmarkManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCommitSha => write!(f, "commit_sha must contain exactly 40 hex digits"),
            Self::EmptyField(field) => write!(f, "benchmark manifest field {field} must not be empty"),
            Self::ZeroDimension(field) => write!(f, "benchmark problem dimension {field} must be non-zero"),
            Self::InvalidHeadGrouping => write!(f, "q_heads must be exactly divisible by kv_heads"),
            Self::ZeroMeasuredIterations => write!(f, "measured_iterations must be non-zero"),
            Self::InvalidPercentileOrdering => {
                write!(f, "p95 latency must be greater than or equal to median latency")
            }
        }
    }
}

impl std::error::Error for BenchmarkManifestError {}

impl BenchmarkManifest {
    /// Validate provenance and workload invariants before a record is promoted.
    pub fn validate(&self) -> Result<(), BenchmarkManifestError> {
        if self.commit_sha.len() != 40 || !self.commit_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(BenchmarkManifestError::InvalidCommitSha);
        }
        for (name, value) in [
            ("benchmark_id", self.benchmark_id.as_str()),
            ("command", self.command.as_str()),
            ("environment.device", self.environment.device.as_str()),
            ("environment.backend", self.environment.backend.as_str()),
            ("environment.driver", self.environment.driver.as_str()),
            ("environment.os", self.environment.os.as_str()),
            ("environment.arch", self.environment.arch.as_str()),
            ("problem.precision", self.problem.precision.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(BenchmarkManifestError::EmptyField(name));
            }
        }
        for (name, value) in [
            ("batch", self.problem.batch),
            ("q_heads", self.problem.q_heads),
            ("kv_heads", self.problem.kv_heads),
            ("query_len", self.problem.query_len),
            ("kv_len", self.problem.kv_len),
            ("head_dim", self.problem.head_dim),
        ] {
            if value == 0 {
                return Err(BenchmarkManifestError::ZeroDimension(name));
            }
        }
        if !self.problem.q_heads.is_multiple_of(self.problem.kv_heads) {
            return Err(BenchmarkManifestError::InvalidHeadGrouping);
        }
        if self.measured_iterations == 0 {
            return Err(BenchmarkManifestError::ZeroMeasuredIterations);
        }
        if self.result.p95_latency_ns < self.result.median_latency_ns {
            return Err(BenchmarkManifestError::InvalidPercentileOrdering);
        }
        Ok(())
    }

    /// Canonical JSON payload. Field order is part of schema version 1.
    pub fn canonical_json(&self) -> Result<String, BenchmarkManifestError> {
        self.validate()?;
        let result_checksum = self.result_checksum();
        let mut out = String::with_capacity(768);
        write!(
            out,
            "{{\"schema_version\":{},\"commit_sha\":{},\"benchmark_id\":{},\"command\":{},",
            BENCHMARK_MANIFEST_SCHEMA_VERSION,
            json_string(&self.commit_sha),
            json_string(&self.benchmark_id),
            json_string(&self.command),
        )
        .expect("writing to String cannot fail");
        write!(
            out,
            "\"environment\":{{\"device\":{},\"backend\":{},\"driver\":{},\"os\":{},\"arch\":{}}},",
            json_string(&self.environment.device),
            json_string(&self.environment.backend),
            json_string(&self.environment.driver),
            json_string(&self.environment.os),
            json_string(&self.environment.arch),
        )
        .expect("writing to String cannot fail");
        write!(
            out,
            "\"problem\":{{\"precision\":{},\"batch\":{},\"q_heads\":{},\"kv_heads\":{},\"query_len\":{},\"kv_len\":{},\"head_dim\":{},\"causal\":{}}},",
            json_string(&self.problem.precision),
            self.problem.batch,
            self.problem.q_heads,
            self.problem.kv_heads,
            self.problem.query_len,
            self.problem.kv_len,
            self.problem.head_dim,
            self.problem.causal,
        )
        .expect("writing to String cannot fail");
        write!(
            out,
            "\"protocol\":{{\"warmup_iterations\":{},\"measured_iterations\":{}}},",
            self.warmup_iterations, self.measured_iterations,
        )
        .expect("writing to String cannot fail");
        write!(
            out,
            "\"result\":{},\"result_checksum\":{{\"algorithm\":\"fnv1a64\",\"value\":\"{}\"}}}}",
            result_json(self.result), result_checksum,
        )
        .expect("writing to String cannot fail");
        Ok(out)
    }

    /// Deterministic integrity checksum over the canonical result object.
    ///
    /// FNV-1a is used only for accidental-corruption/reproducibility detection;
    /// it is not a cryptographic authenticity primitive.
    #[must_use]
    pub fn result_checksum(&self) -> String {
        let hash = fnv1a64(result_json(self.result).as_bytes());
        format!("{hash:016x}")
    }
}

fn result_json(result: BenchmarkResult) -> String {
    format!(
        "{{\"median_latency_ns\":{},\"p95_latency_ns\":{},\"tokens_per_second_milli\":{}}}",
        result.median_latency_ns, result.p95_latency_ns, result.tokens_per_second_milli
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character <= '\u{1f}' => {
                write!(out, "\\u{:04x}", character as u32).expect("writing to String cannot fail");
            }
            character => out.push(character),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> BenchmarkManifest {
        BenchmarkManifest {
            commit_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            benchmark_id: "m40-test".into(),
            command: "cargo run --release --example bench -- --case \\\"gqa\\\"".into(),
            environment: BenchmarkEnvironment {
                device: "Device \"A\"".into(),
                backend: "Vulkan".into(),
                driver: "driver-1".into(),
                os: "Linux".into(),
                arch: "aarch64".into(),
            },
            problem: BenchmarkProblem {
                precision: "f32".into(),
                batch: 1,
                q_heads: 8,
                kv_heads: 2,
                query_len: 128,
                kv_len: 128,
                head_dim: 64,
                causal: true,
            },
            warmup_iterations: 3,
            measured_iterations: 11,
            result: BenchmarkResult {
                median_latency_ns: 1_234_567,
                p95_latency_ns: 1_456_789,
                tokens_per_second_milli: 103_680_000,
            },
        }
    }

    #[test]
    fn canonical_record_is_bit_reproducible() {
        let first = manifest();
        let second = first.clone();
        assert_eq!(first.canonical_json().unwrap(), second.canonical_json().unwrap());
        assert_eq!(first.result_checksum(), second.result_checksum());
        assert!(first.canonical_json().unwrap().contains("Device \\\"A\\\""));
    }

    #[test]
    fn result_change_invalidates_checksum() {
        let first = manifest();
        let mut second = first.clone();
        second.result.median_latency_ns += 1;
        assert_ne!(first.result_checksum(), second.result_checksum());
    }

    #[test]
    fn malformed_provenance_fails_closed() {
        let mut record = manifest();
        record.commit_sha = "main".into();
        assert_eq!(record.validate(), Err(BenchmarkManifestError::InvalidCommitSha));

        let mut record = manifest();
        record.measured_iterations = 0;
        assert_eq!(
            record.validate(),
            Err(BenchmarkManifestError::ZeroMeasuredIterations)
        );

        let mut record = manifest();
        record.problem.kv_heads = 3;
        assert_eq!(record.validate(), Err(BenchmarkManifestError::InvalidHeadGrouping));
    }
}
