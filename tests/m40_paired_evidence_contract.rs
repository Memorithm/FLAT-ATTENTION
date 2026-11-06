//! M40 physical paired-evidence admission. Host-only; no benchmark is run here.
//! This checks provenance and comparability, never certifies measured speedup.
use toml::Value;

fn required<'a>(row: &'a Value, key: &str) -> Result<&'a str, String> {
    row.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("missing {key}"))
}

fn exact_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate(input: &str) -> Result<(), String> {
    let record: Value = toml::from_str(input).map_err(|error| format!("invalid TOML: {error}"))?;
    if record.get("schema").and_then(Value::as_str) != Some("flat.m40.paired.v1") {
        return Err("unknown evidence schema".into());
    }
    for key in ["flat_sha", "scirust_sha"] {
        if !exact_sha(required(&record, key)?) {
            return Err(format!("{key} must be an exact 40-hex commit"));
        }
    }
    for key in [
        "adapter",
        "driver",
        "backend",
        "dtype",
        "geometry",
        "cache_state",
        "measurement_boundary",
        "artifact_url",
    ] {
        required(&record, key)?;
    }
    // An artifact locator is not an attestation: require an explicit HTTPS origin,
    // while leaving existence, immutability and authentication to RemoteOps review.
    let artifact_url = required(&record, "artifact_url")?;
    let Some(authority_and_path) = artifact_url.strip_prefix("https://") else {
        return Err("artifact_url must use HTTPS".into());
    };
    let authority = authority_and_path.split('/').next().unwrap_or("");
    if authority.is_empty()
        || authority.contains('@')
        || authority.contains('?')
        || authority.contains('#')
        || authority.chars().any(char::is_whitespace)
    {
        return Err("artifact_url must have a nonempty HTTPS host without credentials".into());
    }
    let adapter = required(&record, "adapter")?.to_ascii_lowercase();
    let backend = required(&record, "backend")?.to_ascii_lowercase();
    if ["lavapipe", "llvmpipe", "warp", "software", "swiftshader"]
        .iter()
        .any(|marker| adapter.contains(marker) || backend.contains(marker))
    {
        return Err("software adapter is not physical-device evidence".into());
    }
    if record.get("physical_device").and_then(Value::as_bool) != Some(true) {
        return Err("physical device must be positively attested".into());
    }
    if record.get("oracle_parity_passed").and_then(Value::as_bool) != Some(true) {
        return Err("oracle parity must pass before timing".into());
    }
    for key in ["warmups", "repeats"] {
        if record
            .get(key)
            .and_then(Value::as_integer)
            .is_none_or(|n| n < 1)
        {
            return Err(format!("{key} must be positive"));
        }
    }
    let baseline = record.get("baseline").ok_or("missing baseline")?;
    let candidate = record.get("candidate").ok_or("missing candidate")?;
    for (label, row) in [("baseline", baseline), ("candidate", candidate)] {
        for key in ["route", "median_ms", "p95_ms"] {
            if key == "route" {
                required(row, key)?;
            } else if row
                .get(key)
                .and_then(Value::as_float)
                .is_none_or(|n| !n.is_finite() || n <= 0.0)
            {
                return Err(format!("{label}.{key} must be finite and positive"));
            }
        }
        if row["p95_ms"].as_float().unwrap() < row["median_ms"].as_float().unwrap() {
            return Err(format!("{label} p95 cannot be below median"));
        }
    }
    if required(baseline, "route")? == required(candidate, "route")? {
        return Err("baseline and candidate must be distinct routes".into());
    }
    if required(&record, "measurement_boundary")? != "resident_only"
        && required(&record, "measurement_boundary")? != "transfer_inclusive"
    {
        return Err("unknown timing boundary".into());
    }
    Ok(())
}

#[test]
fn paired_physical_record_accepts_only_explicit_evidence() {
    let valid = format!(
        r#"schema = "flat.m40.paired.v1"
flat_sha = "{sha}"
scirust_sha = "{sha}"
adapter = "physical test adapter"
driver = "test driver"
backend = "vulkan"
dtype = "f32"
geometry = "B1 H8 KV2 Q1 K128 D64"
cache_state = "fresh"
measurement_boundary = "resident_only"
artifact_url = "https://example.invalid/immutable-test-artifact"
physical_device = true
oracle_parity_passed = true
warmups = 3
repeats = 10
[baseline]
route = "scirust_multi_dispatch"
median_ms = 2.0
p95_ms = 3.0
[candidate]
route = "flat_resident"
median_ms = 1.5
p95_ms = 2.5
"#,
        sha = "a".repeat(40)
    );
    assert!(
        validate(&valid).is_ok(),
        "valid fixture rejected: {:?}",
        validate(&valid)
    );
    for bad in [
        valid.replace("physical_device = true", "physical_device = false"),
        valid.replace("https://example.invalid/", "http://example.invalid/"),
        valid.replace("https://example.invalid/", "https:///"),
        valid.replace(
            "https://example.invalid/",
            "https://user:pass@example.invalid/",
        ),
        valid.replace("https://example.invalid/", "https://example.invalid?bad/"),
        valid.replace(
            "oracle_parity_passed = true",
            "oracle_parity_passed = false",
        ),
        valid.replace("physical test adapter", "lavapipe"),
        valid.replace("warmups = 3", "warmups = 0"),
        valid.replace("p95_ms = 2.5", "p95_ms = 1.0"),
        valid.replace("flat.m40.paired.v1", "unknown"),
        valid.replace("resident_only", "unscoped"),
        valid.replace(&"a".repeat(40), "unversioned"),
        valid.replace(
            "route = \"flat_resident\"",
            "route = \"scirust_multi_dispatch\"",
        ),
    ] {
        assert!(validate(&bad).is_err(), "invalid evidence was accepted");
    }
}
