//! Canonical evaluation records for point, expectation, and distribution transforms.
//!
//! Integer metadata is emitted as JSON integers. Binary64 values use the same
//! shortest-round-trip spelling as definition canonical JSON, with negative
//! zero normalized to `0.0`.

use sha2::{Digest, Sha256};

use super::error::{ContractResult, fail};
use super::expectation::{Expectation, ResolvedMethod};
use super::pushforward::ALGORITHM;
use crate::CurveParameters;

const POINT_KERNEL: &str = "crc-impact/1";
const ARRAY_ENCODING: &str = "crc-array-f64le/1";

/// Retained or caller-preserved snapshot. Hashing alone does not retain data.
#[derive(Clone, Debug)]
pub struct SnapshotRef {
    pub uri: String,
    pub revision: Option<String>,
    pub content_sha256: String,
    pub retention: String,
}

impl SnapshotRef {
    pub fn new(
        uri: String,
        revision: Option<String>,
        content_sha256: String,
        retention: String,
    ) -> ContractResult<Self> {
        if uri.trim().is_empty() || uri != uri.trim() {
            return Err(fail(
                "invalid_value",
                "/source/snapshot/uri",
                "snapshot uri must be a non-empty identifier",
            ));
        }
        if !is_sha256(&content_sha256) {
            return Err(fail(
                "invalid_value",
                "/source/snapshot/content_sha256",
                "snapshot digest must be 64 lowercase hexadecimal characters",
            ));
        }
        if retention != "retained" && retention != "external" {
            return Err(fail(
                "unsupported_value",
                "/source/snapshot/retention",
                "snapshot retention must be retained or external",
            ));
        }
        if revision
            .as_ref()
            .is_some_and(|value| value.trim().is_empty() || value != value.trim())
        {
            return Err(fail(
                "invalid_value",
                "/source/snapshot/revision",
                "snapshot revision must be a non-empty identifier",
            ));
        }
        Ok(Self {
            uri,
            revision,
            content_sha256,
            retention,
        })
    }
}

/// Source identity stored on an evaluation record.
#[derive(Clone, Debug)]
pub enum RecordedSource {
    PointArray {
        shape: Vec<u64>,
        digest: Option<String>,
    },
    Histogram {
        interval_count: u64,
        original_mass_sum: f64,
        normalization_factor: f64,
    },
    Canonical(CurveParameters),
    /// Discrete and empirical laws have no record-schema tag yet.
    Untracked { description: String },
}

/// Canonical JSON of one impact evaluation record.
#[derive(Clone, Debug)]
pub struct ImpactEvaluationRecord {
    canonical: String,
}

impl ImpactEvaluationRecord {
    pub fn canonical_json(&self) -> &str {
        &self.canonical
    }

    pub fn point(
        definition_hash: &str,
        source: RecordedSource,
        snapshot: Option<&SnapshotRef>,
        framework_version: &str,
        target: &str,
    ) -> Self {
        Self::build(
            definition_hash,
            Canon::object([
                ("kind", Canon::string("point")),
                (
                    "settings",
                    Canon::object([
                        ("alignment", Canon::string("event_aligned")),
                        ("shape", Canon::string("preserve")),
                    ]),
                ),
            ]),
            source,
            snapshot,
            diagnostics("direct", None, None, Vec::new()),
            numerical_methods(&[("point", POINT_KERNEL)]),
            framework_version,
            target,
        )
    }

    pub fn expectation(
        definition_hash: &str,
        source: RecordedSource,
        snapshot: Option<&SnapshotRef>,
        result: &Expectation,
        framework_version: &str,
        target: &str,
    ) -> Self {
        let (exactness, components) = expectation_components(result);
        let error = match result.method {
            ResolvedMethod::Exact { .. } => None,
            ResolvedMethod::GaussKronrod { .. } => Some(result.error_estimate),
        };
        let mut methods = vec![("point", POINT_KERNEL)];
        let version = match result.method {
            ResolvedMethod::Exact { version } => version,
            ResolvedMethod::GaussKronrod { version, .. } => version,
        };
        methods.push((result.method.kind(), version));
        Self::build(
            definition_hash,
            Canon::object([
                ("kind", Canon::string("expected_value")),
                (
                    "settings",
                    Canon::object([
                        ("requested_method", Canon::string(requested(result))),
                        ("method", method_json(&result.method)),
                    ]),
                ),
            ]),
            source,
            snapshot,
            diagnostics(
                exactness,
                error,
                Some(result.evaluations),
                components,
            ),
            numerical_methods(&methods),
            framework_version,
            target,
        )
    }

    pub fn transform(
        definition_hash: &str,
        source: RecordedSource,
        snapshot: Option<&SnapshotRef>,
        framework_version: &str,
        target: &str,
    ) -> Self {
        Self::build(
            definition_hash,
            Canon::object([
                ("kind", Canon::string("transform_distribution")),
                (
                    "settings",
                    Canon::object([("algorithm", Canon::string(ALGORITHM))]),
                ),
            ]),
            source,
            snapshot,
            diagnostics("exact", None, None, Vec::new()),
            numerical_methods(&[("point", POINT_KERNEL), ("pushforward", ALGORITHM)]),
            framework_version,
            target,
        )
    }

    fn build(
        definition_hash: &str,
        operation: Canon,
        source: RecordedSource,
        snapshot: Option<&SnapshotRef>,
        diagnostics: Canon,
        methods: Canon,
        framework_version: &str,
        target: &str,
    ) -> Self {
        // Untracked sources (discrete, empirical) have no `snapshot` member in
        // their schema variant, so a caller-supplied snapshot never actually
        // lands in the record for them. Replay completeness must track what
        // the source object can carry, not merely whether a snapshot argument
        // was passed.
        let replay_complete =
            snapshot.is_some() && !matches!(&source, RecordedSource::Untracked { .. });
        let record = Canon::object([
            ("schema_version", Canon::string("1.0")),
            (
                "function",
                Canon::object([
                    ("kind", Canon::string("definition")),
                    ("definition_hash", Canon::string(definition_hash)),
                ]),
            ),
            ("operation", operation),
            (
                "engine",
                Canon::object([
                    ("framework_version", Canon::string(framework_version)),
                    ("source_revision", Canon::Null),
                    ("build_id", Canon::Null),
                    ("target", Canon::string(target)),
                    ("numerical_methods", methods),
                ]),
            ),
            ("source", source_json(source, snapshot)),
            ("preprocessing", Canon::Array(Vec::new())),
            ("resolution", Canon::Null),
            ("diagnostics", diagnostics),
            ("replay", replay(replay_complete)),
        ]);
        Self {
            canonical: record.encode(),
        }
    }
}

/// SHA-256 of a C-order little-endian float64 array, with negative zero normalized.
pub fn array_content_sha256(shape: &[u64], values: &[f64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(ARRAY_ENCODING.as_bytes());
    hasher.update([0]);
    hasher.update((shape.len() as u64).to_le_bytes());
    for dimension in shape {
        hasher.update(dimension.to_le_bytes());
    }
    for value in values {
        let bits = if *value == 0.0 { 0 } else { value.to_bits() };
        hasher.update(bits.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn requested(result: &Expectation) -> &'static str {
    match result.requested {
        super::expectation::RequestedMethod::Auto => "auto",
        super::expectation::RequestedMethod::Exact => "exact",
        super::expectation::RequestedMethod::GaussKronrod => "gauss_kronrod",
    }
}

fn method_json(method: &ResolvedMethod) -> Canon {
    match method {
        ResolvedMethod::Exact { version } => Canon::object([
            ("kind", Canon::string("exact")),
            ("version", Canon::string(*version)),
        ]),
        ResolvedMethod::GaussKronrod {
            version,
            absolute_tolerance,
            relative_tolerance,
            max_evaluations,
        } => Canon::object([
            ("kind", Canon::string("gauss_kronrod")),
            ("version", Canon::string(*version)),
            ("absolute_tolerance", Canon::Float(*absolute_tolerance)),
            ("relative_tolerance", Canon::Float(*relative_tolerance)),
            ("max_evaluations", Canon::Int(*max_evaluations)),
        ]),
    }
}

fn expectation_components(result: &Expectation) -> (&'static str, Vec<Canon>) {
    let exact = result.exact_component_present;
    let numerical = result.numerical_component_present;
    match (exact, numerical) {
        (true, true) => (
            "mixed",
            vec![
                component("exact", result.exact_component),
                component("numerical", result.numerical_component),
            ],
        ),
        (false, true) => (
            "numerical",
            vec![component("numerical", result.numerical_component)],
        ),
        _ => ("exact", vec![component("exact", result.exact_component)]),
    }
}

fn component(kind: &'static str, value: f64) -> Canon {
    Canon::object([
        ("kind", Canon::string(kind)),
        ("value", Canon::Float(value)),
    ])
}

fn diagnostics(
    exactness: &'static str,
    estimated_error: Option<f64>,
    evaluations: Option<u64>,
    components: Vec<Canon>,
) -> Canon {
    Canon::object([
        ("exactness", Canon::string(exactness)),
        (
            "estimated_error",
            estimated_error.map(Canon::Float).unwrap_or(Canon::Null),
        ),
        (
            "evaluations",
            evaluations.map(Canon::Int).unwrap_or(Canon::Null),
        ),
        ("components", Canon::Array(components)),
    ])
}

fn numerical_methods(entries: &[(&str, &str)]) -> Canon {
    Canon::Object(
        entries
            .iter()
            .map(|(key, value)| ((*key).to_string(), Canon::string(*value)))
            .collect(),
    )
}

fn replay(complete: bool) -> Canon {
    if complete {
        Canon::object([
            ("status", Canon::string("complete")),
            ("missing_requirements", Canon::Array(Vec::new())),
        ])
    } else {
        Canon::object([
            ("status", Canon::string("missing_source_snapshot")),
            (
                "missing_requirements",
                Canon::Array(vec![Canon::string("source_snapshot")]),
            ),
        ])
    }
}

fn source_json(source: RecordedSource, snapshot: Option<&SnapshotRef>) -> Canon {
    let snapshot_json = snapshot.map(snapshot_json).unwrap_or(Canon::Null);
    match source {
        RecordedSource::PointArray { shape, digest } => Canon::object([
            ("kind", Canon::string("point_array")),
            (
                "shape",
                Canon::Array(shape.into_iter().map(Canon::Int).collect()),
            ),
            ("content_digest", digest_json(ARRAY_ENCODING, digest)),
            ("snapshot", snapshot_json),
            ("selection", Canon::Null),
        ]),
        RecordedSource::Histogram {
            interval_count,
            original_mass_sum,
            normalization_factor,
        } => Canon::object([
            ("kind", Canon::string("histogram")),
            ("interval_count", Canon::Int(interval_count)),
            ("within_bin", Canon::string("uniform")),
            ("original_mass_sum", Canon::Float(original_mass_sum)),
            (
                "normalization_factor",
                Canon::Float(normalization_factor),
            ),
            ("content_digest", Canon::Null),
            ("snapshot", snapshot_json),
            ("selection", Canon::Null),
        ]),
        RecordedSource::Canonical(parameters) => Canon::object([
            ("kind", Canon::string("canonical_distribution")),
            ("parameters", curve_json(&parameters)),
            ("content_digest", Canon::Null),
            ("snapshot", snapshot_json),
            ("selection", Canon::Null),
        ]),
        RecordedSource::Untracked { description } => Canon::object([
            ("kind", Canon::string("untracked")),
            ("description", Canon::string(description)),
        ]),
    }
}

fn digest_json(encoding: &str, digest: Option<String>) -> Canon {
    match digest {
        Some(sha256) => Canon::object([
            ("encoding", Canon::string(encoding)),
            ("sha256", Canon::string(sha256)),
        ]),
        None => Canon::Null,
    }
}

fn snapshot_json(snapshot: &SnapshotRef) -> Canon {
    Canon::object([
        ("uri", Canon::string(&snapshot.uri)),
        (
            "revision",
            snapshot
                .revision
                .as_ref()
                .map(|value| Canon::string(value))
                .unwrap_or(Canon::Null),
        ),
        ("content_sha256", Canon::string(&snapshot.content_sha256)),
        ("retention", Canon::string(&snapshot.retention)),
    ])
}

fn curve_json(parameters: &CurveParameters) -> Canon {
    Canon::object([
        ("curve_kind", Canon::string(&parameters.curve_kind)),
        ("curve_type", Canon::string(&parameters.curve_type)),
        ("curve_shape", optional_float(parameters.curve_shape)),
        ("curve_location", optional_float(parameters.curve_location)),
        ("curve_scale", optional_float(parameters.curve_scale)),
        (
            "curve_atom_probability",
            optional_float(parameters.curve_atom_probability),
        ),
        (
            "curve_atom_location",
            optional_float(parameters.curve_atom_location),
        ),
        (
            "curve_probabilities",
            optional_floats(parameters.curve_probabilities.as_deref()),
        ),
        (
            "curve_values",
            optional_floats(parameters.curve_values.as_deref()),
        ),
    ])
}

fn optional_float(value: Option<f64>) -> Canon {
    value.map(Canon::Float).unwrap_or(Canon::Null)
}

fn optional_floats(values: Option<&[f64]>) -> Canon {
    match values {
        Some(values) => Canon::Array(values.iter().copied().map(Canon::Float).collect()),
        None => Canon::Null,
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

enum Canon {
    Null,
    Int(u64),
    Float(f64),
    Str(String),
    Array(Vec<Canon>),
    Object(Vec<(String, Canon)>),
}

impl Canon {
    fn string(value: impl Into<String>) -> Self {
        Self::Str(value.into())
    }

    fn object<const N: usize>(entries: [(&'static str, Canon); N]) -> Self {
        Self::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    fn encode(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Int(value) => value.to_string(),
            Self::Float(value) => {
                if *value == 0.0 {
                    "0.0".to_string()
                } else {
                    serde_json::to_string(value).expect("finite binary64 has a JSON spelling")
                }
            }
            Self::Str(value) => serde_json::to_string(value).expect("strings are valid JSON"),
            Self::Array(items) => {
                let encoded = items.iter().map(Self::encode).collect::<Vec<_>>().join(",");
                format!("[{encoded}]")
            }
            Self::Object(entries) => {
                let mut order = (0..entries.len()).collect::<Vec<_>>();
                order.sort_by(|&left, &right| entries[left].0.as_bytes().cmp(entries[right].0.as_bytes()));
                let encoded = order
                    .into_iter()
                    .map(|index| {
                        let (key, value) = &entries[index];
                        format!(
                            "{}:{}",
                            serde_json::to_string(key).expect("keys are strings"),
                            value.encode()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{{encoded}}}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::expectation::{Expectation, RequestedMethod, ResolvedMethod};

    fn snapshot() -> SnapshotRef {
        SnapshotRef::new(
            "s3://bucket/key".to_string(),
            None,
            "0".repeat(64),
            "retained".to_string(),
        )
        .expect("valid snapshot")
    }

    #[test]
    fn snapshot_revision_rejects_surrounding_whitespace() {
        let digest = "0".repeat(64);
        let error = SnapshotRef::new(
            "s3://bucket/key".to_string(),
            Some(" v1 ".to_string()),
            digest.clone(),
            "retained".to_string(),
        )
        .unwrap_err();
        assert_eq!(error.code, "invalid_value");
        assert_eq!(error.path, "/source/snapshot/revision");

        let accepted = SnapshotRef::new(
            "s3://bucket/key".to_string(),
            Some("v 1".to_string()),
            digest,
            "retained".to_string(),
        )
        .expect("interior whitespace is part of the identifier");
        assert_eq!(accepted.revision.as_deref(), Some("v 1"));
    }

    #[test]
    fn mixed_expectation_keeps_a_zero_exact_or_numerical_component() {
        // A peeled atom whose response is exactly zero still ran: dropping it
        // would make diagnostics disagree with the methods that actually executed.
        let zero_exact = Expectation {
            value: 3.0,
            requested: RequestedMethod::Auto,
            method: ResolvedMethod::GaussKronrod {
                version: "crc-gk15/1",
                absolute_tolerance: 1e-10,
                relative_tolerance: 1e-8,
                max_evaluations: 100_000,
            },
            evaluations: 42,
            error_estimate: 1e-12,
            exact_component: 0.0,
            numerical_component: 3.0,
            exact_component_present: true,
            numerical_component_present: true,
            original_mass_sum: None,
            normalization_factor: None,
        };
        let (exactness, components) = expectation_components(&zero_exact);
        assert_eq!(exactness, "mixed");
        assert_eq!(components.len(), 2);

        // A quadrature integral that happens to evaluate to zero still ran:
        // the numerical component must not be dropped.
        let zero_numerical = Expectation {
            numerical_component: 0.0,
            ..zero_exact
        };
        let (exactness, components) = expectation_components(&zero_numerical);
        assert_eq!(exactness, "mixed");
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn untracked_source_never_claims_complete_replay() {
        let record = ImpactEvaluationRecord::point(
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            RecordedSource::Untracked {
                description: "discrete distribution".to_string(),
            },
            Some(&snapshot()),
            "0.5.0",
            "x86_64-unknown-linux-gnu",
        );
        let json = record.canonical_json();
        // The untracked source schema has no snapshot member, so a passed-in
        // snapshot cannot land in the record; replay must not claim completeness.
        assert!(!json.contains("\"snapshot\""));
        assert!(json.contains("\"status\":\"missing_source_snapshot\""));
    }

    #[test]
    fn point_records_spell_shape_as_integers_and_ignore_signed_zero() {
        let digest = array_content_sha256(&[2], &[0.0, -0.0]);
        let negative = array_content_sha256(&[2], &[-0.0, 0.0]);
        assert_eq!(digest, negative);
        assert_eq!(digest.len(), 64);
        let record = ImpactEvaluationRecord::point(
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            RecordedSource::PointArray {
                shape: vec![2],
                digest: Some(digest),
            },
            None,
            "0.4.2",
            "x86_64-unknown-linux-gnu",
        );
        let json = record.canonical_json();
        assert!(json.contains("\"shape\":[2]"));
        assert!(json.contains("\"status\":\"missing_source_snapshot\""));
        assert!(!json.contains("[2.0]"));
    }
}
