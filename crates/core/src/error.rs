use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, CrcError>;

#[derive(Debug, Clone, PartialEq)]
pub enum CrcError {
    InvalidInput(String),
    Unsupported(String),
    MissingParameter(String),
    ConvergenceFailed {
        family: String,
        iterations: usize,
    },
    OutOfSupport {
        probability: f64,
        minimum: f64,
        maximum: f64,
    },
    BranchLimitExceeded {
        factors: usize,
        branches: usize,
        limit: usize,
    },
}

impl Display for CrcError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported operation: {message}"),
            Self::MissingParameter(message) => write!(f, "missing parameter: {message}"),
            Self::ConvergenceFailed { family, iterations } => write!(
                f,
                "{family} fitting did not converge after {iterations} iterations"
            ),
            Self::OutOfSupport {
                probability,
                minimum,
                maximum,
            } => write!(
                f,
                "probability {probability} is outside tabulated support [{minimum}, {maximum}]"
            ),
            Self::BranchLimitExceeded {
                factors,
                branches,
                limit,
            } => write!(
                f,
                "{factors} factors require {branches} spanning branches, above limit {limit}"
            ),
        }
    }
}

impl std::error::Error for CrcError {}
