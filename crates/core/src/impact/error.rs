use std::fmt::{Display, Formatter};

/// A contract violation identified by a stable code and JSON Pointer path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactContractError {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

impl ImpactContractError {
    pub(crate) fn new(
        code: &'static str,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl Display for ImpactContractError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let path = if self.path.is_empty() {
            "/"
        } else {
            self.path.as_str()
        };
        write!(formatter, "{} at {path}: {}", self.code, self.message)
    }
}

impl std::error::Error for ImpactContractError {}

pub(crate) type ContractResult<T> = Result<T, ImpactContractError>;

pub(crate) fn fail(
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) -> ImpactContractError {
    ImpactContractError::new(code, path, message)
}
