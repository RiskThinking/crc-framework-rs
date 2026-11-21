//! Primary insurance terms, one-peril simulation, and annual loss metrics.

#[cfg(feature = "sampling")]
mod dependence;
mod metrics;
mod policy;
#[cfg(feature = "sampling")]
mod sampling;
#[cfg(feature = "sampling")]
mod simulation;

#[cfg(feature = "sampling")]
pub use dependence::CorrelationRepair;
pub use metrics::{
    AnnualLossExceedance, AnnualLossLevel, AnnualLossSummary, summarize_annual_losses,
};
pub use policy::PolicyTerms;
#[cfg(feature = "sampling")]
pub use sampling::ANNUAL_LOSS_RNG_ALGORITHM;
#[cfg(feature = "sampling")]
pub use simulation::{
    AnnualLossSimulation, AnnualLossSpec, DamageCurve, InsuredInterest, SimulationOptions,
    simulate_annual_losses,
};
