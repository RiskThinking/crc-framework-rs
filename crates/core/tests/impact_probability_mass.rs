//! Exact impact composition must retain atoms and arbitrarily small tails.
use crc_framework_core::{DiscreteDistribution, HistogramExposure, ImpactDefinition, ImpactSource};
use serde_json::json;

fn impact(model: serde_json::Value, clamp: serde_json::Value) -> ImpactDefinition {
    ImpactDefinition::from_json(&json!({
        "schema_version": "1.0", "semantics_version": "crc-impact/1",
        "calibration": {"exposure_variable":"x","exposure_unit":"1","impact_metric":"y","impact_unit":"1"},
        "model": model, "clamp": clamp,
        "provenance": {"origin":{"kind":"user_defined"}}
    }).to_string()).unwrap()
}

#[test]
fn finite_source_supports_non_monotone_image_exactly() {
    let function = impact(
        json!({"kind":"tabulated","points":[{"x":0.0,"y":0.0},{"x":1.0,"y":2.0},{"x":2.0,"y":0.0}]}),
        serde_json::Value::Null,
    );
    let source = ImpactSource::Discrete(
        DiscreteDistribution::new(vec![0.0, 1.0, 2.0], vec![0.2, 0.3, 0.5]).unwrap(),
    );
    let image = function.transform_distribution(&source).unwrap();
    assert_eq!(image.atoms().len(), 2);
    assert_eq!(image.atoms()[0].value, 0.0);
    assert_eq!(image.atoms()[0].probability, 0.7);
    assert_eq!(image.quantile(0.7).unwrap(), 0.0);
    assert_eq!(image.quantile(0.8).unwrap(), 2.0);
}

#[test]
fn positive_sub_epsilon_atoms_are_retained() {
    let function = impact(
        json!({"kind":"linear","slope":1.0,"intercept":0.0}),
        serde_json::Value::Null,
    );
    let source = ImpactSource::Discrete(
        DiscreteDistribution::new(vec![0.0, 1e16], vec![1.0 - 1e-16, 1e-16]).unwrap(),
    );
    let image = function.transform_distribution(&source).unwrap();
    assert_eq!(image.atoms().len(), 2);
    assert_eq!(image.atoms()[1].probability, 1e-16);
    assert_eq!(image.quantile(1.0).unwrap(), 1e16);
}

#[test]
fn almost_all_flat_mass_does_not_delete_the_remaining_continuous_tail() {
    let floor = 1.0 - 5e-13;
    let function = impact(
        json!({"kind":"linear","slope":1.0,"intercept":0.0}),
        json!({"minimum":floor}),
    );
    let source =
        ImpactSource::Histogram(HistogramExposure::new(vec![0.0, 1.0], vec![1.0]).unwrap());
    let image = function.transform_distribution(&source).unwrap();
    assert_eq!(image.atoms().len(), 1);
    assert_eq!(image.atoms()[0].probability, floor);
    assert_eq!(image.quantile(1.0).unwrap(), 1.0);
    assert!(image.quantile(1.0 - 1e-13).unwrap() > floor);
}

#[test]
fn non_monotone_bins_have_an_exact_discrete_image() {
    let function = impact(
        json!({"kind":"binned","bin_edges":[0.0,1.0,2.0,3.0],"values":[4.0,1.0,3.0],"outside_domain":"error"}),
        serde_json::Value::Null,
    );
    let source =
        ImpactSource::Histogram(HistogramExposure::new(vec![0.0, 3.0], vec![1.0]).unwrap());
    let image = function.transform_distribution(&source).unwrap();
    assert_eq!(image.atoms().len(), 3);
    assert_eq!(image.quantile(0.1).unwrap(), 1.0);
    assert_eq!(image.quantile(0.5).unwrap(), 3.0);
    assert_eq!(image.quantile(0.9).unwrap(), 4.0);
}

#[test]
fn inverse_survival_preserves_deep_parametric_tail_coordinates() {
    use crc_framework_core::{DistributionFamily, FittedDistribution};
    let function = impact(
        json!({"kind":"linear","slope":1.0,"intercept":0.0}),
        serde_json::Value::Null,
    );
    let source = ImpactSource::Fitted(
        FittedDistribution::from_parameters(DistributionFamily::GumbelRight, None, 0.0, 1.0)
            .unwrap(),
    );
    let image = function.transform_distribution(&source).unwrap();
    let survival = 1e-20_f64;
    assert!((image.inverse_survival(survival).unwrap() + survival.ln()).abs() < 1e-12);
    let tail_mean = image.tail_integral(0.0, survival, 0.0, false).unwrap() / survival;
    assert!((tail_mean - (-survival.ln() + 1.0)).abs() < 1e-6);
}

#[test]
fn inverse_survival_and_integral_keep_sub_ulp_atoms() {
    let function = impact(
        json!({"kind":"linear","slope":1.0,"intercept":0.0}),
        serde_json::Value::Null,
    );
    let source = ImpactSource::Discrete(
        DiscreteDistribution::new(vec![0.0, 1e20], vec![1.0, 1e-20]).unwrap(),
    );
    let image = function.transform_distribution(&source).unwrap();
    assert_eq!(image.inverse_survival(0.5e-20).unwrap(), 1e20);
    assert_eq!(image.tail_integral(0.0, 1e-20, 0.0, false).unwrap(), 1.0);
}

#[test]
fn zero_mass_histogram_end_bins_do_not_extend_support() {
    let function = impact(
        json!({"kind":"linear","slope":1.0,"intercept":0.0}),
        serde_json::Value::Null,
    );
    let source = ImpactSource::Histogram(
        HistogramExposure::new(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 0.0]).unwrap(),
    );
    let image = function.transform_distribution(&source).unwrap();
    assert_eq!(image.quantile(0.0).unwrap(), 1.0);
    assert_eq!(image.quantile(1.0).unwrap(), 2.0);
    assert_eq!(image.inverse_survival(0.0).unwrap(), 2.0);
    assert_eq!(image.inverse_survival(1.0).unwrap(), 1.0);
}

#[test]
fn inverse_survival_respects_the_lower_edge_of_a_response_jump() {
    let function = impact(
        json!({"kind":"sigmoid","midpoint":1.0,"steepness":2.0,"zero_below":2.0}),
        serde_json::Value::Null,
    );
    let source =
        ImpactSource::Histogram(HistogramExposure::new(vec![0.0, 4.0], vec![1.0]).unwrap());
    let image = function.transform_distribution(&source).unwrap();
    assert_eq!(image.quantile(0.5).unwrap(), 0.0);
    assert_eq!(image.inverse_survival(0.5).unwrap(), 0.0);
    assert!(image.inverse_survival(0.5 - 1e-10).unwrap() > 0.0);
}
