# Impact mathematics and identities

[Documentation index](README.md) · [Python guide](python.md)

The public mathematical interface is `Impact`: evaluate points with `evaluate`,
integrate a complete law with `expectation`, or create its exact response law
with `pushforward`. The same resolved function applies to custom data and
prebuilt datasets. Variable and pathway names do not alter the mathematics.

Input and output `Measurement` declarations describe the function's quantities
and units. A built-in `ImpactReference` additionally pins its model identity and
reference tables; `requirements` describes the input science and typed context
needed to resolve it. Resolution produces an ordinary context-free `Impact`.
When a required reference table has no entry for the supplied context,
resolution raises `ImpactError` with code `missing_table_key`.

```python
from crc_framework import Distribution, Impact, Ratio

function = Impact.linear(slope=-2, intercept=3, input=Ratio(), output=Ratio())
source = Distribution.discrete([0, 1], [0.9, 0.1])
response = function.pushforward(source)
assert response.mass(1) == 0.1
assert response.mass(3) == 0.9
```

Point evaluation preserves scalar and array shape. Exact pushforwards preserve
atoms and threshold conventions even when the function decreases. Unsupported
exact operations raise an explicit capability error. No implicit quantile grid
or approximation policy substitutes for a probability law.

Public impact serialization uses format `2.0`. `to_json` includes the calibrated
input and output declarations; `from_json` validates the current envelope.
`sha256` identifies the canonical mathematical definition and its declarations.
Numerical laws have their separate format `1.0`, including recorded completion
and conditioning choices.

The shared Rust kernels and independent Python reference implementation have
scientific fixtures for boundary conventions, point values, integration, and
pushforward atoms. Their generated resources are implementation assets, not a
second Python mathematical API. Edit kernel reference sources in
`tools/impact_contracts`, regenerate their resources, and run their integrity and
numerical conformance checks with the relevant native tests.
