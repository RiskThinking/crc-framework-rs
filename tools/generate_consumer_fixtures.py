"""Regenerate native-Python reference quantiles for Rust and WASM parity tests.

Run from the repository root after rebuilding the Python extension:
    uv run python tools/generate_consumer_fixtures.py
"""
import json
from pathlib import Path

from crc_framework.distributions import (
    FittedDistribution, HurdleDistribution, PointMassDistribution, TabulatedDistribution,
)


def distribution(row):
    kind = row['curve_kind']
    if kind == 'no_data':
        return None
    if kind == 'point_mass':
        return PointMassDistribution(row['curve_location'])
    if kind == 'tabulated':
        return TabulatedDistribution(row['curve_probabilities'], row['curve_values'])
    base = FittedDistribution.from_parameters(
        row['curve_type'], shape=row.get('curve_shape'),
        location=row['curve_location'], scale=row['curve_scale'],
    )
    if kind == 'fitted':
        return base
    return HurdleDistribution(base, atom_probability=row['curve_atom_probability'],
                              atom_location=row['curve_atom_location'])


def expected(row, q):
    kind = row['curve_kind']
    if kind == 'no_data' or (kind == 'fitted' and q in (0, 1)) or (kind == 'hurdle' and q == 1):
        return None
    if kind == 'tabulated' and not row['curve_probabilities'][0] <= q <= row['curve_probabilities'][-1]:
        return None
    return float(distribution(row).ppf(q))


def cases():
    result = []
    for family, shape, location, scale in [
        ('genextreme', .1, 1.2, 2), ('weibull_min', 2.2, -.5, 1.5),
        ('weibull_max', 1.7, 4, 1.2), ('skewnorm', 1.5, .5, 1.3),
        ('gumbel_r', None, 1, .8), ('gumbel_l', None, 1, .8), ('genpareto', .15, -.2, 1.1),
        ('genextreme', 0, 0, 1), ('genpareto', 0, 0, 1),
        ('skewnorm', -8, 0, 1), ('skewnorm', 0, 0, 1),
    ]:
        row = dict(curve_kind='fitted', curve_type=family, curve_shape=shape,
                   curve_location=location, curve_scale=scale)
        result.append((family + '-' + str(shape), row, [0, .0001, .01, .1, .5, .9, .99, .9999, 1]))
    for family, shape, atom in [('gumbel_r', None, 0), ('gumbel_r', None, 40),
                                 ('genpareto', .2, 100), ('skewnorm', 1.5, 0)]:
        row = dict(curve_kind='hurdle', curve_type=family, curve_shape=shape,
                   curve_location=0, curve_scale=1, curve_atom_probability=.2, curve_atom_location=atom)
        result.append(('hurdle-' + family + '-' + str(atom), row, [0, .1, .2, .200001, .5, .99, .9999, 1]))
    result.extend([
        ('point-mass-negative-zero', dict(curve_kind='point_mass', curve_type='point_mass',
            curve_location=-0.0, curve_scale=0, curve_atom_probability=1, curve_atom_location=-0.0), [0, .5, 1]),
        ('tabulated-plateau', dict(curve_kind='tabulated', curve_type='linear_probability',
            curve_probabilities=[0, .2, .5, .9, 1], curve_values=[0, 1, 1, 8, 10]), [0, .1, .2, .35, .5, .85, .9, .99, 1]),
        ('tabulated-narrow', dict(curve_kind='tabulated', curve_type='linear_probability',
            curve_probabilities=[.998, .999], curve_values=[.1597156259295522, .17687592042871514]),
            [.5, .99, .998, .9985, .999, .9995]),
    ])
    for reason in ['below_effective_resolution', 'insufficient_informative_support',
                   'degenerate_effective_range', 'scientific_exclusion']:
        result.append((reason, dict(curve_kind='no_data', curve_type=reason), [0, .5, 1]))
    return [dict(name=name, properties=row, probabilities=qs,
                 expected=[expected(row, q) for q in qs]) for name, row, qs in result]


if __name__ == '__main__':
    Path('tests/fixtures/canonical-quantiles.json').write_text(json.dumps(cases(), indent=2) + '\n')
