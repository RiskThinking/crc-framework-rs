"""Native Python distributions and WASM use the same shared reference cases."""
import json
from pathlib import Path

import pytest

from tools.generate_consumer_fixtures import expected

CASES = json.loads((Path(__file__).parent / 'fixtures/canonical-quantiles.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_consumer_parity(case):
    for q, reference in zip(case['probabilities'], case['expected']):
        value = expected(case['properties'], q)
        if reference is None:
            assert value is None
        else:
            assert value == pytest.approx(reference, rel=2e-10, abs=2e-10)
