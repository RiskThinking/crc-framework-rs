from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from crc_framework import _core

from .base import ArrayLike


def _evaluate(
    native: _core.NativeDistribution, method: str, values: ArrayLike
) -> Union[float, npt.NDArray[np.float64]]:
    array = np.asarray(values, dtype=np.float64)
    function = getattr(native, method)
    if array.ndim == 0:
        return float(function(float(array)))
    flat = np.fromiter((function(float(value)) for value in array.flat), dtype=np.float64)
    return flat.reshape(array.shape)


class EmpiricalDistribution:
    """An empirical CDF constructed from raw observations."""

    def __init__(self, samples: ArrayLike) -> None:
        self.samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._native = _core.empirical_distribution(self.samples.tolist())

    def pdf(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        return _evaluate(self._native, "pdf", x)

    def cdf(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        return _evaluate(self._native, "cdf", x)

    def ppf(self, probability: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        return _evaluate(self._native, "ppf", probability)

    def quantiles(self, probabilities: ArrayLike) -> npt.NDArray[np.float64]:
        values = np.asarray(probabilities, dtype=np.float64)
        return np.asarray(self._native.quantiles(values.reshape(-1).tolist())).reshape(values.shape)

    def sample(
        self,
        size: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        if rng is not None:
            if seed is not None:
                raise ValueError("provide either rng or seed, not both")
            return self.quantiles(rng.random(size))
        return np.asarray(self._native.sample(size, seed), dtype=np.float64)
