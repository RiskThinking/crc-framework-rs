from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from crc_framework import _core

from .base import ArrayLike
from .empirical import _evaluate

Interpolation = Literal["linear_probability", "log_return_period"]
Tail = Literal["upper", "lower"]


class TabulatedDistribution:
    """A quantile function represented by explicit probability/value pairs."""

    def __init__(
        self,
        probabilities: ArrayLike,
        values: ArrayLike,
        *,
        interpolation: Interpolation = "linear_probability",
        extrapolate: bool = False,
    ) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        self.values = np.asarray(values, dtype=np.float64).reshape(-1)
        self.interpolation = interpolation
        self.extrapolate = extrapolate
        self._native = _core.tabulated_distribution(
            self.probabilities.tolist(),
            self.values.tolist(),
            interpolation,
            extrapolate,
        )

    @classmethod
    def _from_native(
        cls,
        native: _core.NativeDistribution,
        probabilities: npt.NDArray[np.float64],
        *,
        interpolation: Interpolation = "linear_probability",
    ) -> "TabulatedDistribution":
        instance = cls.__new__(cls)
        instance.probabilities = probabilities
        instance.values = np.asarray(native.quantiles(probabilities.tolist()), dtype=np.float64)
        instance.interpolation = interpolation
        instance.extrapolate = False
        instance._native = native
        return instance

    @classmethod
    def from_return_periods(
        cls,
        periods: ArrayLike,
        values: ArrayLike,
        *,
        tail: Tail,
        interpolation: Interpolation = "log_return_period",
        extrapolate: bool = False,
    ) -> "TabulatedDistribution":
        instance = cls.__new__(cls)
        raw_periods = np.asarray(periods, dtype=np.float64).reshape(-1)
        raw_values = np.asarray(values, dtype=np.float64).reshape(-1)
        probabilities = (
            1.0 - 1.0 / raw_periods if tail == "upper" else 1.0 / raw_periods
        )
        order = np.argsort(probabilities)
        instance.probabilities = probabilities[order]
        instance.values = raw_values[order]
        instance.interpolation = interpolation
        instance.extrapolate = extrapolate
        instance._native = _core.return_period_distribution(
            raw_periods.tolist(),
            raw_values.tolist(),
            tail,
            interpolation,
            extrapolate,
        )
        return instance

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
        if rng is not None and seed is not None:
            raise ValueError("provide either rng or seed, not both")
        generator = rng or np.random.default_rng(seed)
        minimum, maximum = self.probabilities[0], self.probabilities[-1]
        return self.quantiles(generator.uniform(minimum, maximum, size))
