from __future__ import annotations

from typing import Literal, Optional, Union, cast

import numpy as np
import numpy.typing as npt

from crc_framework import _core

from .base import ArrayLike
from .empirical import _evaluate

DistributionFamily = Literal[
    "genextreme",
    "weibull_min",
    "weibull_max",
    "skewnorm",
    "gumbel_r",
    "gumbel_l",
    "genpareto",
]


class FittedDistribution:
    """An immutable parametric distribution implemented by the Rust core."""

    def __init__(
        self,
        family: DistributionFamily,
        *,
        location: float,
        scale: float,
        shape: Optional[float] = None,
        _native: Optional[_core.NativeDistribution] = None,
    ) -> None:
        self._native = _native or _core.fitted_distribution(
            family, location, scale, shape
        )
        self.family = family
        self.location = float(location)
        self.scale = float(scale)
        self.shape = shape

    @classmethod
    def from_parameters(
        cls,
        family: DistributionFamily,
        *,
        location: float,
        scale: float,
        shape: Optional[float] = None,
    ) -> "FittedDistribution":
        return cls(family, location=location, scale=scale, shape=shape)

    @classmethod
    def _from_native(cls, native: _core.NativeDistribution) -> "FittedDistribution":
        if native.family is None or native.location is None or native.scale is None:
            raise ValueError("native distribution does not contain fitted parameters")
        return cls(
            cast(DistributionFamily, native.family),
            location=native.location,
            scale=native.scale,
            shape=native.shape,
            _native=native,
        )

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
