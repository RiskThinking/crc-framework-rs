from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crc_framework import _core

from .base import ArrayLike
from .empirical import _evaluate


class PointMassDistribution:
    """A distribution concentrated exactly at one finite value."""

    def __init__(
        self,
        location: float,
        *,
        _native: _core.NativeDistribution | None = None,
    ) -> None:
        self.location = float(location)
        self._native = _native or _core.point_mass_distribution(self.location)

    def point_mass(self, x: ArrayLike) -> float | npt.NDArray[np.float64]:
        values = np.asarray(x, dtype=np.float64)
        masses = np.where(values == self.location, 1.0, 0.0)
        if values.ndim == 0:
            return float(masses)
        return np.asarray(masses, dtype=np.float64)

    def pdf(self, x: ArrayLike) -> float | npt.NDArray[np.float64]:
        return _evaluate(self._native, "pdf", x)

    def cdf(self, x: ArrayLike) -> float | npt.NDArray[np.float64]:
        return _evaluate(self._native, "cdf", x)

    def ppf(self, probability: ArrayLike) -> float | npt.NDArray[np.float64]:
        return _evaluate(self._native, "ppf", probability)

    def quantiles(self, probabilities: ArrayLike) -> npt.NDArray[np.float64]:
        values = np.asarray(probabilities, dtype=np.float64)
        return np.asarray(
            self._native.quantiles(values.reshape(-1).tolist()), dtype=np.float64
        ).reshape(values.shape)

    def sample(
        self,
        size: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> npt.NDArray[np.float64]:
        if rng is not None and seed is not None:
            raise ValueError("provide either rng or seed, not both")
        return np.full(size, self.location, dtype=np.float64)
