from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from crc_framework import _core

from .base import ArrayLike
from .empirical import _evaluate
from .fitted import FittedDistribution


class HurdleDistribution:
    """A point mass followed by a truncated continuous parametric tail."""

    def __init__(
        self,
        base: FittedDistribution,
        *,
        atom_probability: float,
        atom_location: float = 0.0,
        _native: Optional[_core.NativeDistribution] = None,
    ) -> None:
        self.base = base
        self.atom_probability = float(atom_probability)
        self.atom_location = float(atom_location)
        self._native = _native or _core.hurdle_distribution(
            base._native, self.atom_probability, self.atom_location
        )

    @classmethod
    def _from_native(
        cls,
        native: _core.NativeDistribution,
        base_native: _core.NativeDistribution,
        *,
        atom_probability: float,
        atom_location: float,
    ) -> "HurdleDistribution":
        return cls(
            FittedDistribution._from_native(base_native),
            atom_probability=atom_probability,
            atom_location=atom_location,
            _native=native,
        )

    def point_mass(
        self, x: ArrayLike
    ) -> Union[float, npt.NDArray[np.float64]]:
        values = np.asarray(x, dtype=np.float64)
        masses = np.where(values == self.atom_location, self.atom_probability, 0.0)
        if values.ndim == 0:
            return float(masses)
        return np.asarray(masses, dtype=np.float64)

    def pdf(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        return _evaluate(self._native, "pdf", x)

    def cdf(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        return _evaluate(self._native, "cdf", x)

    def ppf(self, probability: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
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
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        if rng is not None:
            if seed is not None:
                raise ValueError("provide either rng or seed, not both")
            return self.quantiles(rng.random(size))
        return np.asarray(self._native.sample(size, seed), dtype=np.float64)
