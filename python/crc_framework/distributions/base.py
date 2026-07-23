from __future__ import annotations

from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

ArrayLike = Union[float, Sequence[float], npt.NDArray[np.float64]]


@runtime_checkable
class Distribution(Protocol):
    """A continuous or interpolated distribution sampled by CDF probability."""

    def pdf(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]: ...

    def cdf(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]: ...

    def ppf(self, probability: ArrayLike) -> Union[float, npt.NDArray[np.float64]]: ...

    def quantiles(self, probabilities: ArrayLike) -> npt.NDArray[np.float64]: ...

    def sample(
        self,
        size: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> npt.NDArray[np.float64]: ...
