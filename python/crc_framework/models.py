from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Geography:
    continent: str
    countries: Tuple[str, ...]


@dataclass(frozen=True)
class ScenarioMetadata:
    cell: Optional[int] = None
    factor: Optional[str] = None
    pathway: Optional[str] = None
    horizon: Optional[int] = None
    country: Optional[str] = None
    continent: Optional[str] = None
    building_type: Optional[str] = None


@dataclass(frozen=True)
class TransformContext:
    cell: Optional[int] = None
    country: Optional[str] = None
    continent: Optional[str] = None
    building_type: Optional[str] = None
    historic_mean: Optional[float] = None
