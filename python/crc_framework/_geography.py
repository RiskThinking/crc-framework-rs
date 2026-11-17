"""Exact H3 reference lookups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from . import _core


@dataclass(frozen=True, slots=True)
class Geography:
    continent: str
    countries: tuple[str, ...]


def _cell(cell: int) -> int:
    if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
        raise TypeError("cell must be a nonnegative H3 integer")
    return cell


def lookup_continent(cell: int) -> str | None:
    """Return the reference continent at H3 resolution four."""
    return _core.lookup_continent(_cell(cell))


def lookup_ipcc_region(cell: int) -> str | None:
    """Return the IPCC reference region at H3 resolution four."""
    return _core.lookup_ipcc_region(_cell(cell))


def lookup_geography(cell: int) -> Geography | None:
    """Return the continent and intersecting ISO3 countries at resolution five."""
    result = _core.lookup_geography(_cell(cell))
    if result is None:
        return None
    return Geography(
        cast(str, result["continent"]), tuple(cast(list[str], result["countries"]))
    )
