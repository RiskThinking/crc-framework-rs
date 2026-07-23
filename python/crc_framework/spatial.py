from __future__ import annotations

from typing import Iterable, Optional, Union, cast

from crc_framework import _core

from .models import Geography

Cell = Union[int, str]


def lookup_ipcc_region(cell: Cell) -> Optional[str]:
    """Return the IPCC region after normalizing the H3 cell to resolution 4."""
    return _core.lookup_ipcc_region(cell)


def lookup_continent(cell: Cell) -> Optional[str]:
    """Return the mapped continent after normalizing to H3 resolution 4."""
    return _core.lookup_continent(cell)


def lookup_geography(cell: Cell) -> Optional[Geography]:
    """Return r5 continent and intersecting ISO3 country codes."""
    value = _core.lookup_geography(cell)
    if value is None:
        return None
    return Geography(
        cast(str, value["continent"]),
        tuple(cast(Iterable[str], value["countries"])),
    )
