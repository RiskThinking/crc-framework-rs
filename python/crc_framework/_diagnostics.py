"""Immutable diagnostics shared by numerical operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

ScientificValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | Sequence["ScientificValue"]
    | Mapping[str, "ScientificValue"]
)


class DiagnosticSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: DiagnosticSeverity
    path: str
    details: Mapping[str, ScientificValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze(self.details))


def _freeze(value: ScientificValue) -> ScientificValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_freeze(item) for item in value)
    return value
