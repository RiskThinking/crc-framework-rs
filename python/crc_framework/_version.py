"""Convert the workspace release number to its Python distribution spelling."""

import re

_PRERELEASE = re.compile(r"(\d+\.\d+\.\d+)-(alpha|beta|rc)(?:\.(\d+))?")
_PYTHON_PHASE = {"alpha": "a", "beta": "b", "rc": "rc"}


def python_version(workspace_version: str) -> str:
    prerelease = _PRERELEASE.fullmatch(workspace_version)
    if prerelease is None:
        return workspace_version
    version, phase, number = prerelease.groups()
    return f"{version}{_PYTHON_PHASE[phase]}{int(number or 0)}"
