"""Cosmology package of cloelib.

The package provides interface with external Boltzmann solvers
or emulators via Python Protocols following the cosmology.API.
Core structure enables defining Background & Perturbation models.

Supported External Codes:
- **Background**: `camb`, `class`
- **Perturbations**: `camb`, `class`, `HMCode2020emu`
"""

from cloelib.cosmology.cosmology import (
    Cosmology,
)  # explicit import of the class or function you need

__all__: list[str] = ["Cosmology"]
