"""
Material property database for the 2D-FLASH-Dose screening tool.

Values are approximate literature/Materials-Project-order-of-magnitude figures
intended for a SCREENING tool, not clinical dosimetry. Anchor with exact
Materials Project entries before publication (see proposal Section 9.2).

Effective atomic number (Zeff) is computed with the standard radiation-physics
mixture rule (photoelectric-weighted):

    Zeff = ( sum_i f_i * Z_i^2.94 ) ** (1 / 2.94)

where f_i is each element's fractional contribution to the total number of
electrons in the compound (electron fraction), and 2.94 is the standard
exponent used for photoelectric-dominated Zeff estimates (Mayneord-type
formula, widely used in radiation dosimetry).
"""

from dataclasses import dataclass, field
from typing import Dict

# Atomic numbers for elements we need
Z_TABLE = {
    "H": 1, "C": 6, "O": 8, "P": 15, "S": 16, "Ti": 22, "Mo": 42, "F": 9,
}

ZEFF_EXPONENT = 2.94


def compute_zeff(composition: Dict[str, float]) -> float:
    """
    composition: dict of element_symbol -> stoichiometric count (atoms per
    formula unit), e.g. {"Ti": 3, "C": 2, "O": 2} for a Ti3C2O2-type MXene.

    Returns the photoelectric-weighted effective atomic number.
    """
    total_electrons = sum(Z_TABLE[el] * n for el, n in composition.items())
    if total_electrons == 0:
        raise ValueError("Composition has zero total electrons.")
    numerator = sum(
        (Z_TABLE[el] * n / total_electrons) * (Z_TABLE[el] ** ZEFF_EXPONENT)
        for el, n in composition.items()
    )
    return numerator ** (1.0 / ZEFF_EXPONENT)


@dataclass
class Material2D:
    name: str
    composition: Dict[str, float]           # atoms per formula unit
    density_g_cm3: float                     # bulk/flake density
    thickness_nm_range: tuple                # realistic (min, max) flake thickness
    molar_mass_g_mol: float                  # g/mol of the formula unit
    zeff: float = field(init=False)

    def __post_init__(self):
        self.zeff = compute_zeff(self.composition)


# ---------------------------------------------------------------------------
# Target materials named in the proposal (Section 9.2)
# Composition / density / molar mass are approximate placeholder values —
# REPLACE with exact Materials Project entries before publication.
# ---------------------------------------------------------------------------

MATERIALS = {
    "Ti3C2Tx_MXene": Material2D(
        name="Ti3C2Tx MXene (O-terminated approx.)",
        composition={"Ti": 3, "C": 2, "O": 2},   # Tx approximated as O2 termination
        density_g_cm3=3.9,
        thickness_nm_range=(1.0, 5.0),
        molar_mass_g_mol=3 * 47.87 + 2 * 12.01 + 2 * 16.00,
    ),
    "MoS2": Material2D(
        name="Molybdenum disulfide (MoS2)",
        composition={"Mo": 1, "S": 2},
        density_g_cm3=5.06,
        thickness_nm_range=(0.65, 5.0),          # ~0.65 nm per monolayer
        molar_mass_g_mol=95.95 + 2 * 32.06,
    ),
    "Black_Phosphorus": Material2D(
        name="Black phosphorus (BP)",
        composition={"P": 1},
        density_g_cm3=2.69,
        thickness_nm_range=(0.53, 5.0),          # ~0.53 nm per monolayer
        molar_mass_g_mol=30.97,
    ),
}

# Reference/comparison materials from prior spherical-nanoparticle literature
REFERENCE_SPHERICAL = {
    "Au_nanoparticle": {"Z": 79, "density_g_cm3": 19.3},
    "HfO2_nanoparticle": {"Z_eff_approx": 67.5, "density_g_cm3": 9.68},
}

if __name__ == "__main__":
    for key, mat in MATERIALS.items():
        print(f"{key:20s}  Zeff = {mat.zeff:6.2f}  density = {mat.density_g_cm3} g/cm3  "
              f"thickness range = {mat.thickness_nm_range} nm")
