"""
Custom material builder.

Lets a user specify ANY material by choosing elements and their stoichiometric
counts (e.g. {"W": 1, "S": 2} for WS2, or any hypothetical composition), and
computes the same derived properties (Zeff) used throughout the tool.

No external API or database lookup is required -- this uses only atomic
numbers and standard atomic masses, both hard-coded locally (periodic table
data), so it works fully offline with no API key.
"""

from .materials import compute_zeff

# Standard atomic weights (g/mol) for common elements relevant to 2D materials.
# Extend this table as needed -- it is intentionally NOT exhaustive of the
# full periodic table, only elements likely to appear in 2D radiosensitizer
# candidates (light elements, transition metals, chalcogens, pnictogens,
# halogens, a few heavy elements for contrast).
ATOMIC_MASS = {
    "H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.01, "N": 14.01, "O": 16.00,
    "F": 19.00, "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.06, "Cl": 35.45, "K": 39.10, "Ca": 40.08, "Ti": 47.87, "V": 50.94,
    "Cr": 52.00, "Mn": 54.94, "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55,
    "Zn": 65.38, "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.97, "Br": 79.90,
    "Zr": 91.22, "Nb": 92.91, "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
    "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76,
    "Te": 127.60, "I": 126.90, "W": 183.84, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Pb": 207.2, "Bi": 208.98, "Hf": 178.49, "Gd": 157.25,
}

ATOMIC_NUMBER = {
    "H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Zr": 40, "Nb": 41,
    "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "W": 74, "Pt": 78, "Au": 79,
    "Hg": 80, "Pb": 82, "Bi": 83, "Hf": 72, "Gd": 64,
}


class CustomMaterial:
    """
    Build a custom 2D material from a dict of {element_symbol: atom_count}.

    Example:
        WS2 = CustomMaterial({"W": 1, "S": 2}, density_g_cm3=7.5,
                              thickness_nm_range=(0.7, 5.0))
    """

    def __init__(self, composition: dict, density_g_cm3: float,
                 thickness_nm_range: tuple, name: str = None):
        unknown = [el for el in composition if el not in ATOMIC_NUMBER]
        if unknown:
            raise ValueError(
                f"Unknown element symbol(s): {unknown}. "
                f"Add them to ATOMIC_MASS / ATOMIC_NUMBER in material_builder.py."
            )
        self.composition = composition
        self.density_g_cm3 = density_g_cm3
        self.thickness_nm_range = thickness_nm_range
        self.molar_mass_g_mol = sum(ATOMIC_MASS[el] * n for el, n in composition.items())
        self.zeff = compute_zeff({el: n for el, n in composition.items()}
                                  if _can_use_shared_zeff(composition)
                                  else composition)
        self.name = name or _formula_string(composition)

    def __repr__(self):
        return (f"CustomMaterial(name={self.name!r}, Zeff={self.zeff:.2f}, "
                f"density={self.density_g_cm3} g/cm3, "
                f"thickness_range={self.thickness_nm_range} nm)")


def _can_use_shared_zeff(composition):
    # materials.compute_zeff uses its own smaller Z_TABLE; this just checks
    # compatibility so we reuse the single canonical Zeff formula everywhere.
    from .materials import Z_TABLE
    return all(el in Z_TABLE for el in composition)


def _formula_string(composition: dict) -> str:
    parts = []
    for el, n in composition.items():
        n_str = "" if n == 1 else (str(int(n)) if float(n).is_integer() else str(n))
        parts.append(f"{el}{n_str}")
    return "".join(parts)


# Patch materials.Z_TABLE at import time with the fuller ATOMIC_NUMBER table,
# so compute_zeff() works for any element in ATOMIC_NUMBER, not just the
# handful originally hard-coded in materials.py.
from . import materials as _materials_module
_materials_module.Z_TABLE.update(ATOMIC_NUMBER)


if __name__ == "__main__":
    ws2 = CustomMaterial({"W": 1, "S": 2}, density_g_cm3=7.5, thickness_nm_range=(0.7, 5.0))
    print(ws2)
    nb2c = CustomMaterial({"Nb": 2, "C": 1}, density_g_cm3=4.3, thickness_nm_range=(1.0, 5.0),
                           name="Nb2C MXene")
    print(nb2c)
