"""
Combined Dose Enhancement Factor model (proposal Section 7.2 / 7.3)

Couples:
  Module 1 (photon_interaction.py)  -> photoelectric absorption contrast
  Module 2 (electron_transport.py)  -> 2D-geometry electron escape probability
  Module 3 (oxygen_kinetics.py)     -> FLASH-branch oxygen depletion fraction (reported
                                        alongside DEF, NOT multiplied into it -- see
                                        proposal Section 5A: this is a separate,
                                        explicitly-scoped contributor, not conflated
                                        with the physical dose-enhancement mechanism)

DEF(r): local dose enhancement at radial distance r (nm) from the flake
surface, modeled with a simple exponential deposition kernel representing
the residual range of escaped electrons depositing energy in surrounding
tissue:

    DEF(r) = 1 + (contrast - 1) * P_escape * exp(-r / lambda_dep)

where lambda_dep is the characteristic electron deposition length in tissue
(taken as the CSDA range of the electron in tissue, NOT in the flake
material -- electrons deposit their energy in the medium they travel
through after escaping).

LIMITATION: this exponential kernel is a standard simplifying approximation
for point/planar dose-deposition kernels (as used in many screening-level
dosimetry tools) and is not a full point-kernel or Monte Carlo convolution.
Replace with a proper dose-point-kernel convolution for publication-grade
numbers.
"""

import numpy as np

from . import photon_interaction as pi
from . import electron_transport as et
from . import oxygen_kinetics as ok

Z_TISSUE_EFF = 7.42          # soft tissue / water-equivalent Zeff
TISSUE_DENSITY_G_CM3 = 1.04  # approx soft tissue density


def compute_DEF_profile(material, energy_keV: float, thickness_nm: float,
                         lateral_nm: float, r_max_nm: float = 2000,
                         n_points: int = 300, auger_energy_keV: float = 20.0):
    """
    Returns (r_array_nm, DEF_array) for a given material at a fixed photon
    energy, flake thickness, and lateral size.

    material: a dose2d.materials.Material2D instance
    auger_energy_keV: representative secondary-electron energy used for the
        escape-probability and deposition-length calculations (order of
        magnitude for Auger/photoelectron cascades in this energy range;
        refine with a proper electron-energy spectrum before publication).
    """
    contrast = pi.dose_enhancement_photon_term(material.zeff, Z_TISSUE_EFF, energy_keV)
    P_escape = et.escape_probability(thickness_nm, auger_energy_keV,
                                      material.density_g_cm3, lateral_nm=lateral_nm)
    lambda_dep_nm = et.csda_range_nm(auger_energy_keV, TISSUE_DENSITY_G_CM3)

    r = np.linspace(0, r_max_nm, n_points)
    DEF = 1.0 + (contrast - 1.0) * P_escape * np.exp(-r / lambda_dep_nm)
    return r, DEF, {"contrast": contrast, "P_escape": P_escape, "lambda_dep_nm": lambda_dep_nm}


def screen_materials(materials_dict, energy_keV: float, lateral_nm: float,
                      dose_rate_regimes=(("conventional", 0.03), ("FLASH", 100.0)),
                      total_dose_Gy: float = 10.0, pO2_baseline_uM: float = 30.0):
    """
    Produces a comparative screening table (proposal Section 7.3):
    for each material, at its representative (mid-range) thickness, reports
    peak DEF, photon contrast, escape probability, deposition length, and
    the Module-3 oxygen-depletion fraction under each dose-rate regime
    (reported separately, per the Section 5A scoping note).
    """
    rows = []
    for key, mat in materials_dict.items():
        t_mid = float(np.mean(mat.thickness_nm_range))
        r, DEF, meta = compute_DEF_profile(mat, energy_keV, t_mid, lateral_nm)
        row = {
            "material": key,
            "Zeff": round(mat.zeff, 2),
            "thickness_nm": round(t_mid, 2),
            "photon_contrast": round(meta["contrast"], 1),
            "P_escape": round(meta["P_escape"], 3),
            "peak_DEF": round(float(DEF[0]), 2),
            "lambda_dep_nm": round(meta["lambda_dep_nm"], 1),
        }
        for label, rate in dose_rate_regimes:
            frac = ok.fractional_depletion(pO2_baseline_uM, rate, total_dose_Gy)
            row[f"O2_depletion_frac_{label}"] = round(frac, 3)
        rows.append(row)
    return rows
