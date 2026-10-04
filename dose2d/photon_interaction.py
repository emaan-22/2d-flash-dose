"""
Module 1 -- Photon interaction (v2: photoelectric + Compton)

UPGRADE from v1: v1 used ONLY a calibrated photoelectric power law, which
produced two known problems (documented in the original README): (a) peak DEF
values far larger than real experimental values, and (b) a contrast ratio
that was completely energy-independent (unphysical).

FIX: this version adds the Compton scattering contribution using the
Klein-Nishina formula -- an EXACT closed-form result from QED (not a fit, not
an approximation), which is roughly Z-independent per electron and dominates
at higher photon energies. Adding it does two correct, physically-expected
things:
  1. Dilutes the material/tissue contrast ratio at high energy (since Compton
     scales with electron density, which is similar for material and tissue,
     while photoelectric is strongly Z-dependent) -- this directly reduces
     the overestimated DEF magnitude problem.
  2. Makes the contrast ratio energy-dependent, as it should be.

The photoelectric term is still the same calibrated power law as v1 (still an
approximation pending direct NIST XCOM interpolation -- that caveat remains).
"""

import numpy as np

# --- Photoelectric term (unchanged from v1, still an approximation) ---
E_REF_KEV = 100.0
MU_PE_WATER_REF = 2.76e-3
MU_PE_GOLD_REF = 5.16
Z_WATER_EFF = 7.42
Z_GOLD = 79.0
N_EXPONENT = 4.6
M_EXPONENT = 3.1
_K_RATIO = MU_PE_GOLD_REF / MU_PE_WATER_REF
_Z_RATIO_PRED = (Z_GOLD / Z_WATER_EFF) ** N_EXPONENT
_CALIBRATION_FACTOR = _K_RATIO / _Z_RATIO_PRED

# --- Compton term: EXACT Klein-Nishina physics ---
ELECTRON_REST_MASS_KEV = 511.0
R_E_CM = 2.8179403262e-13   # classical electron radius, cm
N_A = 6.02214076e23          # Avogadro's number


def klein_nishina_cross_section_cm2(energy_keV: float) -> float:
    """
    Total Klein-Nishina cross section per electron (cm^2), as a function of
    photon energy. This is an EXACT formula from QED -- no fitting involved.
    """
    eps = energy_keV / ELECTRON_REST_MASS_KEV
    if eps <= 0:
        return 0.0
    term1 = (1 + eps) / eps**2
    term2 = (2 * (1 + eps) / (1 + 2 * eps)) - (np.log(1 + 2 * eps) / eps)
    term3 = np.log(1 + 2 * eps) / (2 * eps)
    term4 = (1 + 3 * eps) / (1 + 2 * eps)**2
    sigma = 2 * np.pi * R_E_CM**2 * (term1 * term2 + term3 - term4)
    return sigma


def z_over_a_ratio(zeff: float) -> float:
    """
    Approximate Z/A ratio (electrons per unit mass, roughly), used to convert
    the per-electron Klein-Nishina cross section into a mass attenuation
    coefficient. Z/A is close to 0.5 for light/medium elements and decreases
    slightly for very heavy elements (e.g. gold Z=79, A=197, Z/A=0.401).
    This linear approximation is a standard simplification in radiation
    physics texts (exact per-element values should be used for final
    publication-grade numbers).
    """
    return max(0.40, 0.55 - 0.0009 * zeff)


def mu_compton(zeff: float, energy_keV: float) -> float:
    """Compton mass attenuation coefficient (cm^2/g) via exact Klein-Nishina."""
    sigma_cm2 = klein_nishina_cross_section_cm2(energy_keV)
    za = z_over_a_ratio(zeff)
    # electrons per gram = Z/A * N_A ; mass atten coeff = sigma * electrons/gram
    electrons_per_gram = za * N_A
    return sigma_cm2 * electrons_per_gram


def mu_photoelectric(zeff: float, energy_keV: float, density_g_cm3: float = None) -> float:
    mu_mass = (
        MU_PE_WATER_REF
        * _CALIBRATION_FACTOR
        * (zeff / Z_WATER_EFF) ** N_EXPONENT
        * (energy_keV / E_REF_KEV) ** (-M_EXPONENT)
    )
    if density_g_cm3 is not None:
        return mu_mass * density_g_cm3
    return mu_mass


def mu_total(zeff: float, energy_keV: float, density_g_cm3: float = None) -> float:
    """
    Total mass (or linear, if density given) attenuation coefficient:
    photoelectric + Compton. This is the physically-correct quantity to
    compare between material and tissue -- using photoelectric alone (as v1
    did) overstates the contrast, especially at higher energies.
    """
    mu_pe = mu_photoelectric(zeff, energy_keV)
    mu_c = mu_compton(zeff, energy_keV)
    mu_mass_total = mu_pe + mu_c
    if density_g_cm3 is not None:
        return mu_mass_total * density_g_cm3
    return mu_mass_total


def interaction_probability(zeff, energy_keV, density_g_cm3, thickness_cm):
    mu_linear = mu_total(zeff, energy_keV, density_g_cm3)
    return 1.0 - np.exp(-mu_linear * thickness_cm)


def dose_enhancement_photon_term(zeff_material: float, zeff_tissue: float,
                                  energy_keV: float) -> float:
    """
    v2: now uses TOTAL (photoelectric + Compton) attenuation contrast, so the
    ratio is energy-dependent and bounded (approaches 1 at high energy where
    Compton, which is similar for material and tissue, dominates both).
    """
    mu_material = mu_total(zeff_material, energy_keV)
    mu_tissue = mu_total(zeff_tissue, energy_keV)
    return mu_material / mu_tissue


def photoelectric_fraction(zeff: float, energy_keV: float) -> float:
    """Diagnostic: what fraction of total attenuation is photoelectric vs Compton."""
    pe = mu_photoelectric(zeff, energy_keV)
    c = mu_compton(zeff, energy_keV)
    return pe / (pe + c)


if __name__ == "__main__":
    print("Energy-dependence check (v2 fixes v1's energy-independence bug):")
    for E in [50, 100, 150, 200, 500]:
        contrast = dose_enhancement_photon_term(35.16, Z_WATER_EFF, E)  # MoS2 vs tissue
        pe_frac = photoelectric_fraction(35.16, E)
        print(f"E = {E:5.0f} keV   MoS2/tissue contrast = {contrast:8.2f}   "
              f"(photoelectric fraction of MoS2 attenuation: {pe_frac:.1%})")
    print()
    print("Peak-DEF sanity check (should now be lower/more realistic than v1's 1282x):")
    for E in [50, 100, 150]:
        print(f"  E={E} keV: contrast = {dose_enhancement_photon_term(35.16, Z_WATER_EFF, E):.1f}x")
