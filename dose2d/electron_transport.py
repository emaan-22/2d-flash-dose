"""
Module 2 — 2D electron transmission (proposal Section 5B / 7.2, Module 2)

Replaces the spherical-particle-radius escape term used in prior nanoparticle
radiosensitization models with a slab-geometry transmission function
parameterized by:
    t     = flake thickness (nm)
    L     = lateral flake dimension (nm)
    Zeff  = effective atomic number of the material
    E     = photoelectron/Auger electron kinetic energy (keV)

Approach: modified continuous-slowing-down-approximation (CSDA) range.
We estimate the CSDA range of a low-energy electron in the flake material
using the Katz-Penfold empirical range-energy relation (a standard,
widely-cited approximation for electron ranges in the tens-of-keV range),
then compute the probability an electron generated at depth z (0 <= z <= t)
escapes the slab, and average over depth assuming uniform generation.

LIMITATION: Katz-Penfold was fit for bulk materials; using it for atomically
thin 2D flakes is an approximation. For t comparable to a few atomic layers,
channeling/surface effects not captured here may matter. Because L is
typically >> escape depth for the electron energies of interest, lateral
(edge) escape is treated as a second-order correction and left as a scale
factor for now (see LATERAL_CORRECTION below) rather than a full 3D transport
calculation -- flag this explicitly if L becomes comparable to t.
"""

import numpy as np

ELECTRON_DENSITY_SCALING = True  # Katz-Penfold range depends on density


def csda_range_g_cm2(energy_keV: float) -> float:
    """
    Katz & Penfold (1952) empirical CSDA range-energy relation, valid
    approximately over ~0.01-2.5 MeV. Returns range in g/cm^2 (density-
    independent form); divide by density to get a linear range in cm.
    """
    E_MeV = energy_keV / 1000.0
    if E_MeV <= 0:
        return 0.0
    if E_MeV < 2.5:
        # R (g/cm^2) = 0.412 * E^(1.265 - 0.0954 ln E)   [E in MeV]
        R = 0.412 * E_MeV ** (1.265 - 0.0954 * np.log(E_MeV))
    else:
        R = 0.530 * E_MeV - 0.106
    return max(R, 1e-9)


def csda_range_nm(energy_keV: float, density_g_cm3: float) -> float:
    R_g_cm2 = csda_range_g_cm2(energy_keV)
    R_cm = R_g_cm2 / density_g_cm3
    return R_cm * 1e7  # cm -> nm


def escape_probability(thickness_nm: float, energy_keV: float,
                        density_g_cm3: float, lateral_nm: float = None) -> float:
    """
    Average escape probability for electrons generated uniformly through the
    slab thickness, assuming straight-line CSDA slowing (no scattering) and
    isotropic emission collapsed to a 1D depth-averaged approximation:

        P_escape(z) = 0.5 * (1 - z / R)   for z <= R   (escape toward nearer surface)
                       ... averaged over emission depth 0..t and both surfaces

    We use the standard thin-source self-absorption result adapted from
    beta-source dosimetry: for a slab of thickness t and CSDA range R,
    the depth-averaged two-sided escape fraction is:

        P_escape_avg = 1 - (t / (2R))              if t <= R
        P_escape_avg = (R / t) - (R / (2t)) ... -> saturates as t >> R

    For simplicity and numerical stability we implement the piecewise form:
        f = min(t / R, 1.0)
        P_escape_avg = 1 - f/2   (exact for uniform generation, thin-slab limit)
    which correctly reduces to ~1 for t << R (thin flake, most electrons escape)
    and approaches 0.5 as t -> R (thick slab limit, only near-surface layer escapes).
    """
    R = csda_range_nm(energy_keV, density_g_cm3)
    if R <= 0:
        return 0.0
    f = min(thickness_nm / R, 1.0)
    P_escape_avg = 1.0 - f / 2.0

    if lateral_nm is not None and lateral_nm < R * 3:
        # Second-order correction: when the lateral flake size is NOT much
        # larger than the CSDA range, edge (in-plane) escape becomes an
        # additional pathway on top of through-thickness escape -- a small
        # flake has proportionally more perimeter/edge area exposed relative
        # to its volume, so escape probability should INCREASE, not decrease,
        # as L shrinks toward R. This is a coarse geometric enhancement
        # factor, not a rigorous 3D transport calculation -- flag for full
        # Monte Carlo transport modeling if this regime (L ~ R) matters to
        # the study, since real edge geometry (flake shape, stacking) will
        # change the exact magnitude.
        edge_enhancement = 1.0 + 0.5 * (1.0 - min(lateral_nm / (R * 3), 1.0))
        P_escape_avg *= edge_enhancement

    return float(np.clip(P_escape_avg, 0.0, 1.0))


if __name__ == "__main__":
    for t in [1, 2, 5, 10]:
        R = csda_range_nm(20, 3.9)  # 20 keV electron in Ti3C2Tx-like density
        P = escape_probability(t, 20, 3.9, lateral_nm=200)
        print(f"thickness = {t:4.1f} nm   CSDA range(20 keV) = {R:7.1f} nm   "
              f"escape probability = {P:5.3f}")
