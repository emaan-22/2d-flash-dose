"""
Module 3 -- Oxygen-depletion kinetics (v2: Michaelis-Menten ROD model)

UPGRADE from v1: v1 used a linear consumption term (-G*Ddot) with an ad-hoc
replenishment term and a placeholder G-value. This version uses a
Michaelis-Menten-form radiolytic-oxygen-depletion (ROD) model with REAL,
CITED parameter values from the FLASH radiobiology literature:

    dp/dt = -G0 * Ddot * p/(k_ROD + p) + k_repl * (p_baseline - p)

- G0 (radiolytic consumption rate): literature range ~0.25-0.75 micromolar/Gy
  (or ~0.1-0.42 mmHg/Gy in the original units). Sources:
    * Petersson et al. model: G = 0.342 uM/Gy (PMC8854455)
    * Pratx & Kapp: 0.75 uM/Gy (cited in PMC8854455)
    * Weiss et al. / Koch: 0.45 uM/Gy (cited in PMC8854455)
    * Espinosa-Rodriguez et al., Phys. Med. Biol. 2023 (arXiv:2310.01281,
      IOPscience ad8291): G0 = 0.25 mmHg/Gy reference value, fitted against
      multiple independent oxygen-depletion measurement datasets
  This tool uses G0 = 0.4 uM/Gy as a representative central literature value
  -- adjustable in the UI.
- k_ROD (Michaelis-Menten half-saturation constant, giving the experimentally
  observed SATURATION of oxygen consumption at high initial oxygen levels):
  literature value ~1-2.5 mmHg (Espinosa-Rodriguez et al. used k_ROD = 1 mmHg;
  a related model used k = 2.5 mmHg). Converted here to approximately
  1.3-3.3 uM using a standard ~1.3 uM/mmHg tissue O2 solubility conversion --
  this conversion factor itself is an approximation and should be checked
  against the exact conditions (temperature, medium) of whichever source
  paper's numbers are being matched most closely.
- k_repl (vascular/diffusive replenishment rate constant): still a
  literature-order-of-magnitude placeholder (not found as a standard
  tabulated "G-value"-style constant in the same way as G0/k_ROD) -- flagged
  for further literature-specific calibration.

IMPORTANT, read before quoting numbers from this module:
Per a 2026 review (Frontiers in Physics, 'Review of oxygen measurement and
relevance to the mechanisms of FLASH radiotherapy'), measured oxygen
consumption data does NOT robustly support radiolytic oxygen depletion alone
as sufficient to explain the full FLASH normal-tissue-sparing effect at
clinically relevant oxygen tensions -- radical-radical recombination kinetics
and immune/inflammatory response are also implicated (consistent with this
tool's original Section 5A scoping note). This module computes ONE
quantifiable, citable contributor, not a complete account of the FLASH
mechanism. State it this way in any paper or thesis.
"""

import numpy as np
from scipy.integrate import solve_ivp

G0_DEFAULT_UM_PER_GY = 0.4          # central literature estimate, see docstring
K_ROD_DEFAULT_UM = 2.0              # Michaelis-Menten half-saturation, approx converted
K_REPL_DEFAULT_PER_S = 0.05         # still a placeholder (see docstring)

G0_LITERATURE_RANGE = (0.25, 0.75)  # uM/Gy, range spanning cited sources above


def deplete_oxygen(pO2_initial_uM: float, dose_rate_Gy_s: float,
                    total_dose_Gy: float, G0_uM_per_Gy: float = G0_DEFAULT_UM_PER_GY,
                    k_ROD_uM: float = K_ROD_DEFAULT_UM,
                    k_repl_per_s: float = K_REPL_DEFAULT_PER_S, n_points: int = 200):
    """
    Integrates dp/dt = -G0*Ddot*p/(k_ROD+p) + k_repl*(p_baseline - p) over the
    irradiation pulse duration, clipped at p=0.
    """
    if dose_rate_Gy_s <= 0:
        raise ValueError("dose_rate_Gy_s must be positive.")

    pulse_duration_s = total_dose_Gy / dose_rate_Gy_s

    def rhs(t, y):
        p = max(y[0], 0.0)
        consumption = G0_uM_per_Gy * dose_rate_Gy_s * (p / (k_ROD_uM + p)) if p > 0 else 0.0
        replenishment = k_repl_per_s * (pO2_initial_uM - p)
        dpdt = -consumption + replenishment
        if p <= 0 and dpdt < 0:
            return [0.0]
        return [dpdt]

    def hit_zero(t, y):
        return y[0]
    hit_zero.terminal = True
    hit_zero.direction = -1

    t_eval = np.linspace(0, pulse_duration_s, n_points)
    sol = solve_ivp(rhs, [0, pulse_duration_s], [pO2_initial_uM], t_eval=t_eval,
                     events=hit_zero, method="RK45", max_step=max(pulse_duration_s / 50, 1e-6))
    return sol.t, np.clip(sol.y[0], 0, None)


def fractional_depletion(pO2_initial_uM: float, dose_rate_Gy_s: float,
                          total_dose_Gy: float, G0_uM_per_Gy: float = G0_DEFAULT_UM_PER_GY,
                          k_ROD_uM: float = K_ROD_DEFAULT_UM,
                          k_repl_per_s: float = K_REPL_DEFAULT_PER_S) -> float:
    t, O2 = deplete_oxygen(pO2_initial_uM, dose_rate_Gy_s, total_dose_Gy,
                            G0_uM_per_Gy, k_ROD_uM, k_repl_per_s)
    final_O2 = O2[-1]
    return float(1.0 - final_O2 / pO2_initial_uM) if pO2_initial_uM > 0 else 0.0


if __name__ == "__main__":
    pO2_baseline = 30.0
    print(f"G0 literature range used: {G0_LITERATURE_RANGE} uM/Gy (central value {G0_DEFAULT_UM_PER_GY})")
    for label, dose_rate in [("conventional", 0.03), ("intermediate", 1.0), ("FLASH", 100.0)]:
        frac = fractional_depletion(pO2_baseline, dose_rate, total_dose_Gy=10.0)
        print(f"{label:14s} dose rate = {dose_rate:7.3f} Gy/s   fractional O2 depletion = {frac:6.3f}")
