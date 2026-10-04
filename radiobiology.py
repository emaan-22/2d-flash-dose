"""
NEW module -- Linear-Quadratic (LQ) radiobiology model.

This connects the tool's physics output (DEF) to an actual "expected cancer
treatment performance" number, using the standard, textbook LQ cell-survival
model:

    S(D) = exp(-alpha*D - beta*D^2)

where S is the surviving fraction of cells after dose D, and alpha, beta are
tissue/tumor-type-specific radiosensitivity parameters (alpha/beta ratio is
the commonly reported/tabulated value in radiobiology).

A radiosensitizing material modifies the EFFECTIVE dose delivered to the
tumor locally, via a Sensitization Enhancement Ratio (SER). This tool
approximates SER from the physical DEF as a conservative, literature-informed
mapping (see SER_FROM_DEF below) -- this mapping is the weakest-justified
part of this module and should be treated as illustrative/order-of-magnitude,
NOT a validated clinical prediction, until calibrated against real SER data
for the specific material (e.g. the real Ti3C2Tx MXene data in this tool's
Material Library, where DEF and SER are reported as different quantities --
see materials_database.py docstring).

Typical literature alpha/beta ratios (Gy), commonly tabulated in radiobiology
textbooks and reviews, used here as presets:
    - Most epithelial tumors (early-responding): alpha/beta ~ 10 Gy
    - Prostate cancer (notably LOW alpha/beta): alpha/beta ~ 1.5-3 Gy
    - Late-responding normal tissue: alpha/beta ~ 3 Gy
These are widely-cited rule-of-thumb ranges, not specific to any one study --
always use the specific published value for the exact tumor type being
discussed in a real analysis.
"""

import numpy as np

ALPHA_BETA_PRESETS = {
    "Generic epithelial tumor (early-responding)": 10.0,
    "Prostate cancer (low alpha/beta)": 2.5,
    "Late-responding normal tissue": 3.0,
    "Glioblastoma (typical range)": 10.0,
}

# A fixed alpha value is also needed (alpha/beta alone isn't enough to define
# the curve) -- using a commonly-cited representative alpha for tumor cells.
ALPHA_DEFAULT_PER_GY = 0.3


def survival_fraction(dose_Gy, alpha_per_Gy: float, beta_per_Gy2: float):
    dose_Gy = np.asarray(dose_Gy, dtype=float)
    return np.exp(-alpha_per_Gy * dose_Gy - beta_per_Gy2 * dose_Gy**2)


def ser_from_def(peak_DEF: float, scaling_power: float = 0.15, cap: float = 2.0) -> float:
    """
    Conservative, DELIBERATELY DAMPENED mapping from physical DEF to a
    biological Sensitization Enhancement Ratio (SER).

    WHY dampened: physical dose enhancement (DEF) is a LOCAL, nanoscale
    quantity (only matters within microns of the material, see Module 2's
    lambda_dep), while SER is measured as a WHOLE-CELL or whole-tumor
    biological endpoint. A DEF of 1.7x at the material surface does NOT mean
    the whole cell sees 1.7x the dose -- only the fraction of cell volume
    within the deposition range does. This mapping uses a damped power law
    (scaling_power << 1) so that even a large DEF produces a modest,
    plausible SER, capped at `cap` -- this is a rough, illustrative
    simplification, NOT a validated biological model. Always prefer a real
    measured SER (see Material Library) over this estimate when one exists.
    """
    if peak_DEF <= 1.0:
        return 1.0
    ser = peak_DEF ** scaling_power
    return float(min(ser, cap))


def expected_treatment_outcome(total_dose_Gy: float, alpha_per_Gy: float,
                                beta_per_Gy2: float, peak_DEF: float):
    """
    Returns a dict comparing survival fraction WITHOUT vs WITH the
    radiosensitizer (via the damped SER mapping above), at a given total dose.
    """
    ser = ser_from_def(peak_DEF)
    S_without = survival_fraction(total_dose_Gy, alpha_per_Gy, beta_per_Gy2)
    S_with = survival_fraction(total_dose_Gy * ser, alpha_per_Gy, beta_per_Gy2)
    return {
        "SER_estimate": ser,
        "survival_fraction_without_sensitizer": float(S_without),
        "survival_fraction_with_sensitizer": float(S_with),
        "relative_cell_kill_improvement": float((S_without - S_with) / S_without) if S_without > 0 else None,
    }


if __name__ == "__main__":
    for name, ab in ALPHA_BETA_PRESETS.items():
        beta = ALPHA_DEFAULT_PER_GY / ab
        out = expected_treatment_outcome(2.0, ALPHA_DEFAULT_PER_GY, beta, peak_DEF=1.7)
        print(f"{name:40s} SER~{out['SER_estimate']:.2f}  "
              f"S(without)={out['survival_fraction_without_sensitizer']:.3f}  "
              f"S(with)={out['survival_fraction_with_sensitizer']:.3f}")
