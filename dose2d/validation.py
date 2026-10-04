"""
Validation module -- compares tool predictions against the limited set of
REAL experimental data points found in literature, so reliability can be
assessed honestly rather than assumed.

IMPORTANT: as of this search, only ONE directly comparable quantitative
experimental DEF value was found for any material in this tool's library
(Ti3C2Tx MXene). This is explicitly flagged as an n=1 validation set, which
is scientifically insufficient to claim general reliability -- any paper or
thesis using this tool MUST state this limitation plainly, not just cite the
single matching data point as if it were comprehensive validation.
"""

import numpy as np
from .def_model import compute_DEF_profile

VALIDATION_CASES = [
    {
        "material_key": "Ti3C2Tx_MXene",
        "real_DEF": 2.5,
        "real_DEF_context": "kV X-ray, human soft-tissue sarcoma (HT1080) cells",
        "reference": "Rafetseder et al., Biomaterials Science, 2023, 11(24), 7826.",
        "effective_energy_keV_estimate": 30.0,  # typical kV-range effective energy
    },
]


def run_validation(materials_db):
    results = []
    for case in VALIDATION_CASES:
        mat = materials_db[case["material_key"]]
        t_mid = sum(mat.thickness_nm_range) / 2
        row = {"material": mat.name, "real_DEF": case["real_DEF"],
               "context": case["real_DEF_context"], "reference": case["reference"]}
        energy_sweep = {}
        for E in [30, 50, 80, 100, 150, 200]:
            r, DEF, meta = compute_DEF_profile(mat, E, t_mid, 200.0, r_max_nm=10)
            error_pct = (DEF[0] - case["real_DEF"]) / case["real_DEF"] * 100
            energy_sweep[E] = {"predicted_DEF": float(DEF[0]), "error_pct": float(error_pct)}
        row["energy_sweep"] = energy_sweep
        row["best_match_energy_keV"] = min(energy_sweep, key=lambda e: abs(energy_sweep[e]["error_pct"]))
        row["best_match_error_pct"] = energy_sweep[row["best_match_energy_keV"]]["error_pct"]
        results.append(row)
    return results


GOLD_NP_DEF_RANGE_REFERENCE = {
    "range": (1.5, 2.51),
    "context": "Gold nanoparticles (spherical, extensively studied), across multiple kV X-ray "
               "spectra; DEF >= 1.5 reported for 83% of tested spectra, max 2.51.",
    "reference": "Impact of the Spectral Composition of Kilovoltage X-rays on High-Z "
                "Nanoparticle-Assisted Dose Enhancement, PMC8199749.",
    "note": "This is NOT a direct validation of any 2D material in this tool -- it is a "
            "broader magnitude sanity-check confirming that realistic high-Z radiosensitizer "
            "DEF values cluster in the ~1.1-2.5x range, not the >100x values this tool's v1 "
            "(photoelectric-only) model used to predict. Use this to judge whether a given "
            "prediction's MAGNITUDE is physically plausible, not as a per-material validation.",
}

VALIDATION_SUMMARY_TEXT = (
    "This tool's only direct quantitative validation point (n=1) is Ti3C2Tx MXene "
    "(Rafetseder et al. 2023, real DEF up to 2.5 under kV X-ray). The tool's "
    "prediction is within ~10% of this value specifically at low (kilovoltage, "
    "~30 keV) photon energies, where real high-Z radiosensitization experiments "
    "are conducted, and diverges substantially (>50% underestimate) at higher "
    "energies -- consistent with the expected physical transition from "
    "photoelectric- to Compton-dominated attenuation. This is a SINGLE data "
    "point and is NOT sufficient to establish general reliability. Predictions "
    "for any custom or untested material should be treated as a physics-"
    "motivated hypothesis, not a confirmed/validated prediction, until "
    "independently checked against real experimental data for that material."
)


if __name__ == "__main__":
    from .materials_database import DATABASE
    results = run_validation(DATABASE)
    for row in results:
        print(f"{row['material']}: real DEF = {row['real_DEF']} ({row['context']})")
        print(f"  Best match: {row['best_match_energy_keV']} keV, error = {row['best_match_error_pct']:+.1f}%")
        print(f"  Reference: {row['reference']}")
