"""
2D-FLASH-Dose -- Phase I screening run (proposal Section 9.3, Phase I scope:
DEF(r) module for 2D geometries under conventional flux, plus Module-3
oxygen-depletion reporting alongside it).

Produces the three outputs named in proposal Section 7.3:
  1. DEF-vs-distance spatial dose maps        -> output/def_vs_distance.png
  2. DEF/O2-depletion sensitivity vs dose rate -> output/dose_rate_sensitivity.png
  3. Comparative screening table              -> output/screening_table.csv

Run with:  python run_screening.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

from dose2d.materials import MATERIALS
from dose2d.def_model import compute_DEF_profile, screen_materials
from dose2d import oxygen_kinetics as ok

OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

ENERGY_KEV = 100.0        # representative orthovoltage/kV beam energy
LATERAL_NM = 200.0        # representative flake lateral size
PO2_BASELINE_UM = 30.0    # representative tissue baseline oxygen
TOTAL_DOSE_GY = 10.0      # representative single-fraction dose

# ---------------------------------------------------------------------------
# Output 1: DEF-vs-distance spatial dose maps, one curve per material
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for key, mat in MATERIALS.items():
    t_mid = float(np.mean(mat.thickness_nm_range))
    r, DEF, meta = compute_DEF_profile(mat, ENERGY_KEV, t_mid, LATERAL_NM)
    plt.plot(r, DEF, label=f"{key} (Zeff={mat.zeff:.1f}, t={t_mid:.1f} nm)")

plt.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="No enhancement (DEF=1)")
plt.xlabel("Distance from flake surface, r (nm)")
plt.ylabel("Dose Enhancement Factor, DEF(r)")
plt.title(f"DEF vs. distance at {ENERGY_KEV:.0f} keV (conventional dose rate)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "def_vs_distance.png"), dpi=150)
plt.close()
print("Saved:", os.path.join(OUTDIR, "def_vs_distance.png"))

# ---------------------------------------------------------------------------
# Output 2: sensitivity of oxygen-depletion fraction across the
# conventional-to-FLASH dose-rate range (Module 3, reported per Section 5A
# as a separate contributor, not multiplied into DEF)
# ---------------------------------------------------------------------------
dose_rates = np.logspace(np.log10(0.01), np.log10(200), 40)  # Gy/s
depletion_fracs = [
    ok.fractional_depletion(PO2_BASELINE_UM, rate, TOTAL_DOSE_GY)
    for rate in dose_rates
]

plt.figure(figsize=(7, 5))
plt.semilogx(dose_rates, depletion_fracs, marker="o", markersize=3)
plt.axvline(40, color="red", linestyle="--", linewidth=1, label="FLASH threshold (40 Gy/s)")
plt.xlabel("Dose rate, Ḋ (Gy/s)")
plt.ylabel("Fractional O\u2082 depletion over pulse (Module 3)")
plt.title(f"O\u2082-depletion sensitivity vs. dose rate ({TOTAL_DOSE_GY:.0f} Gy total dose)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "dose_rate_sensitivity.png"), dpi=150)
plt.close()
print("Saved:", os.path.join(OUTDIR, "dose_rate_sensitivity.png"))

# ---------------------------------------------------------------------------
# Output 3: comparative screening table across candidate materials
# ---------------------------------------------------------------------------
rows = screen_materials(MATERIALS, ENERGY_KEV, LATERAL_NM,
                         total_dose_Gy=TOTAL_DOSE_GY, pO2_baseline_uM=PO2_BASELINE_UM)

csv_path = os.path.join(OUTDIR, "screening_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("Saved:", csv_path)

print("\n--- Screening table preview ---")
for row in rows:
    print(row)
