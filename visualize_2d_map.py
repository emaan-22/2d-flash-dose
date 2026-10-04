"""
2D spatial DEF heatmap -- matRad-panel-(e)-style visualization
(proposal Section 7.3 output, extended per supervisor's visual-comparison request)

IMPORTANT PHYSICAL NOTE ON SCALE (read before interpreting the figure):

The DEF(r) decay length (lambda_dep, the electron deposition range in tissue)
is on the order of ~1-10 micrometers (see def_model.py / electron_transport.py).
A tumor cross-section, by contrast, is on the order of centimeters. That is a
factor of ~1000-10,000 difference in scale.

This means: IF a 2D-material radiosensitizer is dispersed roughly uniformly
throughout the tumor volume (flake-to-flake spacing << lambda_dep), the
enhancement fields from neighbouring flakes overlap and the tumor interior is
essentially uniformly enhanced at ~peak_DEF, while OUTSIDE the tumor boundary
the enhancement drops to baseline within a few micrometers -- i.e. at the
scale of a whole-tumor image, the transition looks like a sharp step, not a
gradual gradient. This is NOT a resolution artifact of the plot -- it is the
correct physical picture at this length scale, and is exactly why 2D-material
dose enhancement is described as "localized" in the proposal.

This script therefore renders:
  Panel (a): the macroscopic DEF map (tumor = ~peak_DEF, healthy tissue = 1x),
             analogous to matRad's panel (e) dose-distribution image.
  Panel (b): a zoomed-in inset at the tumor boundary showing the REAL
             micrometer-scale decay from def_model.compute_DEF_profile(),
             so the sharp step in panel (a) isn't mistaken for "no physics".

LIMITATION: the tumor shape here is an idealized ellipse, not real patient CT
anatomy -- this is an illustrative/schematic map, not a treatment-planning-
grade dose distribution (see proposal Section 7.4 scope limitations; matRad-
style CT-based dose maps require a full treatment planning system, which is
out of scope for this tool).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from dose2d.materials import MATERIALS
from dose2d.def_model import compute_DEF_profile

OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Choose a material + geometry
# ---------------------------------------------------------------------------
MATERIAL_KEY = "MoS2"
mat = MATERIALS[MATERIAL_KEY]
THICKNESS_NM = float(np.mean(mat.thickness_nm_range))
ENERGY_KEV = 100.0
LATERAL_NM = 200.0

# Get the real DEF(r) profile (micrometer-scale physics) for the inset
r_nm, DEF_r, meta = compute_DEF_profile(mat, ENERGY_KEV, THICKNESS_NM, LATERAL_NM,
                                         r_max_nm=20000, n_points=400)
peak_DEF = float(DEF_r[0])
lambda_dep_um = meta["lambda_dep_nm"] / 1000.0

# ---------------------------------------------------------------------------
# Panel (a): macroscopic 2D DEF map over an idealized elliptical tumor
# ---------------------------------------------------------------------------
GRID_MM = 60.0          # cross-section spans -30mm to +30mm in x and y
N_GRID = 300
tumor_a_mm, tumor_b_mm = 14.0, 10.0   # semi-axes of the elliptical tumor (PTV)

x = np.linspace(-GRID_MM / 2, GRID_MM / 2, N_GRID)
y = np.linspace(-GRID_MM / 2, GRID_MM / 2, N_GRID)
X, Y = np.meshgrid(x, y)

# Signed "distance" in normalized ellipse coordinates: <1 inside tumor
ellipse_dist = (X / tumor_a_mm) ** 2 + (Y / tumor_b_mm) ** 2

# At this macroscopic scale, the um-scale DEF decay (lambda_dep_um) is far
# smaller than 1 grid pixel (GRID_MM/N_GRID = 0.2 mm = 200 um), so the
# physically-correct rendering is a step function at the tumor boundary
# (see module docstring). A very slight smoothing (~1 pixel) is applied only
# to avoid hard-edge aliasing in the image, NOT to represent real um-scale
# physics at this zoom level.
DEF_map = np.where(ellipse_dist <= 1.0, peak_DEF, 1.0)
from scipy.ndimage import gaussian_filter
DEF_map_smoothed = gaussian_filter(DEF_map, sigma=0.8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# --- Panel (a): macroscopic map ---
ax = axes[0]
im = ax.imshow(DEF_map_smoothed, extent=[-GRID_MM/2, GRID_MM/2, -GRID_MM/2, GRID_MM/2],
               origin="lower", cmap="turbo", vmin=1, vmax=peak_DEF)
tumor_outline = Ellipse((0, 0), width=2*tumor_a_mm, height=2*tumor_b_mm,
                         fill=False, edgecolor="white", linewidth=1.8, linestyle="-")
ax.add_patch(tumor_outline)
ax.plot([], [], color="white", label="Tumor boundary (PTV)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(f"(a) {MATERIAL_KEY}-loaded tumor: macroscopic DEF map")
ax.legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Dose Enhancement Factor")

# --- Panel (b): zoomed micrometer-scale boundary inset ---
ax2 = axes[1]
r_um = r_nm / 1000.0
ax2.plot(r_um, DEF_r, color="#d1495b", linewidth=2)
ax2.axhline(1.0, color="grey", linestyle="--", linewidth=1)
ax2.set_xlabel("Distance from tumor/flake boundary, r (\u03bcm)")
ax2.set_ylabel("Dose Enhancement Factor, DEF(r)")
ax2.set_title(f"(b) Real boundary decay (\u03bb$_{{dep}}$ \u2248 {lambda_dep_um:.1f} \u03bcm)")
ax2.set_xlim(0, min(20, r_um.max()))
ax2.grid(alpha=0.3)

fig.suptitle(f"{MATERIAL_KEY} radiosensitization: macroscopic map vs. true microscopic decay length",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
outpath = os.path.join(OUTDIR, "def_spatial_map_with_boundary_inset.png")
plt.savefig(outpath, dpi=160)
plt.close()
print("Saved:", outpath)
print(f"Peak DEF used: {peak_DEF:.1f}   lambda_dep: {lambda_dep_um:.2f} um   "
      f"(grid pixel size: {GRID_MM/N_GRID*1000:.0f} um -- {GRID_MM/N_GRID*1000/lambda_dep_um:.0f}x "
      f"larger than the decay length, hence the sharp boundary in panel a)")
