"""
Graphical abstract generator -- a schematic "mechanism diagram" showing how
a 2D-material radiosensitizer is expected to affect a cancer cell under
radiation, at conventional vs. FLASH dose rates. This is the kind of
illustrative figure journals call a "graphical abstract" -- built here from
matplotlib shapes, driven by the tool's own computed numbers (DEF, escape
probability, O2 depletion), not stock art.

IMPORTANT: this is a CONCEPTUAL/SCHEMATIC illustration of the proposed
mechanism, not a rendering of any real microscopy or simulation image. Label
it this way in any thesis or presentation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge
import matplotlib.path as mpath

CELL_COLOR = "#f2c6b0"
NUCLEUS_COLOR = "#8e6fa6"
MATERIAL_COLOR = "#4fc3f7"
RADIATION_COLOR = "#ffd54f"
DAMAGE_COLOR = "#e53935"
OXYGEN_COLOR = "#42a5f5"


def _draw_cell(ax, cx, cy, radius=1.5, n_flakes=5, flake_positions=None, seed=0):
    rng = np.random.default_rng(seed)
    ax.add_patch(Circle((cx, cy), radius, facecolor=CELL_COLOR, edgecolor="#b5651d",
                         linewidth=1.5, alpha=0.9, zorder=1))
    ax.add_patch(Circle((cx, cy), radius * 0.45, facecolor=NUCLEUS_COLOR,
                         edgecolor="#4a3466", linewidth=1.2, alpha=0.9, zorder=2))
    # DNA squiggle inside nucleus
    t = np.linspace(0, 4 * np.pi, 100)
    dna_x = cx + 0.25 * np.sin(t) * (radius * 0.3 / 1.5)
    dna_y = cy - radius * 0.3 + t / (4 * np.pi) * (radius * 0.6)
    ax.plot(dna_x, dna_y, color="white", linewidth=1.5, zorder=3)

    if flake_positions is None:
        angles = rng.uniform(0, 2 * np.pi, n_flakes)
        radii = rng.uniform(radius * 0.55, radius * 0.85, n_flakes)
        flake_positions = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(angles, radii)]
    for fx, fy in flake_positions:
        ax.add_patch(Rectangle((fx - 0.12, fy - 0.03), 0.24, 0.06, angle=rng.uniform(0, 180),
                                facecolor=MATERIAL_COLOR, edgecolor="#01579b",
                                linewidth=0.8, zorder=4))
    return flake_positions


def _draw_radiation_arrows(ax, cx, cy, radius, n_arrows=6, y_start_offset=2.5, color=RADIATION_COLOR):
    for i in range(n_arrows):
        x = cx - radius * 0.8 + (i / (n_arrows - 1)) * radius * 1.6
        arrow = FancyArrowPatch((x, cy + y_start_offset), (x, cy + radius * 0.9),
                                 arrowstyle="-|>", mutation_scale=14, color=color, linewidth=2, zorder=5)
        ax.add_patch(arrow)


def _draw_damage_burst(ax, x, y, size=0.12, color=DAMAGE_COLOR, n_points=8):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    for a in angles:
        ax.plot([x, x + size * np.cos(a)], [y, y + size * np.sin(a)],
                color=color, linewidth=1.3, zorder=6)


def draw_mechanism_diagram(material_name: str, peak_DEF: float, P_escape: float,
                            O2_depletion_conventional: float, O2_depletion_FLASH: float):
    """
    Two-panel graphical abstract: conventional dose rate (left) vs FLASH (right),
    showing the same cell/material, radiation arrows, DNA-damage bursts scaled by
    DEF, and oxygen dots (depleted more in the FLASH panel).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    n_damage_sites = int(np.clip(round(peak_DEF), 1, 8))

    for ax, label, o2_depl, dose_label in zip(
        axes, ["Conventional dose rate", "FLASH dose rate"],
        [O2_depletion_conventional, O2_depletion_FLASH],
        ["~0.03 Gy/s", "\u2265 40 Gy/s"]
    ):
        cx, cy = 0, 0
        flake_positions = _draw_cell(ax, cx, cy, radius=1.5, n_flakes=5, seed=1)
        _draw_radiation_arrows(ax, cx, cy, radius=1.5)

        # Damage bursts near material flakes, count scaled by DEF
        for i, (fx, fy) in enumerate(flake_positions[:n_damage_sites]):
            _draw_damage_burst(ax, fx, fy, size=0.15)

        # Oxygen dots in surrounding tissue (outside cell), fewer if depleted.
        # NOTE: the visual reduction is deliberately AMPLIFIED (sqrt scaling)
        # so a real ~1-15% depletion is visible as a dot-count difference --
        # the PRINTED percentage above each panel is the accurate value; the
        # dot count is a visual aid only, not a literal 1:1 representation.
        rng = np.random.default_rng(2)
        n_o2_total = 24
        visual_depletion = o2_depl ** 0.3  # amplify small fractions for visibility
        n_o2_shown = max(2, int(n_o2_total * (1 - visual_depletion)))
        for _ in range(n_o2_shown):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(1.7, 2.3)
            ox, oy = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
            ax.add_patch(Circle((ox, oy), 0.045, facecolor=OXYGEN_COLOR,
                                 edgecolor="#0d47a1", linewidth=0.5, zorder=3))

        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.8, 3.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{label}  ({dose_label})\nO2 depletion: {o2_depl:.1%}", fontsize=11)

    fig.suptitle(f"{material_name} radiosensitization -- conceptual mechanism "
                 f"(predicted peak DEF = {peak_DEF:.2f}x, escape probability = {P_escape:.2f})",
                 fontsize=12)

    # Legend (shared)
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MATERIAL_COLOR,
                   markeredgecolor='#01579b', markersize=10, label='2D material flake'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=OXYGEN_COLOR,
                   markeredgecolor='#0d47a1', markersize=8, label='Dissolved O2'),
        plt.Line2D([0], [0], color=DAMAGE_COLOR, linewidth=2, label='Localized DNA damage (schematic)'),
        plt.Line2D([0], [0], color=RADIATION_COLOR, linewidth=2, marker='>', label='Incident radiation'),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.text(0.5, 1.0, "CONCEPTUAL/SCHEMATIC MECHANISM DIAGRAM -- not a microscopy image "
             "or simulation rendering", ha="center", fontsize=8, color="red", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


if __name__ == "__main__":
    fig = draw_mechanism_diagram("MoS2", peak_DEF=1.73, P_escape=1.0,
                                  O2_depletion_conventional=0.007, O2_depletion_FLASH=0.124)
    fig.savefig("/tmp/test_mechanism.png", dpi=130)
    print("Saved /tmp/test_mechanism.png")
