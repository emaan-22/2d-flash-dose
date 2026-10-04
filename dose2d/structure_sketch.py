"""
Structure visualization module (v2): molecular sketch WITH bonds, plus a
side-view "chip" (layer-stack) rendering showing real thickness.

IMPORTANT, read first: these are ILLUSTRATIVE SCHEMATICS. Atom positions are
generated from a simple 2D lattice-tiling algorithm (not a DFT-relaxed
structure); bonds are drawn between any two atoms within a cutoff distance
(a standard schematic convention), not computed from actual bond orders or
quantum-mechanical bonding. This gives a presentation-quality molecular
picture, but it is NOT a substitute for a real relaxed structure from
VASP/Quantum ESPRESSO or a Materials Project entry. Label every figure this
way when using it in a thesis or paper.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from scipy.spatial import cKDTree

ELEMENT_COLOR = {
    "Ti": "#8a9ba8", "C": "#404040", "O": "#e74c3c", "Mo": "#5c6bc0",
    "S": "#f9c74f", "P": "#8e44ad", "H": "#ecf0f1", "N": "#2980b9",
    "B": "#ff6f61", "W": "#455a64", "Se": "#ff9800", "Bi": "#7b1fa2",
    "Nb": "#00838f", "Au": "#ffd700", "Gd": "#90ee90",
}
ELEMENT_RADIUS = {   # relative covalent-radius-like scaling, illustrative
    "Ti": 0.32, "C": 0.20, "O": 0.22, "Mo": 0.34, "S": 0.26, "P": 0.26,
    "H": 0.12, "N": 0.22, "B": 0.22, "W": 0.36, "Se": 0.28, "Bi": 0.36,
    "Nb": 0.33, "Au": 0.34, "Gd": 0.34,
}
DEFAULT_COLOR = "#78909c"
DEFAULT_RADIUS = 0.24


def classify_lattice(composition: dict) -> str:
    elements = set(composition.keys())
    if "Ti" in elements and "C" in elements:
        return "mxene_slab"
    if len(elements) == 2:
        metals = {"Mo", "W", "Nb", "Ti", "Zr", "Hf"}
        chalcogens = {"S", "Se", "Te"}
        if elements & metals and elements & chalcogens:
            return "tmd_honeycomb"
    if elements == {"P"}:
        return "puckered"
    if "C" in elements and "O" in elements:
        return "graphene_oxide_like"
    if elements == {"B", "N"}:
        return "hexagonal_bn"
    return "generic_hexagonal"


def _generate_positions(lattice_type: str, composition: dict, rows=5, cols=5):
    """Returns (positions Nx2 array, element_labels list)."""
    elements = list(composition.keys())
    a = 1.0
    positions, labels = [], []

    if lattice_type == "tmd_honeycomb":
        metal_el = [e for e in elements if e not in {"S", "Se", "Te"}][0]
        chal_el = [e for e in elements if e in {"S", "Se", "Te"}][0]
        dx, dy = a * 1.5, a * np.sqrt(3)
        for i in range(rows):
            for j in range(cols):
                x0 = j * dx
                y0 = i * dy + (dy / 2 if j % 2 else 0)
                positions.append((x0, y0)); labels.append(metal_el)
                positions.append((x0 + a * 0.85, y0 + 0.1)); labels.append(chal_el)

    elif lattice_type == "mxene_slab":
        dx, dy = a * 1.6, a * np.sqrt(3)
        for i in range(rows):
            for j in range(cols):
                x0, y0 = j * dx, i * dy + (dy / 2 if j % 2 else 0)
                positions.append((x0, y0)); labels.append("Ti")
                positions.append((x0 + 0.8, y0 + 0.3)); labels.append("C")
                if "O" in elements:
                    positions.append((x0 + 0.3, y0 + 0.9)); labels.append("O")

    elif lattice_type == "puckered":
        dx, dy = a * 1.1, a * 1.3
        for i in range(rows):
            for j in range(cols):
                x0 = j * dx
                y0 = i * dy + 0.3 * np.sin(j * 1.3)
                positions.append((x0, y0)); labels.append("P")

    elif lattice_type == "graphene_oxide_like":
        dx, dy = a * 1.5, a * np.sqrt(3)
        k = 0
        for i in range(rows):
            for j in range(cols):
                x0 = j * dx
                y0 = i * dy + (dy / 2 if j % 2 else 0)
                el = "O" if k % 5 == 0 else "C"
                positions.append((x0, y0)); labels.append(el)
                k += 1

    else:
        dx, dy = a * 1.5, a * np.sqrt(3)
        k = 0
        for i in range(rows):
            for j in range(cols):
                x0 = j * dx
                y0 = i * dy + (dy / 2 if j % 2 else 0)
                positions.append((x0, y0)); labels.append(elements[k % len(elements)])
                k += 1

    return np.array(positions), labels


def draw_molecular_structure(composition: dict, material_name: str, ax=None, bond_cutoff=1.3):
    """Top-down view: atoms as spheres, bonds as lines between near neighbors."""
    lattice_type = classify_lattice(composition)
    positions, labels = _generate_positions(lattice_type, composition)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        created_fig = True

    # Bonds: connect any two atoms within bond_cutoff (schematic convention)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=bond_cutoff)
    for i, j in pairs:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        ax.plot([x1, x2], [y1, y2], color="#999999", linewidth=1.2, zorder=1, alpha=0.8)

    seen_elements = set()
    for (x, y), el in zip(positions, labels):
        r = ELEMENT_RADIUS.get(el, DEFAULT_RADIUS)
        color = ELEMENT_COLOR.get(el, DEFAULT_COLOR)
        ax.add_patch(Circle((x, y), r, color=color, ec="black", lw=0.7, zorder=2))
        seen_elements.add(el)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=ELEMENT_COLOR.get(el, DEFAULT_COLOR),
               markeredgecolor='black', markersize=10, label=el)
               for el in sorted(seen_elements)]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{material_name}  (top-down view)\n{lattice_type.replace('_', ' ')}", fontsize=10)
    ax.text(0.02, 0.02, "SCHEMATIC -- not a DFT-relaxed structure",
            transform=ax.transAxes, fontsize=7, color="red", style="italic")

    if created_fig:
        plt.tight_layout()
        return fig
    return ax


def draw_chip_sideview(composition: dict, material_name: str, thickness_nm: float,
                        n_layers: int = None, ax=None):
    """
    Side-view 'chip' rendering: a stack of thin layers showing real thickness,
    with a scale bar/arrow. n_layers is estimated from thickness if not given
    (assuming ~0.6-0.7 nm per monolayer, typical for 2D materials).
    """
    if n_layers is None:
        n_layers = max(1, round(thickness_nm / 0.65))

    elements = list(composition.keys())
    layer_color = ELEMENT_COLOR.get(elements[0], DEFAULT_COLOR)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        created_fig = True

    width = 4.0
    layer_h = 0.25
    gap = 0.08
    y0 = 0
    for i in range(min(n_layers, 20)):  # cap drawn layers for readability
        ax.add_patch(Rectangle((0, y0), width, layer_h, facecolor=layer_color,
                                edgecolor="black", linewidth=0.8, alpha=0.85))
        y0 += layer_h + gap
    total_h = y0 - gap

    # Scale bar / thickness arrow
    arrow = FancyArrowPatch((width + 0.6, 0), (width + 0.6, total_h),
                             arrowstyle="<->", mutation_scale=15, color="black", linewidth=1.3)
    ax.add_patch(arrow)
    ax.text(width + 0.9, total_h / 2, f"{thickness_nm:.2f} nm\n(~{n_layers} layers)",
            fontsize=9, va="center")

    ax.set_xlim(-0.5, width + 2.5)
    ax.set_ylim(-0.5, max(total_h + 0.5, 2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{material_name} -- side view (layer stack)", fontsize=10)
    ax.text(0.02, 0.02, "SCHEMATIC -- illustrative layer count, not measured",
            transform=ax.transAxes, fontsize=7, color="red", style="italic")

    if created_fig:
        plt.tight_layout()
        return fig
    return ax


def draw_combined_structure_panel(composition: dict, material_name: str, thickness_nm: float):
    """Side-by-side: molecular top-down view + chip side-view, one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    draw_molecular_structure(composition, material_name, ax=axes[0])
    draw_chip_sideview(composition, material_name, thickness_nm, ax=axes[1])
    plt.tight_layout()
    return fig


# Backward-compatible alias (v1 name)
def draw_schematic(composition, material_name, ax=None):
    return draw_molecular_structure(composition, material_name, ax=ax)


if __name__ == "__main__":
    fig = draw_combined_structure_panel({"Mo": 1, "S": 2}, "MoS2", thickness_nm=2.8)
    fig.savefig("/tmp/test_combined.png", dpi=120)
    print("Saved /tmp/test_combined.png")
