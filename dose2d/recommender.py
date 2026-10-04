"""
Recommendation engine: if a material looks promising from the DEF screening,
what should an experimentalist/computational researcher actually do next?

This is a RULE-BASED expert system built on standard, well-established
computational-materials-science heuristics (not a trained model, not a
database lookup) -- it encodes commonly-cited guidance from the DFT
methodology literature. It is meant to save a researcher's first afternoon of
"which functional do I even start with" -- it is NOT a substitute for reading
functional-specific benchmark papers for the exact material in question.

Heuristics encoded (with the reasoning you can show a supervisor):
- Pure GGA (PBE) is a reasonable, cheap starting point for structure
  relaxation and total energies, but is well known to underestimate band
  gaps -- relevant if the project cares about electronic structure /
  photoelectric-threshold behavior.
- Materials containing 3d/4d/5d transition metals with localized d/f
  electrons often need a Hubbard-U correction (DFT+U) to avoid
  self-interaction-error artifacts in the electronic structure.
- Hybrid functionals (HSE06) give much better band gaps than PBE at
  significantly higher computational cost -- recommended once a candidate
  clears the cheap PBE screening stage.
- 2D materials are held together (if multilayer) or interact with substrates
  via van der Waals forces, which plain PBE does NOT capture -- a dispersion
  correction (e.g. DFT-D3, or a vdW-inclusive functional) is recommended
  whenever thickness > 1 monolayer or substrate interactions matter.
- Heavy elements (Z > ~50) benefit from relativistic treatment (scalar-
  relativistic pseudopotentials at minimum; spin-orbit coupling if band
  topology or heavy-atom spectroscopy matters).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Recommendation:
    functional: List[str]
    functional_reasoning: str
    environment: List[str]
    environment_reasoning: str
    dimensionality: str
    dimensionality_reasoning: str
    caveats: List[str]


HEAVY_Z_THRESHOLD = 50
TRANSITION_METAL_Z_RANGES = [(21, 30), (39, 48), (72, 80)]  # 3d, 4d, 5d series


def _is_transition_metal(z: int) -> bool:
    return any(lo <= z <= hi for lo, hi in TRANSITION_METAL_Z_RANGES)


def recommend(material_zeff: float, composition: dict, thickness_nm_range: tuple,
               atomic_numbers: dict, known_oxidation_sensitive: bool = False) -> Recommendation:
    """
    atomic_numbers: dict of element_symbol -> Z, for the elements in `composition`
                     (so the engine can check for transition metals / heavy atoms
                     without re-deriving them).
    """
    has_transition_metal = any(_is_transition_metal(z) for z in atomic_numbers.values())
    has_heavy_element = any(z > HEAVY_Z_THRESHOLD for z in atomic_numbers.values())
    is_multilayer = thickness_nm_range[1] > 1.0  # realistic range extends beyond monolayer

    # --- Functional recommendation ---
    functional = ["PBE (GGA) -- cheap first pass for structure relaxation and total energy"]
    reasoning_parts = ["PBE is a reasonable, low-cost starting point for geometry and total energy."]

    if has_transition_metal:
        functional.append("PBE+U (Hubbard-U correction on the transition-metal d-orbitals)")
        reasoning_parts.append(
            "The composition includes a 3d/4d/5d transition metal; plain GGA is known to "
            "poorly describe localized d-electron self-interaction, so a +U correction "
            "(fit or taken from literature for that element) is recommended before trusting "
            "electronic-structure results."
        )
    functional.append("HSE06 (hybrid functional) -- for accurate band gap, once PBE screening looks promising")
    reasoning_parts.append(
        "PBE systematically underestimates band gaps; if the project needs the band gap "
        "specifically (relevant to photoelectric threshold behavior), re-run the promising "
        "candidate(s) with HSE06 despite its higher cost."
    )
    if has_heavy_element:
        functional.append("Scalar-relativistic pseudopotentials (add spin-orbit coupling if band "
                           "topology or heavy-atom core-level behavior matters)")
        reasoning_parts.append(
            f"The composition includes an element with Z > {HEAVY_Z_THRESHOLD}; relativistic "
            "effects on core/valence states become non-negligible and should be included."
        )

    # --- Environment / dispersion recommendation ---
    environment = []
    env_reasoning_parts = []
    if is_multilayer:
        environment.append("Dispersion correction required: DFT-D3 (Grimme) or a vdW-inclusive "
                            "functional (e.g. optB88-vdW, rVV10)")
        env_reasoning_parts.append(
            "The realistic thickness range extends beyond a single monolayer, so interlayer "
            "van der Waals binding must be captured -- plain PBE does not include dispersion "
            "forces and will give unreliable interlayer spacing/binding energy."
        )
    else:
        environment.append("Dispersion correction still recommended if modeling substrate "
                            "interaction or solvent/aqueous dispersion stability")
        env_reasoning_parts.append(
            "Even a true monolayer will interact with its surrounding medium (substrate, "
            "solvent, biological fluid) via van der Waals forces relevant to colloidal "
            "stability -- include a dispersion correction if that interaction is being modeled."
        )
    if known_oxidation_sensitive:
        environment.append("Explicit surface-oxidation / termination-group modeling "
                            "(e.g. -O, -F, -OH terminations for MXenes) recommended")
        env_reasoning_parts.append(
            "This material class is known to be oxidation-sensitive in aqueous/biological "
            "media (flagged from literature) -- bare-lattice calculations alone will not "
            "represent the material as it actually exists in a tumor environment; model "
            "realistic surface terminations."
        )

    # --- Dimensionality recommendation ---
    t_min, t_max = thickness_nm_range
    if t_min < 1.0:
        dimensionality = (f"Start with monolayer ({t_min:.2f} nm) for maximum surface-area-to-"
                           f"volume ratio and electron escape efficiency; compare against "
                           f"few-layer ({t_max:.1f} nm) for higher absorbing mass per flake")
        dim_reasoning = (
            "Thinner flakes maximize the fraction of generated secondary electrons that "
            "escape into surrounding tissue (Module 2 of this tool), but carry less "
            "absorbing mass per flake -- the true optimum depends on achievable tumor "
            "loading concentration, which should be tested experimentally across this range."
        )
    else:
        dimensionality = f"Few-layer ({t_min:.1f}-{t_max:.1f} nm) is the realistic starting range for this material"
        dim_reasoning = (
            "This material's practically synthesizable thickness range does not extend to a "
            "true monolayer -- optimize within the realistic range rather than assuming "
            "monolayer behavior."
        )

    caveats = [
        "These are STARTING POINTS based on general DFT methodology heuristics, not "
        "material-specific benchmarks -- check for existing DFT studies of this exact "
        "material before committing significant compute time.",
        "This tool does not run DFT itself; it only recommends which functional/settings "
        "an experimentalist or computational collaborator should reach for next.",
    ]

    return Recommendation(
        functional=functional,
        functional_reasoning=" ".join(reasoning_parts),
        environment=environment,
        environment_reasoning=" ".join(env_reasoning_parts),
        dimensionality=dimensionality,
        dimensionality_reasoning=dim_reasoning,
        caveats=caveats,
    )


if __name__ == "__main__":
    from .material_builder import ATOMIC_NUMBER
    rec = recommend(
        material_zeff=19.61,
        composition={"Ti": 3, "C": 2, "O": 2},
        thickness_nm_range=(1.0, 5.0),
        atomic_numbers={"Ti": ATOMIC_NUMBER["Ti"], "C": ATOMIC_NUMBER["C"], "O": ATOMIC_NUMBER["O"]},
        known_oxidation_sensitive=True,
    )
    print("Functional recommendations:", rec.functional)
    print("Reasoning:", rec.functional_reasoning)
    print()
    print("Environment recommendations:", rec.environment)
    print("Reasoning:", rec.environment_reasoning)
    print()
    print("Dimensionality:", rec.dimensionality)
    print("Reasoning:", rec.dimensionality_reasoning)
