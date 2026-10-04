"""
Expanded 2D-materials database with literature status flags.

Every entry below is backed by a real, findable publication (checked via
literature search, October 2026). Where a real experimental dose-enhancement
or radiosensitization number exists in the literature, it is recorded in
`experimental_def_or_ser` with its source, so the tool's own PREDICTED DEF can
be shown side-by-side with what has actually been measured -- this is the
"compare against experiment" feature.

IMPORTANT, read before using this for a paper:
- `experimental_def_or_ser` numbers come from different cell lines, beam
  energies, and endpoints (DEF = physical dose enhancement; SER = biological
  sensitization enhancement ratio -- these are NOT the same quantity and
  should never be silently treated as interchangeable). Each entry states
  which one was measured.
- "status" flags whether radiosensitization for that material has been
  studied EXPERIMENTALLY, THEORETICALLY/COMPUTATIONALLY, or NOT YET STUDIED
  (to the best of this search). This should be re-verified independently
  before being quoted in a thesis or paper -- literature moves fast.
- Composition/density/thickness values are still approximate (placeholders
  pending exact Materials Project entries, per the original proposal).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from .materials import compute_zeff, Z_TABLE

# Extend the element table with what we need for the new entries
Z_TABLE.update({"N": 7, "B": 5, "Bi": 83, "Se": 34, "W": 74, "Nb": 41})


@dataclass
class MaterialEntry:
    name: str
    composition: Dict[str, float]
    density_g_cm3: float
    thickness_nm_range: tuple
    molar_mass_g_mol: float
    status: str                                  # "experimental" / "theoretical" / "unstudied"
    experimental_def_or_ser: Optional[str]        # e.g. "DEF up to 2.5" or "SER 1.3" or None
    experimental_context: Optional[str]           # cell line / beam / conditions
    reference: Optional[str]                      # citation string
    reference_url: Optional[str]
    zeff: float = field(init=False)

    def __post_init__(self):
        self.zeff = compute_zeff(self.composition)


DATABASE = {

    "Ti3C2Tx_MXene": MaterialEntry(
        name="Ti3C2Tx MXene",
        composition={"Ti": 3, "C": 2, "O": 2},
        density_g_cm3=3.9,
        thickness_nm_range=(1.0, 5.0),
        molar_mass_g_mol=3 * 47.87 + 2 * 12.01 + 2 * 16.00,
        status="experimental",
        experimental_def_or_ser="DEF up to 2.5 (physical dose enhancement)",
        experimental_context="Human soft-tissue sarcoma cells (HT1080), kV X-ray, 8 Gy, "
                              "no toxicity to healthy fibroblasts at matched dose",
        reference="Rafetseder et al., 'X-ray radio-enhancement by Ti3C2Tx MXenes in soft "
                  "tissue sarcoma', Biomaterials Science, 2023, 11(24), 7826.",
        reference_url="https://pubs.rsc.org/en/content/articlelanding/2023/bm/d3bm00607g",
    ),

    "Black_Phosphorus": MaterialEntry(
        name="Black phosphorus (BP) nanosheets",
        composition={"P": 1},
        density_g_cm3=2.69,
        thickness_nm_range=(0.53, 5.0),
        molar_mass_g_mol=30.97,
        status="experimental",
        experimental_def_or_ser="Radiosensitization confirmed (quantitative DEF/SER not "
                                 "directly comparable across the cited studies)",
        experimental_context="Black phosphorus quantum dots enhanced IR-induced apoptosis in "
                              "human renal cell carcinoma (RCC) cells via DNA-PKcs inhibition; "
                              "separately, coordination-modified BP nanosensitizers (RGD-Ir@BP) "
                              "inhibited nasopharyngeal carcinoma tumor growth in vivo under X-ray",
        reference="Lang et al., Cells, 2022, 11(10), 1651 (DOI 10.3390/cells11101651); "
                  "and ACS Nano 2021, DOI 10.1021/acsnano.0c09454.",
        reference_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9139844/",
    ),

    "Graphene_Oxide": MaterialEntry(
        name="Graphene oxide (GO) nanosheets",
        composition={"C": 10, "O": 3, "H": 4},   # approximate GO stoichiometry
        density_g_cm3=1.8,
        thickness_nm_range=(0.7, 3.0),
        molar_mass_g_mol=10 * 12.01 + 3 * 16.00 + 4 * 1.008,
        status="experimental",
        experimental_def_or_ser="Radiosensitization confirmed qualitatively (apoptosis "
                                 "increase, Bcl-2 reduction); quantitative DEF not reported",
        experimental_context="Nasopharyngeal carcinoma cells (C666-1, HK-1); comparative "
                              "study also found Ti3C2 MXene outperforms GO as a radiosensitizer "
                              "in breast cancer cells under clinical X-ray",
        reference="Radiosensitization of Nasopharyngeal Carcinoma by Graphene Oxide "
                  "Nanosheets, PubMed ID 36913208, 2023.",
        reference_url="https://pubmed.ncbi.nlm.nih.gov/36913208/",
        # NOTE (discovered during tool testing, worth keeping): GO's Zeff (~6.6) is
        # BELOW tissue Zeff (~7.4), so this tool's Module 1 (photoelectric contrast)
        # predicts DEF < 1 for GO -- i.e. physically it should NOT enhance photon
        # dose. Yet GO is experimentally confirmed to radiosensitize cells. This
        # mismatch is informative, not a bug: it implies GO's real radiosensitizing
        # mechanism is likely ROS/redox chemistry (consistent with its documented
        # role in promoting reactive oxygen species and apoptosis), NOT photoelectric
        # dose enhancement -- a mechanism this tool does not model. Flag this
        # explicitly whenever GO's predicted DEF is shown, rather than presenting a
        # contradiction silently.
    ),

    "MoS2": MaterialEntry(
        name="Molybdenum disulfide (MoS2)",
        composition={"Mo": 1, "S": 2},
        density_g_cm3=5.06,
        thickness_nm_range=(0.65, 5.0),
        molar_mass_g_mol=95.95 + 2 * 32.06,
        status="theoretical",
        experimental_def_or_ser=None,
        experimental_context="Standalone MoS2 radiosensitization (as the primary dose-"
                              "enhancing agent) was NOT found as a direct experimental study "
                              "in this search; MoS2 has been used as a CT contrast/photothermal "
                              "agent, and combined with HfO2 (MoS2/HfO2-Dextran) where HfO2 "
                              "supplies the dose enhancement. Treat standalone MoS2 as a "
                              "theoretical/computational candidate pending direct validation.",
        reference="Advances in nanoparticle-based radiotherapy for cancer treatment, "
                  "ScienceDirect review, 2024 (MoS2/HfO2-Dextran combination).",
        reference_url="https://www.sciencedirect.com/science/article/pii/S2589004224028293",
    ),

    "hBN": MaterialEntry(
        name="Hexagonal boron nitride (h-BN)",
        composition={"B": 1, "N": 1},
        density_g_cm3=2.1,
        thickness_nm_range=(0.33, 3.0),
        molar_mass_g_mol=10.81 + 14.01,
        status="unstudied",
        experimental_def_or_ser=None,
        experimental_context="No radiosensitization study found in this search. Low Zeff "
                              "(light elements only) makes it a weak candidate for photon dose "
                              "enhancement on physics grounds alone -- included here as a "
                              "negative-control / low-priority reference point.",
        reference=None,
        reference_url=None,
    ),

    "Bi2Se3": MaterialEntry(
        name="Bismuth selenide (Bi2Se3) nanosheets",
        composition={"Bi": 2, "Se": 3},
        density_g_cm3=7.7,
        thickness_nm_range=(1.0, 6.0),
        molar_mass_g_mol=2 * 208.98 + 3 * 78.97,
        status="theoretical",
        experimental_def_or_ser=None,
        experimental_context="Bismuth-based nanoplates (not necessarily this exact 2D form) "
                              "have shown dose enhancement in vitro/in vivo in the broader "
                              "nanoparticle-radiosensitization literature; Bi2Se3 specifically "
                              "as a 2D radiosensitizer was not directly found in this search -- "
                              "flagged as a physics-motivated candidate (very high Zeff) pending "
                              "direct validation.",
        reference="Sarcoma cell-specific radiation sensitization by titanate scrolled "
                  "nanosheets (bismuth-based NP context), PMC10853196.",
        reference_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10853196/",
    ),
}


def list_by_status(status: str):
    return {k: v for k, v in DATABASE.items() if v.status == status}


if __name__ == "__main__":
    for key, m in DATABASE.items():
        print(f"{key:20s} Zeff={m.zeff:6.2f}  status={m.status:12s} "
              f"exp_data={'yes' if m.experimental_def_or_ser else 'no'}")
