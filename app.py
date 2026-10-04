"""
2D-FLASH-Dose -- Web App (v3: narrated, validated, with molecular visuals
and graphical abstracts)
Run with: streamlit run app.py
No API key required -- all data is local/offline.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")

from dose2d.materials_database import DATABASE
from dose2d.material_builder import CustomMaterial, ATOMIC_NUMBER
from dose2d.def_model import compute_DEF_profile, Z_TISSUE_EFF
from dose2d import oxygen_kinetics as ok
from dose2d.recommender import recommend
from dose2d.radiobiology import (expected_treatment_outcome, ALPHA_BETA_PRESETS,
                                  ALPHA_DEFAULT_PER_GY, survival_fraction)
from dose2d.structure_sketch import draw_combined_structure_panel, classify_lattice
from dose2d.graphical_abstract import draw_mechanism_diagram

def safe_draw(draw_fn, *args, **kwargs):
    """Wraps a matplotlib-figure-producing function so a rendering edge case
    (e.g. an unusual custom composition) shows a friendly message instead of
    crashing the whole app."""
    try:
        return draw_fn(*args, **kwargs)
    except Exception as e:
        st.warning(f"Could not render this figure for the current material/parameters "
                   f"({type(e).__name__}: {e}). Try a different thickness or composition. "
                   f"The rest of the app is unaffected.")
        return None
from dose2d.validation import run_validation, VALIDATION_SUMMARY_TEXT, GOLD_NP_DEF_RANGE_REFERENCE
from dose2d.narratives import (narrate_def_plot, narrate_oxygen_plot, narrate_survival_plot,
                                narrate_comparison_plot, narrate_validation_plot)

st.set_page_config(page_title="2D-FLASH-Dose", page_icon="\u269b\ufe0f", layout="wide")

st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #e8e4d8; }
.stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 600; }
div[data-testid="stMetricValue"] { font-size: 24px; }
.badge-exp { background-color: #1f6f43; color: white; padding: 2px 10px; border-radius: 10px; font-size: 12px; }
.badge-theo { background-color: #a3690f; color: white; padding: 2px 10px; border-radius: 10px; font-size: 12px; }
.badge-unstudied { background-color: #555555; color: white; padding: 2px 10px; border-radius: 10px; font-size: 12px; }
.caveat-box { background-color: #2b2410; border-left: 4px solid #c9a227; padding: 10px 14px; border-radius: 4px; margin: 6px 0; }
.good-box { background-color: #122b1c; border-left: 4px solid #2e9e5b; padding: 10px 14px; border-radius: 4px; margin: 6px 0; }
.narrative-box { background-color: #13202e; border-left: 4px solid #4fc3f7; padding: 10px 14px; border-radius: 4px; margin: 10px 0; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

if "custom_materials" not in st.session_state:
    st.session_state.custom_materials = {}

# ---------------------------------------------------------------------------
# HEADER + INTRODUCTION
# ---------------------------------------------------------------------------
st.title("\u269b\ufe0f 2D-FLASH-Dose")
st.caption("Physics-based screening tool connecting computational physics to dose enhancement "
           "in cancer radiotherapy. Runs fully offline, no API key. Not a trained AI/ML model.")

with st.expander("\U0001F4D6 Introduction -- what this tool does and why it matters", expanded=True):
    st.markdown("""
**The problem:** Radiotherapy kills cancer cells with ionizing radiation, but the same dose
damages healthy tissue. High-atomic-number (high-Z) materials placed in a tumor can locally
amplify radiation dose via the photoelectric effect, sparing tissue further away. Separately,
**FLASH radiotherapy** -- delivering the same dose at ultra-high speed (\u226540 Gy/s instead of
~0.03 Gy/s) -- has shown normal-tissue-sparing effects in its own right.

**The physics-to-medicine connection this tool explores:** Two-dimensional nanomaterials
(MXenes, transition-metal dichalcogenides, black phosphorus) have very different electron
transport and absorption physics compared to the spherical nanoparticles (gold, gadolinium)
most radiosensitization research has focused on. This tool computes, from first-principles
physics formulas (not machine learning), how a given 2D material's atomic composition and
geometry should enhance radiation dose, and how that interacts with FLASH dose-rate physics
-- turning atomic-scale material properties into a predicted, citable, cancer-treatment-relevant
number.

**How to use this tool:** Browse the Material Library or build a custom material, then move
through the tabs to see its predicted dose enhancement, FLASH behavior, expected treatment
performance, molecular structure, and a full research roadmap -- each with the real formulas
and literature citations behind the numbers, and honest validation against the (limited) real
experimental data available.
    """)

tabs = st.tabs(["\U0001F4DA Material Library", "\U0001F527 Build Custom Material",
                "\U0001F4CA DEF Calculator", "\u26A1 FLASH / Oxygen Kinetics",
                "\U0001FA7A Expected Treatment Performance", "\U0001F9EC Molecular Structure",
                "\U0001F3A8 Mechanism Diagram", "\u2705 Validation",
                "\U0001F9ED Next-Step Roadmap", "\U0001F4C8 Compare All Materials",
                "\u2139\ufe0f About & Limitations"])

STATUS_BADGE = {
    "experimental": '<span class="badge-exp">EXPERIMENTALLY STUDIED</span>',
    "theoretical": '<span class="badge-theo">THEORETICAL / COMPUTATIONAL ONLY</span>',
    "unstudied": '<span class="badge-unstudied">NOT YET STUDIED (found in this search)</span>',
}

def narrative(text):
    st.markdown(f'<div class="narrative-box">\U0001F4AC {text}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 1: Material Library
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Preset material library")
    st.write("Every entry is backed by a real, cited publication check (October 2026 search).")
    for key, mat in DATABASE.items():
        with st.expander(f"**{mat.name}**  \u2014  Zeff = {mat.zeff:.2f}", expanded=False):
            st.markdown(STATUS_BADGE[mat.status], unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Density:** {mat.density_g_cm3} g/cm\u00b3")
                st.markdown(f"**Thickness range:** {mat.thickness_nm_range[0]}\u2013{mat.thickness_nm_range[1]} nm")
                st.markdown(f"**Molar mass:** {mat.molar_mass_g_mol:.1f} g/mol")
            with c2:
                st.markdown(f"**Experimental result:** {mat.experimental_def_or_ser or 'none found'}")
                if mat.experimental_context:
                    st.markdown(f"**Context:** {mat.experimental_context}")
            if mat.reference:
                st.markdown(f"**Reference:** {mat.reference}")
                if mat.reference_url:
                    st.markdown(f"[Open source]({mat.reference_url})")

# ---------------------------------------------------------------------------
# TAB 2: Custom Material Builder
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Build a custom material from atoms")
    col1, col2 = st.columns([2, 1])
    with col1:
        elements = st.multiselect("Elements in the formula unit", sorted(ATOMIC_NUMBER.keys()),
                                   default=["W", "S"])
        counts = {}
        if elements:
            cols = st.columns(min(len(elements), 4))
            for i, el in enumerate(elements):
                with cols[i % len(cols)]:
                    counts[el] = st.number_input(f"{el} atoms", min_value=1, value=1, step=1,
                                                  key=f"count_{el}")
    with col2:
        density = st.number_input("Density (g/cm\u00b3)", min_value=0.1, value=5.0, step=0.1)
        t_min = st.number_input("Min thickness (nm)", min_value=0.1, value=0.7, step=0.1)
        t_max = st.number_input("Max thickness (nm)", min_value=0.1, value=5.0, step=0.1)
        name = st.text_input("Material name (optional)", value="")

    st.markdown("**Optional additional properties**")
    c3, c4, c5 = st.columns(3)
    with c3:
        band_gap = st.number_input("Known band gap (eV), 0 = unknown", min_value=0.0, value=0.0, step=0.1)
    with c4:
        oxidation_sensitive = st.checkbox("Oxidation-sensitive in biological media?")
    with c5:
        synthesis_method = st.selectbox("Synthesis method (if known)",
                                         ["Unknown", "Liquid-phase exfoliation", "CVD",
                                          "Mechanical exfoliation", "MAX-phase etching (MXenes)",
                                          "Hydrothermal/solvothermal"])

    if st.button("Build material", type="primary"):
        if not elements:
            st.error("Select at least one element.")
        elif t_min > t_max:
            st.error(f"Min thickness ({t_min} nm) is greater than max thickness ({t_max} nm). "
                     f"Please fix before building.")
        else:
            try:
                mat = CustomMaterial(counts, density_g_cm3=density,
                                      thickness_nm_range=(t_min, t_max), name=name or None)
                mat.band_gap_eV = band_gap if band_gap > 0 else None
                mat.oxidation_sensitive = oxidation_sensitive
                mat.synthesis_method = synthesis_method
                st.session_state.custom_materials[mat.name] = mat
                st.success(f"Built **{mat.name}**  \u2014  Zeff = {mat.zeff:.2f}")
            except ValueError as e:
                st.error(str(e))

    if st.session_state.custom_materials:
        st.markdown("---")
        for nm, mat in st.session_state.custom_materials.items():
            st.write(f"- {nm}: Zeff={mat.zeff:.2f}, density={mat.density_g_cm3} g/cm\u00b3, "
                     f"thickness {mat.thickness_nm_range} nm")

def pick_material(key_prefix: str):
    source = st.radio("Material source", ["Preset library", "My custom materials"],
                       horizontal=True, key=f"{key_prefix}_source")
    if source == "Preset library":
        sel = st.selectbox("Choose material", list(DATABASE.keys()), key=f"{key_prefix}_preset")
        mat = DATABASE[sel]
    else:
        if not st.session_state.custom_materials:
            st.info("No custom materials built yet \u2014 go to 'Build Custom Material' tab first.")
            return None, None
        sel = st.selectbox("Choose your material", list(st.session_state.custom_materials.keys()),
                            key=f"{key_prefix}_custom")
        mat = st.session_state.custom_materials[sel]
    atomic_numbers = {el: ATOMIC_NUMBER.get(el, 0) for el in mat.composition}
    return mat, atomic_numbers

# ---------------------------------------------------------------------------
# TAB 3: DEF Calculator
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Dose Enhancement Factor calculator  (photoelectric + Compton physics)")
    mat, _ = pick_material("def")
    if mat is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            energy_keV = st.slider("Photon energy (keV)", 20, 500, 100)
        with c2:
            t_lo, t_hi = mat.thickness_nm_range
            thickness_nm = st.slider("Flake thickness (nm)", float(t_lo), float(t_hi), float((t_lo+t_hi)/2))
        with c3:
            lateral_nm = st.slider("Lateral flake size (nm)", 10, 2000, 200)

        r, DEF, meta = compute_DEF_profile(mat, energy_keV, thickness_nm, lateral_nm, r_max_nm=3000)

        m1, m2, m3 = st.columns(3)
        m1.metric("Peak DEF (at flake surface)", f"{DEF[0]:.2f}x")
        m2.metric("Photon contrast", f"{meta['contrast']:.2f}x")
        m3.metric("Escape probability", f"{meta['P_escape']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r, y=DEF, mode="lines", name=mat.name,
                                  line=dict(color="#4fc3f7", width=3)))
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="No enhancement")
        fig.update_layout(title=f"DEF(r) for {mat.name} at {energy_keV} keV",
                           xaxis_title="Distance from flake surface, r (nm)",
                           yaxis_title="Dose Enhancement Factor", template="plotly_dark", height=420)
        st.plotly_chart(fig, width='stretch')
        narrative(narrate_def_plot(mat.name, DEF[0], energy_keV, meta['contrast'],
                                    meta['P_escape'], meta['lambda_dep_nm']))

        if DEF[0] < 1.0:
            st.markdown('<div class="caveat-box">\u26A0\ufe0f Predicted DEF below 1.0 -- see '
                        'About tab for the graphene-oxide mechanism note.</div>', unsafe_allow_html=True)
        if getattr(mat, "experimental_def_or_ser", None):
            st.markdown(f'<div class="good-box">\u2713 Real experimental data: '
                        f'{mat.experimental_def_or_ser} ({mat.experimental_context})</div>',
                        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 4: FLASH / Oxygen Kinetics
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Oxygen-depletion sensitivity (Michaelis-Menten ROD, cited G0)")
    c1, c2, c3 = st.columns(3)
    with c1:
        pO2 = st.slider("Baseline tissue pO2 (\u03bcM)", 5.0, 60.0, 30.0)
    with c2:
        total_dose = st.slider("Total dose (Gy)", 1.0, 30.0, 10.0)
    with c3:
        G0 = st.slider("G0 (\u03bcM/Gy) \u2014 literature range 0.25-0.75", 0.1, 1.0, 0.4, step=0.05)

    dose_rates = np.logspace(-2, np.log10(200), 50)
    fracs = [ok.fractional_depletion(pO2, r, total_dose, G0_uM_per_Gy=G0) for r in dose_rates]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dose_rates, y=fracs, mode="lines+markers",
                               line=dict(color="#ff7043", width=3)))
    fig2.add_vline(x=40, line_dash="dash", line_color="red", annotation_text="FLASH threshold")
    fig2.update_layout(title="Fractional O2 depletion vs. dose rate", xaxis_title="Dose rate (Gy/s)",
                        xaxis_type="log", yaxis_title="Fractional O2 depletion",
                        template="plotly_dark", height=420)
    st.plotly_chart(fig2, width='stretch')
    narrative(narrate_oxygen_plot(pO2, total_dose, G0))
    st.markdown('<div class="caveat-box">\u26A0\ufe0f Per a 2026 review (Frontiers in Physics), '
                'ROD alone may not fully explain FLASH sparing -- this models one citable '
                'contributor, not the complete mechanism.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 5: Expected Treatment Performance
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Expected cancer-treatment performance (Linear-Quadratic model)")
    mat, _ = pick_material("rb")
    if mat is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            energy_keV = st.slider("Photon energy (keV)", 20, 500, 100, key="rb_E")
        with c2:
            dose_Gy = st.slider("Dose per fraction (Gy)", 0.5, 10.0, 2.0, key="rb_D")
        with c3:
            ab_choice = st.selectbox("Tumor/tissue type", list(ALPHA_BETA_PRESETS.keys()))

        t_mid = sum(mat.thickness_nm_range) / 2
        r, DEF, meta = compute_DEF_profile(mat, energy_keV, t_mid, 200.0, r_max_nm=2000)
        ab = ALPHA_BETA_PRESETS[ab_choice]
        beta = ALPHA_DEFAULT_PER_GY / ab
        outcome = expected_treatment_outcome(dose_Gy, ALPHA_DEFAULT_PER_GY, beta, peak_DEF=DEF[0])

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated SER", f"{outcome['SER_estimate']:.2f}x")
        m2.metric("Survival WITHOUT sensitizer", f"{outcome['survival_fraction_without_sensitizer']:.1%}")
        m3.metric("Survival WITH sensitizer", f"{outcome['survival_fraction_with_sensitizer']:.1%}")

        doses = np.linspace(0, 10, 50)
        S_wo = survival_fraction(doses, ALPHA_DEFAULT_PER_GY, beta)
        S_w = survival_fraction(doses * outcome['SER_estimate'], ALPHA_DEFAULT_PER_GY, beta)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=doses, y=S_wo, name="Without sensitizer", line=dict(color="grey")))
        fig3.add_trace(go.Scatter(x=doses, y=S_w, name=f"With {mat.name}", line=dict(color="#4fc3f7")))
        fig3.update_layout(title="Cell survival curve (Linear-Quadratic model)",
                            xaxis_title="Dose (Gy)", yaxis_title="Surviving fraction",
                            yaxis_type="log", template="plotly_dark", height=420)
        st.plotly_chart(fig3, width='stretch')
        narrative(narrate_survival_plot(mat.name, outcome['SER_estimate'], dose_Gy,
                                         outcome['relative_cell_kill_improvement']*100, ab))
        st.markdown('<div class="caveat-box">\u26A0\ufe0f SER is a deliberately dampened, '
                    'illustrative mapping from physical DEF -- not a validated biological '
                    'prediction.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 6: Molecular Structure
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Molecular structure (bonds + thickness chip view)")
    mat, _ = pick_material("mol")
    if mat is not None:
        t_mid = sum(mat.thickness_nm_range) / 2
        thickness_nm = st.slider("Thickness to render (nm)", float(mat.thickness_nm_range[0]),
                                  float(mat.thickness_nm_range[1]), float(t_mid), key="mol_t")
        lattice = classify_lattice(mat.composition)
        st.write(f"Classified lattice type: **{lattice.replace('_', ' ')}**")
        fig = safe_draw(draw_combined_structure_panel, mat.composition, mat.name, thickness_nm)
        if fig is not None:
            st.pyplot(fig)
        narrative(f"{mat.name} is classified as a {lattice.replace('_', ' ')} lattice based on its "
                  f"composition. The left panel shows a top-down bonded view (schematic connectivity, "
                  f"not DFT-relaxed); the right panel shows the selected thickness ({thickness_nm:.2f} nm) "
                  f"as an approximate layer stack, assuming ~0.65 nm per monolayer.")

# ---------------------------------------------------------------------------
# TAB 7: Mechanism Diagram (graphical abstract)
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Mechanism diagram (graphical abstract style)")
    mat, _ = pick_material("mech")
    if mat is not None:
        energy_keV = st.slider("Photon energy (keV)", 20, 500, 100, key="mech_E")
        t_mid = sum(mat.thickness_nm_range) / 2
        r, DEF, meta = compute_DEF_profile(mat, energy_keV, t_mid, 200.0, r_max_nm=2000)
        frac_conv = ok.fractional_depletion(30.0, 0.03, 10.0)
        frac_flash = ok.fractional_depletion(30.0, 100.0, 10.0)
        fig = safe_draw(draw_mechanism_diagram, mat.name, DEF[0], meta['P_escape'], frac_conv, frac_flash)
        if fig is not None:
            st.pyplot(fig)
        narrative(f"This conceptual diagram shows {mat.name} flakes (blue) near a cell nucleus, with "
                  f"radiation (yellow arrows) producing localized damage (red bursts) scaled by the "
                  f"predicted DEF ({DEF[0]:.2f}x). Dissolved oxygen (blue dots) is visibly reduced in "
                  f"the FLASH panel, reflecting the {frac_flash:.1%} vs {frac_conv:.1%} depletion "
                  f"difference computed in the FLASH/Oxygen Kinetics tab.")
        st.markdown('<div class="caveat-box">\u26A0\ufe0f This is a CONCEPTUAL/SCHEMATIC mechanism '
                    'illustration, not a microscopy image or simulation rendering. Oxygen dot-count '
                    'reduction is visually amplified for clarity -- the printed percentages are the '
                    'accurate values.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 8: Validation
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Validation against real experimental data")
    st.markdown(f'<div class="caveat-box">\u26A0\ufe0f {VALIDATION_SUMMARY_TEXT}</div>',
                unsafe_allow_html=True)
    results = run_validation(DATABASE)
    for row in results:
        st.markdown(f"### {row['material']}")
        st.write(f"Real experimental value: **{row['real_DEF']}** ({row['context']})")
        st.write(f"Reference: {row['reference']}")

        energies = list(row['energy_sweep'].keys())
        predicted = [row['energy_sweep'][e]['predicted_DEF'] for e in energies]
        errors = [row['energy_sweep'][e]['error_pct'] for e in energies]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=energies, y=predicted, mode="lines+markers",
                                   name="Tool prediction", line=dict(color="#4fc3f7")))
        fig4.add_hline(y=row['real_DEF'], line_dash="dash", line_color="#2e9e5b",
                       annotation_text=f"Real experimental value ({row['real_DEF']})")
        fig4.update_layout(title=f"{row['material']}: predicted vs real DEF across energy",
                            xaxis_title="Photon energy (keV)", yaxis_title="DEF",
                            template="plotly_dark", height=400)
        st.plotly_chart(fig4, width='stretch')
        narrative(narrate_validation_plot(row['best_match_energy_keV'], row['best_match_error_pct']))

    st.markdown("---")
    st.markdown("### Additional context: broader magnitude sanity-check")
    lo, hi = GOLD_NP_DEF_RANGE_REFERENCE["range"]
    st.write(f"Real gold-nanoparticle DEF range across published kV X-ray spectra: "
             f"**{lo}\u2013{hi}x** ({GOLD_NP_DEF_RANGE_REFERENCE['context']})")
    st.caption(GOLD_NP_DEF_RANGE_REFERENCE["note"])
    st.caption(f"Reference: {GOLD_NP_DEF_RANGE_REFERENCE['reference']}")

# ---------------------------------------------------------------------------
# TAB 9: Next-Step Roadmap
# ---------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Full research roadmap for this material")
    mat, atomic_numbers = pick_material("road")
    if mat is not None:
        t_mid = sum(mat.thickness_nm_range) / 2
        r, DEF, meta = compute_DEF_profile(mat, 100.0, t_mid, 200.0, r_max_nm=2000)
        oxidation_flag = getattr(mat, "oxidation_sensitive", False) or ("MXene" in mat.name or "phosphorus" in mat.name.lower())
        rec = recommend(mat.zeff, mat.composition, mat.thickness_nm_range, atomic_numbers,
                         known_oxidation_sensitive=oxidation_flag)

        st.markdown(f"## {mat.name}")
        st.markdown("#### 1. Material identity")
        st.write(f"Zeff = {mat.zeff:.2f} | Density = {mat.density_g_cm3} g/cm\u00b3 | "
                 f"Thickness range = {mat.thickness_nm_range} nm")
        roadmap_fig = safe_draw(draw_combined_structure_panel, mat.composition, mat.name, t_mid)
        if roadmap_fig is not None:
            st.pyplot(roadmap_fig)

        st.markdown("#### 2. Physics prediction (at 100 keV, mid-range thickness)")
        st.write(f"Predicted peak DEF: **{DEF[0]:.2f}x**")
        if hasattr(mat, "status"):
            st.markdown(STATUS_BADGE.get(mat.status, ""), unsafe_allow_html=True)
            if getattr(mat, "experimental_def_or_ser", None):
                st.write(f"Real experimental result: {mat.experimental_def_or_ser}")

        st.markdown("#### 3. FLASH behavior")
        frac_conv = ok.fractional_depletion(30.0, 0.03, 10.0)
        frac_flash = ok.fractional_depletion(30.0, 100.0, 10.0)
        st.write(f"O2 depletion: conventional = {frac_conv:.1%}, FLASH = {frac_flash:.1%}")

        st.markdown("#### 4. Recommended next steps")
        for f in rec.functional:
            st.markdown(f"- {f}")
        for e in rec.environment:
            st.markdown(f"- {e}")
        st.markdown(f"**Dimensionality:** {rec.dimensionality}")

        st.markdown("#### 5. Honest confidence assessment")
        if DEF[0] < 1.0:
            st.markdown('<div class="caveat-box">Predicted DEF < 1 -- mechanism may be non-photoelectric.</div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="caveat-box">See Validation tab -- this tool has only n=1 '
                    f'real experimental validation point overall.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 10: Compare All Materials
# ---------------------------------------------------------------------------
with tabs[9]:
    st.subheader("Compare predicted DEF across all materials")
    energy_cmp = st.slider("Photon energy for comparison (keV)", 20, 500, 100, key="cmp_E")
    all_mats = dict(DATABASE)
    all_mats.update(st.session_state.custom_materials)
    names, preds, exp_flags = [], [], []
    for key, mat in all_mats.items():
        t_mid = sum(mat.thickness_nm_range) / 2
        r, DEF, meta = compute_DEF_profile(mat, energy_cmp, t_mid, 200.0, r_max_nm=500)
        names.append(mat.name); preds.append(DEF[0])
        exp_flags.append(getattr(mat, "status", "custom") == "experimental")
    colors = ["#2e9e5b" if f else "#4fc3f7" for f in exp_flags]
    fig5 = go.Figure(go.Bar(x=names, y=preds, marker_color=colors))
    fig5.add_hline(y=1.0, line_dash="dash", line_color="grey")
    fig5.update_layout(title=f"Predicted peak DEF at {energy_cmp} keV (green = has real experimental data)",
                        yaxis_title="Predicted peak DEF", template="plotly_dark", height=450)
    st.plotly_chart(fig5, width='stretch')
    if preds:
        best_i, worst_i = int(np.argmax(preds)), int(np.argmin(preds))
        narrative(narrate_comparison_plot(names[best_i], preds[best_i], names[worst_i], preds[worst_i], energy_cmp))

# ---------------------------------------------------------------------------
# TAB 11: About & Limitations
# ---------------------------------------------------------------------------
with tabs[10]:
    st.subheader("What this tool does and does not do")
    st.markdown("""
**Physics-based tool, not AI/ML.** Every number comes from an explicit formula or cited value.

**v3 adds:** narrated captions on every plot, molecular structure with real bonds, thickness
chip-view, graphical-abstract mechanism diagrams, and a built-in Validation tab comparing
predictions against the one real experimental data point found in literature.

**GO/low-Zeff mechanism note:** materials with Zeff below tissue's predict DEF < 1 (Module 1),
yet graphene oxide has real experimental radiosensitization evidence -- implying its mechanism
is likely ROS/redox chemistry, not photoelectric dose enhancement, which this tool does not model.

**Still not resolved:** no full Monte Carlo transport; no real DFT-relaxed structures (Molecular
Structure tab is schematic only); SER-from-DEF mapping is illustrative; only n=1 direct
experimental validation point exists (see Validation tab) -- treat all custom-material
predictions as unvalidated hypotheses pending real comparison data.
    """)
    st.markdown("---")
    st.markdown("**References:**")
    for key, mat in DATABASE.items():
        if mat.reference:
            st.markdown(f"- {mat.reference}")
    st.markdown("- Espinosa-Rodriguez et al., Phys. Med. Biol. 2023 (arXiv:2310.01281) -- Module 3 G0/k_ROD.")
    st.markdown("- Review of oxygen measurement and FLASH mechanisms, Frontiers in Physics, 2026.")
