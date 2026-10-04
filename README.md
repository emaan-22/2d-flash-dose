# 2D-FLASH-Dose — Web App

A physics-based screening tool for 2D-material radiosensitizers under
conventional and FLASH dose rates. Runs entirely **offline, with no API key**
required — all material data is a local, cited, curated dataset.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser tab (usually `http://localhost:8501`) with the full app.

## What's in the app (6 tabs)

1. **Material Library** — preset materials (Ti3C2Tx MXene, black phosphorus,
   graphene oxide, MoS2, h-BN, Bi2Se3), each tagged as
   `EXPERIMENTALLY STUDIED`, `THEORETICAL/COMPUTATIONAL ONLY`, or `NOT YET
   STUDIED`, with real citations (verified via literature search, Oct 2026)
   and — where available — the actual published experimental dose-enhancement
   or sensitization-enhancement-ratio number, so you can compare the tool's
   prediction against real data.

2. **Build Custom Material** — pick any elements and stoichiometry, no
   restriction to the preset list. Computes effective atomic number (Zeff)
   from first principles (standard radiation-physics mixture rule), fully
   offline.

3. **DEF Calculator** — spatial Dose Enhancement Factor curve for any
   material (preset or custom), with live sliders for photon energy, flake
   thickness, and lateral size.

4. **FLASH / Oxygen Kinetics** — oxygen-depletion sensitivity vs. dose rate,
   showing where the FLASH-sparing mechanism kicks in.

5. **Next-Step Recommendations** — rule-based suggestions for which DFT
   functional, dispersion treatment, and flake dimensionality to pursue next
   for a given material, with the physics reasoning behind each suggestion.

6. **About & Limitations** — what the tool does and does not do, stated
   plainly.

## Important finding baked into this version

During testing, the tool's own photon-interaction model (Module 1) predicts
**DEF < 1.0 for graphene oxide** (its effective atomic number is below
tissue's) — yet GO has real, published experimental radiosensitization
evidence. This is flagged explicitly in the app rather than hidden: it implies
GO's real radiosensitizing mechanism is most likely ROS/redox chemistry, not
photoelectric dose enhancement — something this tool does not model. This is
a genuine, citable observation about the tool's scope, not a bug to apologize
for.

Likewise, **Ti3C2Tx MXene already has real experimental dose-enhancement data**
(DEF up to 2.5 in soft-tissue sarcoma cells, Biomaterials Science 2023) — so
the novelty of this project is specifically the **FLASH-rate coupling**
(Module 3), not "2D materials as radiosensitizers" in general, which is
already an active experimental field. State the novelty claim this way in
any paper or thesis.

## Known limitations (read before publishing anything from this tool)

- **Module 1 (photon interaction)** uses a single-exponent Z-scaling power
  law calibrated at only two points — it tends to overestimate peak DEF
  (predicting >1000x for some materials vs. the real ~1-3x range seen in
  nanoparticle literature). Treat absolute DEF numbers as a *relative*
  ranking tool, not literal predictions, until replaced with direct NIST XCOM
  interpolation.
- **No full Monte Carlo transport** (Geant4/TOPAS) — this is a fast
  screening tool, explicitly positioned as a pre-screening step, not a
  replacement.
- **No biological outcome / cell-survival prediction.**
- **Module 3's G-value and replenishment rate constant are literature
  order-of-magnitude placeholders**, not calibrated to a specific tissue or
  experimental system.
- **Composition/density/thickness for preset materials are approximate**,
  pending exact Materials Project entries.
- **The Recommendations tab is a rule-based heuristic engine**, not a
  material-specific DFT benchmark — always check for existing literature on
  the exact material before committing significant compute time.
- **No live Materials Project / external API integration** — this was a
  deliberate design choice (no API key required, works fully offline). The
  preset library is a curated subset, not the full Materials Project catalog.

## File structure

```
app.py                          <- Streamlit web app (run this)
dose2d/
  materials.py                  <- Zeff calculation + original 3-material set
  materials_database.py         <- Expanded library with citations/status flags
  material_builder.py           <- Custom material builder (any element/stoichiometry)
  photon_interaction.py         <- Module 1: photoelectric contrast
  electron_transport.py         <- Module 2: 2D-geometry electron escape
  oxygen_kinetics.py             <- Module 3: oxygen-depletion ODE
  def_model.py                  <- Combines Modules 1+2 into DEF(r)
  recommender.py                <- DFT functional / environment / dimensionality rules
run_screening.py                <- Original command-line batch screening script
visualize_2d_map.py             <- 2D macroscopic DEF heatmap generator
requirements.txt
```

## Next steps (unchanged from the original proposal)

- [ ] Replace Module 1's power-law with direct NIST XCOM interpolation.
- [ ] Validate against the gold/gadolinium experimental data, and now also
      against the real Ti3C2Tx MXene DEF-2.5 data point found above.
- [ ] Replace placeholder material properties with exact Materials Project
      entries (would need a free API key to fetch automatically -- out of
      scope for this offline version by design).
- [ ] Calibrate Module 3's G-value and k_repl against published FLASH
      radiobiology data.
- [ ] Investigate non-photoelectric mechanisms (ROS/redox) for materials
      like GO where Module 1 and experimental evidence disagree.
