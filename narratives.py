"""
Dynamic narrative caption generator -- produces 2-3 sentence plain-language
explanations for each plot, based on the ACTUAL computed numbers for the
material/parameters currently selected (not static boilerplate text).
"""


def narrate_def_plot(material_name, DEF0, energy_keV, contrast, P_escape, lambda_dep_nm):
    pe_dominant = energy_keV < 80
    regime = ("photoelectric-dominated (high-Z advantage is strongest here)" if pe_dominant
              else "Compton-influenced (material/tissue contrast is diluted at this energy)")
    return (
        f"At {energy_keV:.0f} keV, {material_name} is predicted to enhance local dose by "
        f"{DEF0:.2f}x at the flake surface, decaying to background (1x) within about "
        f"{lambda_dep_nm/1000:.1f} \u03bcm. This energy sits in the {regime} regime. "
        f"Roughly {P_escape:.0%} of secondary electrons generated in the flake are predicted "
        f"to escape into surrounding tissue, where they deposit this enhanced dose."
    )


def narrate_oxygen_plot(pO2, total_dose, G0, dose_rate_flash=100.0, dose_rate_conv=0.03):
    from . import oxygen_kinetics as ok
    frac_conv = ok.fractional_depletion(pO2, dose_rate_conv, total_dose, G0_uM_per_Gy=G0)
    frac_flash = ok.fractional_depletion(pO2, dose_rate_flash, total_dose, G0_uM_per_Gy=G0)
    return (
        f"At a baseline tissue oxygen level of {pO2:.0f} \u03bcM and {total_dose:.0f} Gy total dose, "
        f"conventional delivery (~0.03 Gy/s) depletes only {frac_conv:.1%} of local oxygen, because "
        f"vascular replenishment keeps pace with radiolytic consumption. FLASH delivery "
        f"(\u226540 Gy/s) depletes {frac_flash:.1%}, since consumption outpaces replenishment within "
        f"the ultra-short pulse -- this is the proposed (though not fully proven, see caveats) "
        f"basis for FLASH's normal-tissue-sparing effect."
    )


def narrate_survival_plot(material_name, SER, dose_Gy, improvement_pct, ab_ratio):
    return (
        f"Using a damped DEF-to-SER mapping, {material_name} is estimated to give a "
        f"sensitization enhancement ratio of {SER:.2f}x for this tumor type (alpha/beta = "
        f"{ab_ratio:.1f} Gy). At {dose_Gy:.1f} Gy per fraction, this corresponds to roughly a "
        f"{improvement_pct:.1f}% relative improvement in cell kill compared to radiation alone. "
        f"This is an illustrative estimate, not a validated clinical prediction (see caveats)."
    )


def narrate_comparison_plot(best_material, best_DEF, worst_material, worst_DEF, energy_keV):
    return (
        f"Across the material library at {energy_keV:.0f} keV, {best_material} shows the "
        f"highest predicted dose enhancement ({best_DEF:.2f}x), while {worst_material} shows "
        f"the lowest ({worst_DEF:.2f}x) -- materials with Zeff below tissue's can even show "
        f"predicted DEF under 1.0, which does not necessarily mean they are ineffective "
        f"radiosensitizers (see the graphene oxide mechanism note in the About tab)."
    )


def narrate_validation_plot(best_match_energy, best_match_error_pct):
    direction = "overestimates" if best_match_error_pct > 0 else "underestimates"
    return (
        f"The tool's closest match to real experimental data occurs at {best_match_energy:.0f} keV, "
        f"where it {direction} the measured value by {abs(best_match_error_pct):.1f}%. This is the "
        f"ONLY direct quantitative validation point available (n=1) -- treat this as an indication "
        f"of the tool's regime of applicability (low-kV, high-Z materials), not proof of general "
        f"reliability across all materials or energies."
    )
