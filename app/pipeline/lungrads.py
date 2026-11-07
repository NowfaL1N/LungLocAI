"""Size/type-based category in the style of ACR Lung-RADS v2022 (baseline screening).

This is a simplified, automatic approximation meant to help a clinician triage the AI findings.
It uses the volume-equivalent diameter and the solid fraction measured by the pipeline; the final
category must be assigned by a radiologist on the original images.
"""

MANAGEMENT = {
    "1": "Negative / benign-appearing (e.g. calcified). Continue routine annual low-dose CT screening.",
    "2": "Benign appearance or behaviour. Continue annual low-dose CT screening in 12 months.",
    "3": "Probably benign. Low-dose CT follow-up in 6 months is suggested.",
    "4A": "Suspicious. Low-dose CT in 3 months; PET/CT may be considered if there is a solid component >= 8 mm.",
    "4B": "Very suspicious. Diagnostic chest CT, PET/CT and/or tissue sampling depending on the "
          "probability of malignancy and comorbidities. Refer for multidisciplinary review.",
}


def categorize(diameter_mm: float, solid_fraction: float, nodule_type: str) -> str:
    d = float(diameter_mm)
    if nodule_type == "calcified":
        return "1"
    if nodule_type == "ground-glass":
        return "2" if d < 30 else "3"
    if nodule_type == "part-solid":
        solid_d = d * max(solid_fraction, 0) ** (1 / 3)   # equivalent diameter of the solid part
        if d < 6:
            return "2"
        if solid_d < 6:
            return "3"
        if solid_d < 8:
            return "4A"
        return "4B"
    # solid
    if d < 6:
        return "2"
    if d < 8:
        return "3"
    if d < 15:
        return "4A"
    return "4B"


def overall_priority(cat: str, p_malignant) -> str:
    """Combine the size category with the XGBoost probability for a single triage flag."""
    try:
        p = float(p_malignant)
    except (TypeError, ValueError):
        p = float("nan")
    high_cat = cat in ("4A", "4B")
    if p == p:  # not NaN
        if p >= 0.7 or (cat == "4B" and p >= 0.3):
            return "Urgent review"
        if p >= 0.3 or high_cat:
            return "Short-term follow-up"
        return "Routine"
    return "Urgent review" if cat == "4B" else ("Short-term follow-up" if cat in ("3", "4A") else "Routine")
