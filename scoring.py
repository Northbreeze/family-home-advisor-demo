from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


NOISE_PENALTY_MAP = {"Low": 0, "Medium": 18, "High": 40, "Unknown": 12}
DISPLAY_COLUMNS = [
    "Address",
    "City",
    "Price",
    "Bedrooms",
    "Bathrooms",
    "Size",
    "price_per_sqft",
    "listing_change_status",
    "is_new_since_last_refresh",
    "final_school",
    "final_fraser_score",
    "noise_risk",
    "noise_model_risk",
    "noise_override_note",
    "noise_context_note",
    "noise_verification_needed",
    "distance_to_highway_m",
    "distance_to_major_road_m",
    "client_status",
    "yard_playability",
    "yard_noise",
    "layout_fit",
    "photo_review_status",
    "photo_yard_playability",
    "photo_yard_type",
    "photo_flatness",
    "photo_fenced",
    "photo_privacy",
    "ai_yard_score",
    "ai_layout_score",
    "ai_privacy_score",
    "ai_fence_score",
    "ai_slope_score",
    "ai_noise_clue_score",
    "ai_photo_url_count",
    "open_house_status",
    "match_score",
    "recommendation_bucket",
    "final_verdict",
    "verification_checklist",
    "assessment_price_delta",
    "assessment_price_ratio",
    "land_value_share",
    "building_value_share",
    "assessment_interpretation",
    "buyer_fit_flags",
    "location_component",
    "location_flags",
    "condition_component",
    "condition_flags",
    "explanation",
    "bc_assessment_total_value",
    "bc_assessment_land_value",
    "bc_assessment_building_value",
    "bc_assessment_year",
    "bc_assessment_status",
    "Listing URL",
    "Google Maps Link",
    "BC Assessment Search Link",
]


def text_signal_score(text: pd.Series, keywords: list[str], strong_keywords: list[str] | None = None) -> pd.Series:
    lowered = text.fillna("").astype(str).str.lower()
    score = pd.Series(0, index=text.index, dtype=float)
    for keyword in keywords:
        score = score.mask(lowered.str.contains(keyword, regex=False), 65)
    for keyword in strong_keywords or []:
        score = score.mask(lowered.str.contains(keyword, regex=False), 100)
    return score


def location_score(text: pd.Series) -> pd.Series:
    lowered = text.fillna("").astype(str).str.lower()
    score = pd.Series(35, index=text.index, dtype=float)
    good_location_terms = [
        "edgemont village",
        "lynn valley",
        "dundarave",
        "ambleside",
        "caulfeild village",
        "horseshoe bay",
        "deep cove",
        "central lonsdale",
        "lower lonsdale",
        "walkable",
        "walking distance",
        "steps to",
        "near parks",
        "close to parks",
        "family-friendly neighborhood",
        "family friendly neighborhood",
        "quiet street",
        "cul-de-sac",
        "no-through",
    ]
    premium_terms = ["edgemont village", "dundarave", "ambleside", "caulfeild village", "lynn valley"]
    for term in good_location_terms:
        score = score.mask(lowered.str.contains(term, regex=False), 70)
    for term in premium_terms:
        score = score.mask(lowered.str.contains(term, regex=False), 95)
    return score


def location_flags(text: str) -> str:
    lowered = str(text).lower()
    flags = []
    for term in ["edgemont village", "lynn valley", "dundarave", "ambleside", "caulfeild village", "quiet street", "cul-de-sac", "walking distance"]:
        if term in lowered:
            flags.append(term)
    return "; ".join(flags) if flags else "location quality not highlighted"


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(50, index=series.index, dtype=float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        score = pd.Series(75, index=series.index, dtype=float)
    else:
        score = 100 * (values - lo) / (hi - lo)
    if not higher_is_better:
        score = 100 - score
    return score.clip(0, 100).fillna(50)


def add_lifestyle_components(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    text_cols = [
        column
        for column in ["Description", "House Category", "Ownership Category", "Nearby Ammenities", "Address"]
        if column in out.columns
    ]
    combined = out[text_cols].fillna("").astype(str).agg(" ".join, axis=1) if text_cols else pd.Series("", index=out.index)

    out["rancher_component"] = text_signal_score(
        combined,
        ["bungalow", "one level", "single level", "main floor living", "no stairs"],
        ["rancher", "ranch-style", "ranch style"],
    )
    out["backyard_component"] = text_signal_score(
        combined,
        ["garden", "outdoor space", "landscaped grounds", "level lot", "flat lot"],
        [
            "backyard",
            "private yard",
            "fully fenced backyard",
            "fenced backyard",
            "flat backyard",
            "level backyard",
            "sunny backyard",
            "south facing backyard",
            "usable backyard",
        ],
    )
    out["mortgage_helper_component"] = text_signal_score(
        combined,
        ["suite potential", "rental income", "separate entrance", "in-law", "nanny suite"],
        ["mortgage helper", "legal suite", "basement suite", "secondary suite"],
    )
    out["layout_component"] = text_signal_score(
        combined,
        ["functional layout", "family room", "main floor", "renovated", "updated", "rec room", "playroom", "bedrooms upstairs"],
        ["open concept", "great layout", "ideal family layout"],
    )
    out["location_component"] = location_score(combined)
    out["location_flags"] = combined.map(location_flags)
    out["condition_component"] = condition_score(combined)
    out["lifestyle_component"] = (
        out["rancher_component"] * 0.20
        + out["backyard_component"] * 0.35
        + out["mortgage_helper_component"] * 0.15
        + out["layout_component"] * 0.10
        + out["location_component"] * 0.10
        + out["condition_component"] * 0.10
    ).round(1)
    out["buyer_fit_flags"] = out.apply(build_fit_flags, axis=1)
    out["condition_flags"] = combined.map(condition_flags)
    out = apply_review_overrides(out)
    return out


def apply_review_overrides(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if "noise_verified" in out.columns:
        verified = out["noise_verified"].astype(str).str.title()
        valid_noise = verified.isin(["Low", "Medium", "High"])
        out.loc[valid_noise, "noise_model_risk"] = out.loc[valid_noise, "noise_risk"]
        out.loc[valid_noise, "noise_risk"] = verified[valid_noise]
        out.loc[valid_noise, "noise_override_note"] = "Human-reviewed noise override."
        out.loc[valid_noise, "noise_verification_needed"] = False

    if "ai_yard_score" in out.columns:
        ai_reviewed = out.get("photo_review_status", pd.Series("", index=out.index)).astype(str).str.contains("AI", case=False, na=False)
        yard_score = pd.to_numeric(out["ai_yard_score"], errors="coerce").clip(0, 100)
        slope_score = pd.to_numeric(out.get("ai_slope_score", pd.Series(50, index=out.index)), errors="coerce").clip(0, 100).fillna(50)
        fence_score = pd.to_numeric(out.get("ai_fence_score", pd.Series(50, index=out.index)), errors="coerce").clip(0, 100).fillna(50)
        privacy_score = pd.to_numeric(out.get("ai_privacy_score", pd.Series(50, index=out.index)), errors="coerce").clip(0, 100).fillna(50)
        photo_yard_component = (yard_score.fillna(50) * 0.55 + slope_score * 0.20 + fence_score * 0.15 + privacy_score * 0.10).round(1)
        out["ai_photo_yard_component"] = photo_yard_component
        out.loc[ai_reviewed, "backyard_component"] = out.loc[ai_reviewed, ["backyard_component", "ai_photo_yard_component"]].max(axis=1)

    if "ai_layout_score" in out.columns:
        ai_reviewed = out.get("photo_review_status", pd.Series("", index=out.index)).astype(str).str.contains("AI", case=False, na=False)
        layout_score = pd.to_numeric(out["ai_layout_score"], errors="coerce").clip(0, 100)
        layout_mask = ai_reviewed & layout_score.notna()
        current_layout = pd.to_numeric(out.loc[layout_mask, "layout_component"], errors="coerce").fillna(0)
        out.loc[layout_mask, "layout_component"] = current_layout.combine(layout_score.loc[layout_mask], max).astype(float)

    if "photo_yard_playability" in out.columns:
        photo_yard = out["photo_yard_playability"].astype(str).str.lower()
        out.loc[photo_yard.eq("great"), "backyard_component"] = 100
        photo_maybe = photo_yard.eq("maybe")
        out.loc[photo_maybe, "backyard_component"] = out.loc[photo_maybe, "backyard_component"].clip(lower=65)
        out.loc[photo_yard.eq("poor"), "backyard_component"] = 0

    if "yard_playability" in out.columns:
        yard = out["yard_playability"].astype(str).str.lower()
        out.loc[yard.eq("great"), "backyard_component"] = 100
        maybe_yard = yard.eq("maybe")
        out.loc[maybe_yard, "backyard_component"] = out.loc[maybe_yard, "backyard_component"].clip(lower=65)
        out.loc[yard.eq("poor"), "backyard_component"] = 0

    if "layout_fit" in out.columns:
        layout = out["layout_fit"].astype(str).str.lower()
        out.loc[layout.eq("great"), "layout_component"] = 100
        good_layout = layout.eq("good")
        out.loc[good_layout, "layout_component"] = out.loc[good_layout, "layout_component"].clip(lower=75)
        concern_layout = layout.eq("concern")
        out.loc[concern_layout, "layout_component"] = out.loc[concern_layout, "layout_component"].clip(upper=35)

    if "yard_noise" in out.columns:
        yard_noise = out["yard_noise"].astype(str).str.title()
        out.loc[yard_noise.eq("High"), "noise_verification_needed"] = True

    component_columns = [
        "rancher_component",
        "backyard_component",
        "mortgage_helper_component",
        "layout_component",
        "location_component",
        "condition_component",
    ]
    for column in component_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(float)

    out["lifestyle_component"] = (
        out["rancher_component"] * 0.15
        + out["backyard_component"] * 0.35
        + out["mortgage_helper_component"] * 0.10
        + out["layout_component"] * 0.15
        + out["location_component"] * 0.15
        + out["condition_component"] * 0.10
    ).astype(float).round(1)

    if "yard_playability" in out.columns:
        poor_yard = out["yard_playability"].astype(str).str.lower().eq("poor")
        out.loc[poor_yard, "backyard_component"] = 0
        out.loc[poor_yard, "lifestyle_component"] = out.loc[poor_yard, "lifestyle_component"].clip(upper=45)
    out["buyer_fit_flags"] = out.apply(build_fit_flags, axis=1)
    return out

def condition_score(text: pd.Series) -> pd.Series:
    lowered = text.fillna("").astype(str).str.lower()
    score = pd.Series(60, index=text.index, dtype=float)

    positive_terms = [
        "renovated", "extensively updated", "updated", "new roof", "new windows",
        "well-maintained", "beautifully maintained", "move-in ready", "turnkey",
    ]
    caution_terms = [
        "original condition", "older home", "mostly original", "needs work",
        "renovation project", "renovator", "as is", "estate sale", "lot value",
        "build your dream", "development opportunity", "tear down", "teardown",
        "land assembly", "holding property",
    ]
    for term in positive_terms:
        score = score.mask(lowered.str.contains(term, regex=False), 80)
    for term in caution_terms:
        score = score.mask(lowered.str.contains(term, regex=False), 25)
    return score


def condition_flags(text: str) -> str:
    lowered = str(text).lower()
    cautions = [
        term
        for term in ["original condition", "older home", "needs work", "renovation project", "estate sale", "lot value", "tear down", "teardown"]
        if term in lowered
    ]
    positives = [
        term
        for term in ["renovated", "updated", "new roof", "well-maintained", "move-in ready", "turnkey"]
        if term in lowered
    ]
    if cautions:
        return "condition caution: " + ", ".join(cautions[:2])
    if positives:
        return "condition positive: " + ", ".join(positives[:2])
    return "condition not clear"


def build_fit_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if row.get("rancher_component", 0) >= 65:
        flags.append("rancher/main-floor signal")
    if row.get("backyard_component", 0) >= 65:
        flags.append("backyard/yard signal")
    if row.get("mortgage_helper_component", 0) >= 65:
        flags.append("mortgage-helper signal")
    if row.get("layout_component", 0) >= 65:
        flags.append("layout signal")
    if row.get("location_component", 0) >= 70:
        flags.append("location signal")
    if row.get("condition_component", 60) <= 30:
        flags.append("condition/age caution")
    if row.get("client_status") in {"Liked", "Offered"}:
        flags.append(f"client {str(row.get('client_status')).lower()}")
    if row.get("yard_noise") == "High":
        flags.append("yard-noise concern")
    if row.get("photo_review_status") in {"AI reviewed", "Reviewed"}:
        flags.append("photo reviewed")
    return "; ".join(flags) if flags else "verify layout/backyard manually"


def score_listings(df: pd.DataFrame, prefs: dict[str, Any]) -> pd.DataFrame:
    data = add_lifestyle_components(df)
    max_price = float(prefs["max_price"])
    min_beds = float(prefs["min_bedrooms"])
    min_school = float(prefs["min_fraser_score"])

    school_weight = float(prefs["school_importance"])
    quiet_weight = float(prefs["quiet_importance"])
    price_weight = float(prefs["price_importance"])
    size_weight = float(prefs["size_importance"])
    lifestyle_weight = float(prefs.get("lifestyle_importance", 3))
    total_weight = max(school_weight + quiet_weight + price_weight + size_weight + lifestyle_weight, 1)

    school_score = (data["fraser_score_numeric"].fillna(0).clip(0, 10) / 10) * 100
    budget_price_score = ((max_price - data["price_numeric"]) / max(max_price, 1) * 100).clip(0, 100).fillna(0)
    sqft_value_score = minmax_score(data.get("price_per_sqft", pd.Series(index=data.index, dtype=float)), higher_is_better=False)
    assessment_total = pd.to_numeric(data.get("bc_assessment_total_value", pd.Series(index=data.index, dtype=float)), errors="coerce")
    assessment_discount = ((assessment_total - data["price_numeric"]) / assessment_total * 100).replace([float("inf"), -float("inf")], pd.NA)
    assessment_score = minmax_score(assessment_discount, higher_is_better=True)
    has_assessment = assessment_total.notna()
    price_score = (budget_price_score * 0.55 + sqft_value_score * 0.30 + assessment_score * 0.15).where(
        has_assessment,
        budget_price_score * 0.65 + sqft_value_score * 0.35,
    )
    bedroom_fit = (data["bedrooms_numeric"].fillna(0) / max(min_beds, 1)).clip(0, 1)
    size_fit = (data["size_sqft"].fillna(0) / 2500).clip(0, 1)
    size_score = ((bedroom_fit * 0.6) + (size_fit * 0.4)) * 100

    noise_penalty = data["noise_risk"].map(NOISE_PENALTY_MAP).fillna(NOISE_PENALTY_MAP["Unknown"])
    quiet_score = (100 - noise_penalty).clip(0, 100)
    if "ai_noise_clue_score" in data.columns:
        ai_reviewed = data.get("photo_review_status", pd.Series("", index=data.index)).astype(str).str.contains("AI", case=False, na=False)
        ai_quiet = pd.to_numeric(data["ai_noise_clue_score"], errors="coerce").clip(0, 100)
        quiet_score = quiet_score.where(~(ai_reviewed & ai_quiet.notna()), quiet_score * 0.70 + ai_quiet * 0.30)

    school_confidence_bonus = data["school_confidence"].astype(str).str.lower().map(
        lambda value: 4 if "high" in value or "official" in value else 2 if "medium" in value else 0
    )
    open_house_bonus = data["open_house_status"].eq("Upcoming").astype(int) * 3
    condition_penalty = ((60 - data["condition_component"]).clip(lower=0) * 0.20).fillna(0)

    weighted = (
        school_score * school_weight
        + quiet_score * quiet_weight
        + price_score * price_weight
        + size_score * size_weight
        + data["lifestyle_component"] * lifestyle_weight
    ) / total_weight

    data["school_component"] = school_score.round(1)
    data["price_component"] = price_score.round(1)
    data["budget_price_component"] = budget_price_score.round(1)
    data["sqft_value_component"] = sqft_value_score.round(1)
    data["assessment_value_component"] = assessment_score.round(1)
    data["size_component"] = size_score.round(1)
    data["quiet_component"] = quiet_score.round(1)
    data["noise_penalty"] = noise_penalty
    data["match_score"] = (weighted + school_confidence_bonus + open_house_bonus - condition_penalty).clip(0, 100).round(1)
    if "yard_playability" in data.columns:
        poor_yard = data["yard_playability"].astype(str).str.lower().eq("poor")
        data.loc[poor_yard, "match_score"] = data.loc[poor_yard, "match_score"].clip(upper=54)

    data["excluded_reason"] = ""
    data.loc[data["price_numeric"] > max_price, "excluded_reason"] += "Over max price. "
    data.loc[data["bedrooms_numeric"] < min_beds, "excluded_reason"] += "Below bedroom target. "
    data.loc[data["fraser_score_numeric"].fillna(-1) < min_school, "excluded_reason"] += "Below school score target. "
    if prefs.get("exclude_high_noise"):
        data.loc[data["noise_risk"].eq("High"), "excluded_reason"] += "High noise risk. "

    data["included"] = data["excluded_reason"].str.len().eq(0)
    data = add_assessment_fields(data)
    data["explanation"] = data.apply(build_explanation, axis=1)
    data = add_decision_fields(data)
    return data.sort_values(["included", "match_score", "price_numeric"], ascending=[False, False, True])


def build_explanation(row: pd.Series) -> str:
    positives: list[str] = []
    cautions: list[str] = []

    if pd.notna(row.get("fraser_score_numeric")):
        positives.append(f"school score {row['fraser_score_numeric']:.1f}")
    else:
        cautions.append("school score missing")

    if row.get("noise_risk") == "Low":
        positives.append("low estimated noise")
    elif row.get("noise_risk") in {"Medium", "High"}:
        cautions.append(f"{row.get('noise_risk')} noise risk")
    else:
        cautions.append("noise needs verification")
    if str(row.get("noise_override_note", "")).strip():
        cautions.append("noise manually adjusted; verify at showing")
    elif str(row.get("noise_context_note", "")).strip():
        cautions.append("noise adjusted by context; verify at showing")

    if row.get("open_house_status") == "Upcoming":
        positives.append("upcoming open house")
    elif row.get("open_house_status") == "Past":
        cautions.append("open house is past")

    if row.get("price_component", 0) >= 50:
        positives.append("good price fit")
    if row.get("size_component", 0) >= 70:
        positives.append("strong size/bedroom fit")
    if row.get("lifestyle_component", 0) >= 65:
        positives.append(row.get("buyer_fit_flags", "strong lifestyle fit"))
    elif row.get("location_component", 0) >= 90:
        positives.append(row.get("location_flags", "strong location"))
    if row.get("condition_component", 60) <= 30:
        cautions.append(row.get("condition_flags", "condition needs review"))

    text = "Recommended for " + ", ".join(positives[:3]) if positives else "Worth reviewing"
    if cautions:
        text += "; verify " + ", ".join(cautions[:3])
    return text + "."



def add_assessment_fields(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    total = pd.to_numeric(out.get("bc_assessment_total_value", pd.Series(index=out.index, dtype=float)), errors="coerce")
    land = pd.to_numeric(out.get("bc_assessment_land_value", pd.Series(index=out.index, dtype=float)), errors="coerce")
    building = pd.to_numeric(out.get("bc_assessment_building_value", pd.Series(index=out.index, dtype=float)), errors="coerce")
    price = pd.to_numeric(out.get("price_numeric", pd.Series(index=out.index, dtype=float)), errors="coerce")

    out["assessment_price_delta"] = (price - total).where(total.notna())
    out["assessment_price_ratio"] = (price / total).where(total.gt(0)).round(2)
    out["land_value_share"] = (land / total * 100).where(total.gt(0)).round(1)
    out["building_value_share"] = (building / total * 100).where(total.gt(0)).round(1)
    out["assessment_interpretation"] = out.apply(assessment_interpretation, axis=1)
    return out


def assessment_interpretation(row: pd.Series) -> str:
    ratio = row.get("assessment_price_ratio")
    land_share = row.get("land_value_share")
    building_share = row.get("building_value_share")
    if pd.isna(ratio):
        return "Assessment not entered; use BC Assessment link or manual value entry."

    parts: list[str] = []
    if ratio >= 1.25:
        parts.append("Price is materially above assessment")
    elif ratio <= 0.95:
        parts.append("Price is at or below assessment")
    else:
        parts.append("Price is close to assessment")

    if pd.notna(land_share) and land_share >= 75:
        parts.append("value is mostly land")
    elif pd.notna(building_share) and building_share >= 35:
        parts.append("building value is meaningful")
    return "; ".join(parts) + "."


def add_decision_fields(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["recommendation_bucket"] = out.apply(recommendation_bucket, axis=1)
    out["verification_checklist"] = out.apply(build_verification_checklist, axis=1)
    out["final_verdict"] = out.apply(final_verdict, axis=1)
    return out


def recommendation_bucket(row: pd.Series) -> str:
    status = str(row.get("client_status", "New"))
    if status in {"Offered", "Liked"}:
        return "Client Liked / Offered"
    if status in {"Rejected", "Disliked"}:
        return "Rejected / Not A Fit"

    if row.get("yard_playability") == "Poor":
        return "Poor Yard / Not A Fit"

    has_hard_noise_gap = row.get("noise_verification_needed") or row.get("yard_noise") == "High"
    has_yard_signal = row.get("backyard_component", 0) >= 65 or row.get("yard_playability") == "Great"
    if row.get("match_score", 0) >= 75 and row.get("noise_risk") == "Low" and not has_hard_noise_gap and has_yard_signal:
        return "Top Shortlist"
    if has_hard_noise_gap or row.get("yard_playability") == "Maybe":
        return "Needs Verification"
    if row.get("match_score", 0) >= 65:
        return "Good Candidate"
    return "Watch / Lower Fit"


def build_verification_checklist(row: pd.Series) -> str:
    checks: list[str] = []
    if row.get("yard_playability") == "Poor":
        checks.append("Manual review says no/poor playable yard")
    elif row.get("yard_playability") in {"Unknown", "Maybe"} and row.get("photo_yard_playability") in {"Unknown", "Maybe", ""}:
        checks.append("Review photos/showing for flat fenced playable yard")
    if row.get("yard_noise") in {"Unknown", "High"} and row.get("noise_risk") in {"Medium", "High"}:
        checks.append("Stand in backyard and verify road/highway noise")
    elif row.get("noise_verification_needed"):
        checks.append("Verify real-world noise at showing")
    if row.get("layout_fit") == "Unknown":
        checks.append("Confirm bedroom layout, family room flow, and stairs")
    if pd.isna(row.get("bc_assessment_total_value")):
        checks.append("Enter BC Assessment land/building values")
    if row.get("final_school") in {"Unknown", ""} or pd.isna(row.get("final_fraser_score")):
        checks.append("Verify school catchment")
    if not checks:
        checks.append("No major verification gaps recorded")
    return " | ".join(checks)


def final_verdict(row: pd.Series) -> str:
    bucket = row.get("recommendation_bucket", "Review")
    positives: list[str] = []
    concerns: list[str] = []

    if row.get("location_component", 0) >= 90:
        positives.append("excellent location")
    if row.get("backyard_component", 0) >= 65:
        positives.append("yard/playability signal")
    if row.get("layout_component", 0) >= 65:
        positives.append("family layout signal")
    if row.get("size_component", 0) >= 80:
        positives.append("strong interior size")
    if row.get("noise_risk") == "High" or row.get("yard_noise") == "High":
        concerns.append("noise risk")
    if row.get("yard_playability") == "Poor":
        concerns.append("manual review says no/poor playable yard")
    elif row.get("yard_playability") == "Maybe" or (row.get("yard_playability") == "Unknown" and row.get("backyard_component", 0) < 65):
        concerns.append("yard needs verification")
    if pd.isna(row.get("bc_assessment_total_value")):
        concerns.append("assessment missing")

    text = f"{bucket}: "
    text += ", ".join(positives[:3]) if positives else "worth reviewing"
    if concerns:
        text += "; verify " + ", ".join(concerns[:3])
    return text + "."


def marker_color(row: pd.Series) -> str:
    if row.get("noise_risk") == "Unknown" or pd.isna(row.get("match_score")):
        return "gray"
    if row.get("noise_risk") == "High" or row.get("match_score", 0) < 55:
        return "red"
    if row.get("match_score", 0) >= 75 and row.get("noise_risk") == "Low":
        return "green"
    return "orange"


def filter_by_preferences(scored: pd.DataFrame, prefs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = scored[scored["included"]].copy()
    excluded = scored[~scored["included"]].copy()

    city = prefs["preferred_city"]
    if city != "Both":
        city_mask = filtered["City"].astype(str).str.contains(city, case=False, na=False)
        excluded = pd.concat([excluded, filtered[~city_mask].assign(excluded_reason="Outside preferred city.")])
        filtered = filtered[city_mask]

    return filtered.sort_values("match_score", ascending=False), excluded.sort_values("match_score", ascending=False)


def display_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [column for column in DISPLAY_COLUMNS if column in df.columns]
    return df[existing].copy()


def scoring_method_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item": [
                "Formula",
                "School score",
                "Price fit",
                "Size/bedroom fit",
                "Rancher/backyard/helper/layout fit",
                "Noise penalty",
                "Open house bonus",
                "School confidence bonus",
            ],
            "Description": [
                "Weighted average of school, price, size, quiet, and yard/layout/location components, plus transparent bonuses and manual review overrides.",
                "final_fraser_score normalized from 0 to 100.",
                "Combines budget fit, price per sqft, and manually entered BC Assessment comparison when available.",
                "Blend of bedroom target fit and size fit, capped at 100.",
                "Rule-based listing-text signals plus AI photo scores for yard, layout, privacy, fencing, slope, and visual noise clues when reviewed.",
                "Low = 0, Medium = 18, High = 40, Unknown = 12 before quiet weighting.",
                "+3 points only when open_house_status is Upcoming.",
                "+4 for high/official confidence, +2 for medium confidence.",
            ],
        }
    )


def export_excel(prefs: dict[str, Any], top: pd.DataFrame, all_filtered: pd.DataFrame, excluded: pd.DataFrame) -> bytes:
    output = BytesIO()
    prefs_df = pd.DataFrame({"Preference": list(prefs.keys()), "Value": [str(value) for value in prefs.values()]})

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        prefs_df.to_excel(writer, sheet_name="Client Preferences", index=False)
        top.to_excel(writer, sheet_name="Top Recommendations", index=False)
        all_filtered.to_excel(writer, sheet_name="All Filtered Listings", index=False)
        excluded.to_excel(writer, sheet_name="Excluded Homes", index=False)
        scoring_method_df().to_excel(writer, sheet_name="Scoring Method", index=False)

    return output.getvalue()


