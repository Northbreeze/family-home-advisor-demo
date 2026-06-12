from __future__ import annotations

import re
from typing import Any

import pandas as pd

from area_filters import add_area_columns
from scoring import score_listings
from v2_product import concerns


def num(value: object, default: float = 0.0) -> float:
    value = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(value) else float(value)


def score(row: pd.Series) -> int:
    return int(round(num(row.get("match_score"), 0)))


def size_sqft_value(row: pd.Series) -> float:
    for name in ["size_sqft", "sqft_numeric"]:
        value = pd.to_numeric(row.get(name), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value)
    text = str(row.get("Size", ""))
    match = re.search(r"([0-9][0-9,]*(?:\.\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    return float("nan")


def interior_size_label(size: float) -> tuple[str, str, float | None]:
    if pd.isna(size) or size <= 0:
        return "Unknown", "Interior size is missing; verify listing sqft.", None
    if size < 1600:
        return "Major concern", f"Interior size is only {size:,.0f} sqft, which is a major family-fit concern.", 58
    if size < 1900:
        return "Modest", f"Interior size is {size:,.0f} sqft, so the layout needs careful verification.", 70
    if size < 2400:
        return "Acceptable", f"Interior size is {size:,.0f} sqft; likely workable if the layout is efficient.", None
    if size < 3200:
        return "Family-friendly", f"Interior size is {size:,.0f} sqft, which looks family-friendly.", None
    return "Spacious", f"Interior size is {size:,.0f} sqft, which is spacious for family use.", None


def interior_size_summary(row: pd.Series) -> str:
    rating = str(row.get("interior_size_rating", "")).strip()
    note = str(row.get("interior_size_note", "")).strip()
    if rating and note and rating.lower() not in {"nan", "none"}:
        return f"Interior: {rating}. {note}"
    size = size_sqft_value(row)
    rating, note, _ = interior_size_label(size)
    return f"Interior: {rating}. {note}"


def family_concern_items(row: pd.Series, profile: dict[str, Any] | None = None, limit: int = 4) -> list[str]:
    profile = profile or {}
    items: list[str] = []
    rating = str(row.get("interior_size_rating", "")).strip()
    note = str(row.get("interior_size_note", "")).strip()
    if not rating or rating.lower() in {"nan", "none"}:
        size = size_sqft_value(row)
        rating, note, _ = interior_size_label(size)
    if rating in {"Major concern", "Modest", "Unknown"} and note:
        items.append(note)
    family_note = str(row.get("family_evaluation_note", "")).strip()
    if family_note:
        for part in [piece.strip() for piece in family_note.split(". ") if piece.strip()]:
            text = part if part.endswith(".") else part + "."
            if "Interior under" not in text and "Interior size is modest" not in text and text not in items:
                items.append(text)
    for item in concerns(row, profile, limit=limit + 4):
        if item.startswith("Interior size") and any("Interior size" in existing for existing in items):
            continue
        if item.startswith("Yard usability is unknown") and any("Yard is unknown" in existing for existing in items):
            continue
        if item not in items:
            items.append(item)
    return items[:limit] if items else ["No major concern is obvious from the current data, but photos and showing still matter."]


def category(row: pd.Series) -> str:
    s = score(row)
    noise = str(row.get("noise_risk", "Unknown"))
    yard = str(row.get("yard_playability", "Unknown"))
    if noise == "High" or yard == "Poor" or s < 60:
        return "Needs Verification"
    if s >= 82 and noise in {"Low", "Unknown"}:
        return "Strong Match"
    return "Worth Visiting"


def tone(row: pd.Series) -> str:
    return {"Strong Match": "green", "Worth Visiting": "yellow", "Needs Verification": "red"}[category(row)]


def base_prefs(max_price: int, min_beds: int, min_school: float, chips: list[str]) -> dict[str, Any]:
    quiet = 5 if any(x in chips for x in ["Quiet street", "Avoid busy roads", "Good for toddlers"]) else 4
    lifestyle = 5 if any(x in chips for x in ["Good for toddlers", "Large backyard", "Move-in ready"]) else 4
    size = 5 if any(x in chips for x in ["Good for toddlers", "Large backyard"]) else 4
    school = 4 if min_school >= 8 or "Strong resale" in chips else 3
    return {
        "profile_name": "Family Fit",
        "deal_breakers": ["Avoid busy roads", "No usable yard", "Too small interior"],
        "max_price": max_price,
        "min_bedrooms": min_beds,
        "min_fraser_score": min_school,
        "quiet_importance": quiet,
        "school_importance": school,
        "price_importance": 3,
        "size_importance": size,
        "lifestyle_importance": lifestyle,
        "preferred_city": "Both",
        "exclude_high_noise": "Avoid busy roads" in chips or "Quiet street" in chips,
    }


def col(df: pd.DataFrame, name: str, default: object = "") -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)

def filter_ranked_data(scored: pd.DataFrame, search: str, price_range: tuple[int, int], locations: list[str], min_beds: int, min_school: float, yard: str, more: dict[str, bool]) -> pd.DataFrame:
    data = scored.copy()
    data = data[pd.to_numeric(data["price_numeric"], errors="coerce").between(price_range[0], price_range[1], inclusive="both")]
    data = data[pd.to_numeric(data["bedrooms_numeric"], errors="coerce").fillna(0) >= min_beds]
    data = data[pd.to_numeric(data["fraser_score_numeric"], errors="coerce").fillna(0) >= min_school]
    if search.strip():
        q = search.strip().lower()
        text = (col(data, "Address", "").astype(str) + " " + col(data, "City", "").astype(str) + " " + col(data, "detected_area", "").astype(str) + " " + col(data, "final_school", "").astype(str)).str.lower()
        data = data[text.str.contains(re.escape(q), na=False)]
    if locations:
        masks = []
        for loc in locations:
            if loc in {"North Vancouver", "West Vancouver"}:
                masks.append(col(data, "City", "").astype(str).str.contains(loc, case=False, na=False))
            else:
                masks.append(col(data, "detected_area", "").astype(str).str.contains(loc, case=False, na=False) | col(data, "Address", "").astype(str).str.contains(loc, case=False, na=False))
        mask = masks[0]
        for item in masks[1:]:
            mask = mask | item
        data = data[mask]

    yard_status = col(data, "yard_playability", "Unknown").astype(str)
    yard_score = pd.to_numeric(col(data, "backyard_component", 0), errors="coerce").fillna(0)
    photo_yard = pd.to_numeric(col(data, "ai_yard_score", pd.NA), errors="coerce")
    yard_text = (
        col(data, "Description", "").astype(str) + " "
        + col(data, "Public Remarks", "").astype(str) + " "
        + col(data, "Address", "").astype(str)
    ).str.lower()
    usable_terms = ["yard", "backyard", "garden", "patio", "fenced", "level lot", "flat lot", "private lot", "outdoor"]
    large_terms = ["large yard", "large backyard", "large lot", "private backyard", "sunny backyard", "oversized lot", "level backyard"]
    text_usable = yard_text.apply(lambda value: any(term in value for term in usable_terms))
    text_large = yard_text.apply(lambda value: any(term in value for term in large_terms))
    positive_yard = yard_status.isin(["Great", "Maybe"]) | yard_score.ge(45) | photo_yard.ge(55).fillna(False) | text_usable
    strong_yard = yard_status.eq("Great") | yard_score.ge(65) | photo_yard.ge(70).fillna(False) | text_large
    known_poor = yard_status.eq("Poor") | yard_score.le(20)
    if yard == "Usable yard":
        data = data[positive_yard]
    elif yard == "Large yard signal":
        data = data[strong_yard]
    elif yard == "Exclude known poor yard":
        data = data[~known_poor]

    if more.get("open_house_only"):
        data = data[data["open_house_status"].eq("Upcoming")]
    if more.get("new_only"):
        data = data[col(data, "is_new_since_last_refresh", False).fillna(False)]
    if more.get("avoid_high_noise"):
        data = data[~data["noise_risk"].eq("High")]
    if more.get("assessment_only"):
        data = data[pd.to_numeric(col(data, "bc_assessment_total_value", pd.NA), errors="coerce").notna()]
    return data.copy()


def sort_visible(df: pd.DataFrame, recommended_first: bool) -> pd.DataFrame:
    if df.empty:
        return df
    if recommended_first:
        return df.sort_values(["match_score", "price_numeric"], ascending=[False, True])
    return df.sort_values(["price_numeric", "match_score"], ascending=[True, False])


def apply_family_evaluation(scored: pd.DataFrame, chips: list[str]) -> pd.DataFrame:
    out = scored.copy()
    size = pd.to_numeric(col(out, "size_sqft", pd.NA), errors="coerce")
    if size.isna().all() and "Size" in out.columns:
        size = out["Size"].astype(str).str.extract(r"([0-9][0-9,]*(?:\.\d+)?)", expand=False).str.replace(",", "", regex=False).pipe(pd.to_numeric, errors="coerce")
    score_values = pd.to_numeric(out["match_score"], errors="coerce").fillna(0)
    out["family_evaluation_note"] = ""
    out["interior_size_sqft"] = size
    labels = size.apply(interior_size_label)
    out["interior_size_rating"] = labels.apply(lambda value: value[0])
    out["interior_size_note"] = labels.apply(lambda value: value[1])

    very_small = size.gt(0) & size.lt(1600)
    small = size.ge(1600) & size.lt(1900)
    high_noise = out["noise_risk"].eq("High")
    poor_yard = col(out, "yard_playability", "Unknown").astype(str).eq("Poor")
    yard_unknown = col(out, "yard_playability", "Unknown").astype(str).eq("Unknown")

    out.loc[very_small, "match_score"] = score_values[very_small].clip(upper=58)
    out.loc[very_small, "family_evaluation_note"] += "Interior under 1,600 sqft is a major family-fit concern. "
    out.loc[small, "match_score"] = pd.to_numeric(out.loc[small, "match_score"], errors="coerce").clip(upper=70)
    out.loc[small, "family_evaluation_note"] += "Interior size is modest; verify layout carefully. "
    out.loc[high_noise, "match_score"] = pd.to_numeric(out.loc[high_noise, "match_score"], errors="coerce").clip(upper=56)
    out.loc[high_noise, "family_evaluation_note"] += "High noise risk is treated as a deal breaker. "
    out.loc[poor_yard, "match_score"] = pd.to_numeric(out.loc[poor_yard, "match_score"], errors="coerce").clip(upper=54)
    out.loc[poor_yard, "family_evaluation_note"] += "No usable yard is treated as a deal breaker. "
    if "Good for toddlers" in chips or "Large backyard" in chips:
        out.loc[yard_unknown, "match_score"] = pd.to_numeric(out.loc[yard_unknown, "match_score"], errors="coerce").clip(upper=76)
        out.loc[yard_unknown, "family_evaluation_note"] += "Yard is unknown and needs photo/showing verification. "
    out["match_score"] = pd.to_numeric(out["match_score"], errors="coerce").fillna(0).round(1)
    return out.sort_values(["match_score", "price_numeric"], ascending=[False, True])


def score_family_fit(df: pd.DataFrame, prefs: dict[str, Any], chips: list[str]) -> pd.DataFrame:
    return apply_family_evaluation(add_area_columns(score_listings(df, prefs)), chips)


def metrics(scored: pd.DataFrame, visible: pd.DataFrame) -> dict[str, int]:
    strong = int(visible.apply(lambda r: category(r) == "Strong Match", axis=1).sum()) if not visible.empty else 0
    open_houses = int(visible.get("open_house_status", pd.Series(index=visible.index)).eq("Upcoming").sum()) if not visible.empty else 0
    changed = int(visible.get("is_new_since_last_refresh", pd.Series(False, index=visible.index)).fillna(False).sum()) if not visible.empty else 0
    return {"all": len(scored), "visible": len(visible), "strong": strong, "open_houses": open_houses, "changed": changed}
