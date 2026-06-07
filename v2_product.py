from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_cleaning import google_maps_link, money

PROFILE_DEFAULTS: dict[str, Any] = {
    "budget_logic": {
        "helper_optional_max": 2100000,
        "helper_helpful_max": 2500000,
        "under_optional_label": "Mortgage helper optional under $2.1M",
        "middle_helpful_label": "Mortgage helper helpful from $2.1M to $2.5M",
        "over_required_label": "Mortgage helper important above $2.5M",
    },
    "preferred_cities": ["West Vancouver", "North Vancouver"],
    "preferred_neighbourhoods": ["Caulfeild", "Ambleside", "Dundarave", "Edgemont", "Canyon Heights", "Lynn Valley"],
    "avoided_areas": [],
    "deal_breakers": ["Highway noise", "Busy arterial road", "Steep or unusable yard", "Too small interior"],
    "important_preferences": [
        "Usable toddler yard",
        "Quiet street",
        "Family-sized layout",
        "Work-from-home space",
        "Good school",
        "Mortgage helper depending on price",
        "Parks / nature nearby",
    ],
    "flexible_preferences": ["Cosmetic renovation okay", "Older house okay if location is good", "Kitchen can be renovated"],
    "learned_rules": [
        "Location matters more than finishes.",
        "Highway noise is a deal breaker.",
        "Mortgage helper only matters strongly above $2.1M.",
        "Small usable yard is better than large unusable or steep lot.",
    ],
}

EVENT_COLUMNS = ["timestamp", "event_type", "address", "reason", "details"]


def clean_text(value: object, default: str = "Unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).replace("|", ", ").strip()
    return text if text and text.lower() != "nan" else default


def parse_num(value: object) -> float:
    return pd.to_numeric(value, errors="coerce")


def load_family_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        save_family_profile(path, PROFILE_DEFAULTS)
        return dict(PROFILE_DEFAULTS)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception:
        return dict(PROFILE_DEFAULTS)
    profile = dict(PROFILE_DEFAULTS)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(profile.get(key), dict):
            merged = dict(profile[key])
            merged.update(value)
            profile[key] = merged
        else:
            profile[key] = value
    return profile


def save_family_profile(path: Path, profile: dict[str, Any]) -> None:
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_listing_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = pd.read_csv(path)
    for column in EVENT_COLUMNS:
        if column not in events.columns:
            events[column] = ""
    return events[EVENT_COLUMNS]


def append_listing_event(path: Path, event_type: str, address: str, reason: str = "", details: str = "") -> None:
    events = load_listing_events(path)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event_type": event_type,
        "address": clean_text(address, ""),
        "reason": reason,
        "details": details,
    }
    pd.concat([events, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def recommendation_category(row: pd.Series) -> str:
    bucket = clean_text(row.get("recommendation_bucket", ""), "").lower()
    score = parse_num(row.get("match_score"))
    noise = clean_text(row.get("noise_risk", "Unknown"))
    yard = clean_text(row.get("yard_playability", "Unknown"))
    if "poor yard" in bucket or "not a fit" in bucket or yard == "Poor":
        return "Excluded by Deal Breaker"
    if noise == "High":
        return "Excluded by Deal Breaker"
    if pd.notna(score) and score >= 82:
        return "Excellent Match"
    if pd.notna(score) and score >= 68:
        return "Worth Touring"
    if "needs verification" in bucket or noise in {"Medium", "Unknown"} or yard == "Unknown":
        return "Needs More Information"
    if pd.notna(score) and score < 58:
        return "Significant Trade-offs"
    return "Worth Touring"


def confidence_level(row: pd.Series) -> str:
    unknowns = 0
    for column in ["yard_playability", "layout_fit", "noise_risk", "photo_review_status"]:
        value = clean_text(row.get(column, "Unknown"))
        if value in {"Unknown", "Not reviewed", ""}:
            unknowns += 1
    if unknowns >= 3:
        return "Low"
    if unknowns >= 1:
        return "Medium"
    return "High"


def mortgage_helper_need(row: pd.Series, profile: dict[str, Any]) -> str:
    price = parse_num(row.get("price_numeric", row.get("Price")))
    logic = profile.get("budget_logic", PROFILE_DEFAULTS["budget_logic"])
    optional_max = float(logic.get("helper_optional_max", 2100000))
    helpful_max = float(logic.get("helper_helpful_max", 2500000))
    if pd.isna(price):
        return "Unknown"
    if price <= optional_max:
        return "Not needed"
    if price <= helpful_max:
        return "Helpful"
    return "Important"


def why_it_may_work(row: pd.Series, profile: dict[str, Any], limit: int = 4) -> list[str]:
    reasons: list[str] = []
    price = parse_num(row.get("price_numeric", row.get("Price")))
    helper_need = mortgage_helper_need(row, profile)
    school_score = parse_num(row.get("final_fraser_score"))
    noise = clean_text(row.get("noise_risk", "Unknown"))
    size = parse_num(row.get("sqft_numeric", row.get("Size")))
    city = clean_text(row.get("City", ""), "")
    area = clean_text(row.get("detected_area", ""), "")
    if pd.notna(price) and price <= float(profile.get("budget_logic", {}).get("helper_optional_max", 2100000)):
        reasons.append("Price is within the no-helper comfort zone.")
    elif helper_need in {"Helpful", "Important"} and row.get("mortgage_helper_component", 0) and parse_num(row.get("mortgage_helper_component")) >= 65:
        reasons.append("Listing language suggests mortgage-helper potential.")
    if city in profile.get("preferred_cities", []):
        reasons.append(f"{city} matches the preferred city list.")
    if area in profile.get("preferred_neighbourhoods", []):
        reasons.append(f"{area} is one of the preferred neighbourhoods.")
    if pd.notna(school_score) and school_score >= 8:
        reasons.append("Strong elementary school catchment.")
    elif pd.notna(school_score) and school_score >= 7:
        reasons.append("School score looks acceptable.")
    if noise == "Low":
        reasons.append("Noise risk is currently low.")
    if pd.notna(size) and size >= 2400:
        reasons.append("Interior size looks family-friendly.")
    if clean_text(row.get("yard_playability", "Unknown")) in {"Great", "Maybe"}:
        reasons.append("Yard may be usable for a child, pending verification.")
    if not reasons:
        reasons.append("It remains in the inventory and may be worth checking if the location works.")
    return reasons[:limit]


def concerns(row: pd.Series, profile: dict[str, Any], limit: int = 4) -> list[str]:
    items: list[str] = []
    noise = clean_text(row.get("noise_risk", "Unknown"))
    yard = clean_text(row.get("yard_playability", "Unknown"))
    layout = clean_text(row.get("layout_fit", "Unknown"))
    assessment = clean_text(row.get("bc_assessment_status", ""), "")
    size = parse_num(row.get("sqft_numeric", row.get("Size")))
    helper_need = mortgage_helper_need(row, profile)
    helper_component = parse_num(row.get("mortgage_helper_component"))
    if noise == "High":
        items.append("High noise risk may be a deal breaker.")
    elif noise in {"Medium", "Unknown"}:
        items.append(f"Noise risk is {noise.lower()} and should be verified in person.")
    if yard == "Poor":
        items.append("Yard appears unsuitable for the family requirement.")
    elif yard == "Unknown":
        items.append("Yard usability is unknown from current data.")
    if layout in {"Concern", "Unknown"}:
        items.append(f"Layout is {layout.lower()} and needs photo/showing review.")
    if helper_need in {"Helpful", "Important"} and (pd.isna(helper_component) or helper_component < 65):
        items.append(f"Mortgage helper is {helper_need.lower()} at this price, but not confirmed.")
    if pd.notna(size) and size < 1800:
        items.append("Interior size may feel small for a family.")
    if not assessment or "not" in assessment.lower() or "missing" in assessment.lower():
        items.append("BC Assessment is not verified yet.")
    if not items:
        items.append("No major concern is obvious from the current data, but photos and showing still matter.")
    return items[:limit]


def verification_steps(row: pd.Series, profile: dict[str, Any], limit: int = 5) -> list[str]:
    steps = [
        "Stand outside for 3-5 minutes and listen for highway or arterial-road noise.",
        "Check whether the yard has a flat, usable toddler play area.",
        "Confirm whether the layout feels family-sized in person.",
    ]
    if mortgage_helper_need(row, profile) in {"Helpful", "Important"}:
        steps.append("Verify whether a lower-level suite or separate entrance is realistic.")
    steps.append("Open photos, Street View, and BC Assessment before prioritizing the tour.")
    return steps[:limit]


def recommendation_sentence(row: pd.Series, profile: dict[str, Any]) -> str:
    category = recommendation_category(row)
    reasons = why_it_may_work(row, profile, limit=2)
    risks = concerns(row, profile, limit=1)
    return f"{category}. {reasons[0]} Main thing to verify: {risks[0]}"


def evidence_table(row: pd.Series, profile: dict[str, Any]) -> pd.DataFrame:
    school_score = parse_num(row.get("final_fraser_score"))
    size = parse_num(row.get("sqft_numeric", row.get("Size")))
    noise = clean_text(row.get("noise_risk", "Unknown"))
    yard = clean_text(row.get("yard_playability", "Unknown"))
    layout = clean_text(row.get("layout_fit", "Unknown"))
    condition_component = parse_num(row.get("condition_component"))
    helper_need = mortgage_helper_need(row, profile)
    price = parse_num(row.get("price_numeric", row.get("Price")))
    location_component = parse_num(row.get("location_component"))
    rows = [
        ("Location", "Excellent" if pd.notna(location_component) and location_component >= 80 else "Good" if pd.notna(location_component) and location_component >= 60 else "Concern"),
        ("Noise", noise),
        ("Yard", "Usable" if yard in {"Great", "Maybe"} else "Concern" if yard == "Poor" else "Unknown"),
        ("Layout", "Good" if layout in {"Great", "Good"} else "Concern" if layout == "Concern" else "Unknown"),
        ("Interior size", "Good" if pd.notna(size) and size >= 2200 else "Small" if pd.notna(size) and size < 1800 else "Unknown"),
        ("School", "Strong" if pd.notna(school_score) and school_score >= 8 else "Good" if pd.notna(school_score) and school_score >= 7 else "Weak/Unknown"),
        ("Renovation", "Updated" if pd.notna(condition_component) and condition_component >= 70 else "Unknown"),
        ("Mortgage helper", helper_need),
        ("Price fit", "Within comfort zone" if pd.notna(price) and price <= 2100000 else "Needs helper" if pd.notna(price) and price <= 2500000 else "Above target"),
    ]
    return pd.DataFrame(rows, columns=["Evidence", "Assessment"])


def listing_links(row: pd.Series) -> dict[str, str]:
    maps = row.get("Google Maps Link") or google_maps_link(row)
    return {
        "Realtor.ca": clean_text(row.get("Listing URL", ""), ""),
        "Google Maps": maps,
        "Street View": maps,
        "BC Assessment": clean_text(row.get("BC Assessment Search Link", ""), ""),
    }


def card_title(row: pd.Series) -> str:
    address = clean_text(row.get("Address", ""), "Unknown address")
    return address.split(",")[0]


def card_subtitle(row: pd.Series) -> str:
    pieces = [clean_text(row.get("City", ""), ""), money(row.get("price_numeric", row.get("Price")))]
    beds = clean_text(row.get("Bedrooms", ""), "")
    baths = clean_text(row.get("Bathrooms", ""), "")
    size = clean_text(row.get("Size", ""), "")
    if beds:
        pieces.append(f"{beds} bed")
    if baths:
        pieces.append(f"{baths} bath")
    if size:
        pieces.append(size)
    return " | ".join([piece for piece in pieces if piece])