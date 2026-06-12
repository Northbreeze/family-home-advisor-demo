from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data_cleaning import money

EVALUATION_VERSION = "family-home-evaluation-v1"
RECOMMENDATIONS = [
    "Strongly consider",
    "Consider but verify",
    "Tour only if price/location fit",
    "Avoid unless discounted",
]


def clean_value(value: object, default: str = "Unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else default


def number(value: object) -> float:
    return pd.to_numeric(value, errors="coerce")


def get_first(row: pd.Series, names: list[str], default: object = "Unknown") -> object:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}:
                return value
    return default


def score_value(row: pd.Series) -> float:
    value = number(row.get("match_score"))
    return 0.0 if pd.isna(value) else float(value)


def size_value(row: pd.Series) -> float:
    for name in ["interior_size_sqft", "size_sqft", "sqft_numeric"]:
        value = number(row.get(name))
        if pd.notna(value) and float(value) > 0:
            return float(value)
    text = clean_value(row.get("Size", ""), "")
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else float("nan")
    except ValueError:
        return float("nan")


def size_label(size: float) -> tuple[str, str, float | None]:
    if pd.isna(size) or size <= 0:
        return "Unknown", "Interior size is missing; verify listing sqft.", None
    if size < 1600:
        return "Major concern", f"Interior size is only {size:,.0f} sqft, which can be a long-term family space constraint.", 58
    if size < 1900:
        return "Modest", f"Interior size is {size:,.0f} sqft; layout efficiency matters a lot.", 70
    if size < 2400:
        return "Acceptable", f"Interior size is {size:,.0f} sqft; likely workable if the plan is efficient.", None
    if size < 3200:
        return "Family-friendly", f"Interior size is {size:,.0f} sqft, which looks family-friendly.", None
    return "Spacious", f"Interior size is {size:,.0f} sqft, which is spacious for family use.", None


def known_facts(row: pd.Series) -> list[dict[str, str]]:
    size = size_value(row)
    rating = clean_value(row.get("interior_size_rating", ""), "")
    size_note = clean_value(row.get("interior_size_note", ""), "")
    if not rating:
        rating, size_note, _ = size_label(size)
    facts = [
        ("Address", get_first(row, ["Address"])),
        ("City", get_first(row, ["City"])),
        ("Neighbourhood / area", get_first(row, ["detected_area", "Neighbourhood", "Area"])),
        ("Price", money(get_first(row, ["price_numeric", "Price"], None))),
        ("Bedrooms", get_first(row, ["Bedrooms", "bedrooms_numeric"])),
        ("Bathrooms", get_first(row, ["Bathrooms"])),
        ("Interior size", f"{size:,.0f} sqft" if pd.notna(size) else "Unknown"),
        ("Interior size rating", rating),
        ("Interior size note", size_note),
        ("School", get_first(row, ["final_school"])),
        ("Fraser score", get_first(row, ["final_fraser_score", "fraser_score_numeric"])),
        ("Noise risk", get_first(row, ["noise_risk"])),
        ("Highway distance", get_first(row, ["distance_to_highway_m"], "Unknown")),
        ("Major road distance", get_first(row, ["distance_to_major_road_m"], "Unknown")),
        ("Yard status", get_first(row, ["yard_playability", "photo_yard_playability"], "Unknown")),
        ("Layout status", get_first(row, ["layout_fit"], "Unknown")),
        ("Open house", get_first(row, ["open_house_status", "Open House"], "Unknown")),
        ("BC Assessment", money(get_first(row, ["bc_assessment_total_value"], None))),
        ("Price per sqft", money(get_first(row, ["price_per_sqft"], None))),
        ("Year built", get_first(row, ["Year Built", "year_built", "Built In"], "Unknown")),
        ("Lot size", get_first(row, ["Lot Size", "Lot", "lot_size"], "Unknown")),
    ]
    return [{"Fact": key, "Value": clean_value(value)} for key, value in facts]


def score_breakdown(row: pd.Series) -> list[dict[str, str]]:
    factors = [
        ("Overall score", row.get("match_score"), "Backbone score from school, price/value, quiet, size, and lifestyle signals."),
        ("School", row.get("school_component"), "School quality and confidence."),
        ("Quiet", row.get("quiet_component"), "Noise risk from road/highway fields and overrides."),
        ("Value", row.get("price_component"), "Budget fit, price per sqft, and BC Assessment when available."),
        ("Interior size", row.get("size_component"), "Bedrooms and interior sqft fit."),
        ("Yard", row.get("backyard_component"), "Listing/manual/AI yard signals."),
        ("Layout", row.get("layout_component"), "Listing/manual/AI layout signals."),
        ("Location", row.get("location_component"), "Neighbourhood/location lifestyle signals."),
        ("Condition", row.get("condition_component"), "Listing text condition and renovation signals."),
    ]
    rows: list[dict[str, str]] = []
    for factor, value, meaning in factors:
        numeric = number(value)
        rows.append({
            "Factor": factor,
            "Score / value": "Unknown" if pd.isna(numeric) else f"{numeric:.1f}",
            "Meaning": meaning,
        })
    note = clean_value(row.get("family_evaluation_note", ""), "")
    if note:
        rows.append({"Factor": "Family-fit caps", "Score / value": "Applied", "Meaning": note})
    return rows


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def cache_key(row: pd.Series, preferences: dict[str, Any]) -> str:
    payload = {
        "version": EVALUATION_VERSION,
        "address": clean_value(row.get("Address", "")),
        "url": clean_value(row.get("Listing URL", "")),
        "mls": clean_value(row.get("MLS", "")),
        "score": score_value(row),
        "facts": known_facts(row),
        "evidence": score_breakdown(row),
        "preferences": preferences,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recommendation_label(row: pd.Series) -> str:
    score = score_value(row)
    noise = clean_value(row.get("noise_risk", "Unknown"))
    yard = clean_value(row.get("yard_playability", "Unknown"))
    rating = clean_value(row.get("interior_size_rating", "Unknown"))
    if noise == "High" or yard == "Poor" or score < 55:
        return "Avoid unless discounted"
    if rating == "Major concern" or score < 68:
        return "Tour only if price/location fit"
    if score >= 82 and noise != "High":
        return "Strongly consider"
    return "Consider but verify"


def verdict_title(row: pd.Series) -> str:
    label = recommendation_label(row)
    rating = clean_value(row.get("interior_size_rating", "Unknown"))
    noise = clean_value(row.get("noise_risk", "Unknown"))
    yard = clean_value(row.get("yard_playability", "Unknown"))
    checks: list[str] = []
    if rating in {"Major concern", "Modest", "Unknown"}:
        checks.append("Verify Interior Size")
    if yard in {"Unknown", "Maybe"}:
        checks.append("Verify Yard")
    if noise in {"Medium", "High", "Unknown"}:
        checks.append("Verify Noise")
    if pd.isna(number(row.get("bc_assessment_total_value"))):
        checks.append("BC Assessment")
    suffix = " and ".join(checks[:2]) if checks else "Good Evidence"
    base = {
        "Strongly consider": "Strong Family Fit",
        "Consider but verify": "Promising Family Fit",
        "Tour only if price/location fit": "Conditional Fit",
        "Avoid unless discounted": "Weak Fit",
    }[label]
    return f"{base} - {suffix}"


def fallback_evaluation(row: pd.Series, preferences: dict[str, Any]) -> dict[str, Any]:
    score = score_value(row)
    facts = {item["Fact"]: item["Value"] for item in known_facts(row)}
    rating = facts.get("Interior size rating", "Unknown")
    noise = facts.get("Noise risk", "Unknown")
    school = facts.get("School", "Unknown")
    school_score = facts.get("Fraser score", "Unknown")
    area = facts.get("Neighbourhood / area", "Unknown")
    city = facts.get("City", "Unknown")
    yard = facts.get("Yard status", "Unknown")
    bc = facts.get("BC Assessment", "$0")
    price_psf = facts.get("Price per sqft", "$0")
    size_note = facts.get("Interior size note", "Verify interior size.")
    label = recommendation_label(row)

    strengths = []
    if city != "Unknown" or area != "Unknown":
        strengths.append(f"The {city if city != 'Unknown' else 'local'} location and {area if area != 'Unknown' else 'neighbourhood'} context may support a family lifestyle and resale demand, but the exact street feel should be checked in person.")
    if school != "Unknown":
        strengths.append(f"The listed school pathway is {school} with Fraser score {school_score}, which can support long-term family planning if the catchment is verified.")
    if noise == "Low":
        strengths.append("The current noise model is low, which is important for indoor comfort and backyard use.")
    if rating in {"Family-friendly", "Spacious", "Acceptable"}:
        strengths.append(size_note)
    if yard in {"Great", "Maybe"}:
        strengths.append("There is at least some positive yard signal, which matters for outdoor family use, but photos/showing should confirm flat usable space.")
    if len(strengths) < 4:
        strengths.append("The rule-based score keeps this home in consideration, but the decision should depend on showing-level evidence rather than score alone.")

    risks = []
    if rating in {"Major concern", "Modest", "Unknown"}:
        risks.append(size_note)
    if yard in {"Unknown", "Maybe"}:
        risks.append("Yard usability is not proven from the current data, so verify whether there is a flat, safe toddler play area.")
    if noise in {"Medium", "High", "Unknown"}:
        risks.append(f"Noise risk is {noise.lower()}, so road and backyard noise should be tested before prioritizing the home.")
    if bc in {"$0", "Unknown"}:
        risks.append("BC Assessment is not verified, so land/building value and price discipline need manual review.")
    if price_psf not in {"$0", "Unknown"}:
        risks.append(f"Price per sqft is {price_psf}; confirm whether the premium is justified by land, school, or neighbourhood rather than interior space.")
    if len(risks) < 4:
        risks.append("Older-home systems, permits, roof, drainage, electrical, and plumbing should be verified before any offer.")

    summary = (
        f"This home scores {score:.0f}/100. It looks like a {label.lower()} because the score supports further review, "
        f"but the family decision depends on the trade-off between location, school, land/yard potential, interior size, and missing verification items."
    )
    return {
        "source": "Rule-based narrative fallback",
        "model": "none",
        "overall_verdict": verdict_title(row),
        "score": round(score, 1),
        "overall_summary": summary,
        "best_for": [
            "buyers who want an evidence-based family fit check before touring",
            "families who care about school, quiet streets, and usable outdoor space",
            "buyers comfortable verifying missing items instead of trusting listing language",
        ],
        "strengths": strengths[:6],
        "tradeoffs_risks": risks[:6],
        "family_fit_scenarios": {
            "excellent_fit_if": [
                "the layout feels larger than the raw sqft suggests",
                "the yard is flat, safe, and usable for children",
                "road noise is low from both inside the home and backyard",
            ],
            "less_ideal_if": [
                "the family needs more bedrooms, storage, or work-from-home space",
                "the home requires major system upgrades beyond budget",
                "the value is mostly location/land while the house itself feels compromised",
            ],
        },
        "investment_resale_view": (
            "Value is most likely driven by a mix of neighbourhood desirability, school pathway, land/outdoor potential, and the condition of the existing structure. "
            "If BC Assessment or comparable sales are missing, verify them before treating the home as good value."
        ),
        "verify_checklist": [
            "Confirm flat, usable yard and privacy.",
            "Walk the layout and check whether the interior size works for daily family life.",
            "Stand inside and outside to verify road/highway noise.",
            "Inspect roof age, drainage, electrical, plumbing, heating, and signs of deferred maintenance.",
            "Check permits for renovations, suite potential, decks, and additions.",
            "Verify BC Assessment land/building values and recent comparable sales.",
        ],
        "ai_recommendation": label,
        "known_facts": known_facts(row),
        "rule_evidence": score_breakdown(row),
    }


def build_prompt(row: pd.Series, preferences: dict[str, Any]) -> str:
    payload = {
        "known_facts": known_facts(row),
        "rule_evidence": score_breakdown(row),
        "user_preferences": preferences,
        "allowed_recommendations": RECOMMENDATIONS,
    }
    return f"""
You are an AI real estate decision advisor for a family evaluating homes in North/West Vancouver.
Use only the known facts and rule evidence below. Do not invent facts. If a fact is missing, say verify.
Separate known facts from interpretation. Be practical, plain-English, and specific.
Return ONLY valid JSON with these keys:
overall_verdict, score, overall_summary, best_for, strengths, tradeoffs_risks, family_fit_scenarios, investment_resale_view, verify_checklist, ai_recommendation.
- best_for, strengths, tradeoffs_risks, verify_checklist must be arrays of strings.
- family_fit_scenarios must have excellent_fit_if and less_ideal_if arrays.
- ai_recommendation must be one of: {RECOMMENDATIONS}.
- score must use the provided rule-based score, not a new score.
Data:
{json.dumps(payload, indent=2, default=str)}
""".strip()


def normalize_evaluation(result: dict[str, Any], row: pd.Series, source: str, model: str) -> dict[str, Any]:
    fallback = fallback_evaluation(row, {})
    normalized = dict(fallback)
    normalized.update({key: value for key, value in result.items() if value not in [None, ""]})
    normalized["source"] = source
    normalized["model"] = model
    normalized["score"] = score_value(row)
    normalized["known_facts"] = known_facts(row)
    normalized["rule_evidence"] = score_breakdown(row)
    if normalized.get("ai_recommendation") not in RECOMMENDATIONS:
        normalized["ai_recommendation"] = recommendation_label(row)
    for key in ["best_for", "strengths", "tradeoffs_risks", "verify_checklist"]:
        if not isinstance(normalized.get(key), list):
            normalized[key] = fallback[key]
    scenarios = normalized.get("family_fit_scenarios")
    if not isinstance(scenarios, dict):
        normalized["family_fit_scenarios"] = fallback["family_fit_scenarios"]
    else:
        scenarios.setdefault("excellent_fit_if", fallback["family_fit_scenarios"]["excellent_fit_if"])
        scenarios.setdefault("less_ideal_if", fallback["family_fit_scenarios"]["less_ideal_if"])
    return normalized


def call_openai_evaluation(row: pd.Series, preferences: dict[str, Any], model: str) -> dict[str, Any] | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        result = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You produce JSON-only real estate family-home evaluations from supplied facts. Never invent missing facts."},
                {"role": "user", "content": build_prompt(row, preferences)},
            ],
        )
        text = result.output_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None
        return normalize_evaluation(parsed, row, "OpenAI narrative", model)
    except Exception:
        return None


def get_home_evaluation(
    row: pd.Series,
    preferences: dict[str, Any],
    cache_path: Path,
    force: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    key = cache_key(row, preferences)
    cache = load_cache(cache_path)
    if not force and key in cache:
        cached = cache[key]
        if isinstance(cached, dict):
            cached.setdefault("known_facts", known_facts(row))
            cached.setdefault("rule_evidence", score_breakdown(row))
            return cached

    evaluation = call_openai_evaluation(row, preferences, model_name)
    if evaluation is None and model_name != "gpt-4o-mini":
        evaluation = call_openai_evaluation(row, preferences, "gpt-4o-mini")
    if evaluation is None:
        evaluation = fallback_evaluation(row, preferences)
    cache[key] = evaluation
    save_cache(cache_path, cache)
    return evaluation


def facts_dataframe(evaluation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(evaluation.get("known_facts", []))


def evidence_dataframe(evaluation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(evaluation.get("rule_evidence", []))
