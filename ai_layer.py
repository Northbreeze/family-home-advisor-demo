from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from data_cleaning import money
from home_evaluation import cache_key, evidence_dataframe, facts_dataframe, fallback_evaluation, get_home_evaluation, load_cache, known_facts, score_breakdown
from photo_review import openai_ready
from scoring_layer import size_sqft_value
from v2_product import card_title

CHAT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "inferred_filters": {"type": "array", "items": {"type": "string"}},
        "ranking_preferences": {"type": "array", "items": {"type": "string"}},
        "homes_referenced": {"type": "array", "items": {"type": "string"}},
        "needs_user_followup": {"type": "boolean"},
    },
    "required": ["answer", "inferred_filters", "ranking_preferences", "homes_referenced", "needs_user_followup"],
}


def configure_openai_key(root: Path, secrets: Any | None = None) -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    if not (root / ".streamlit" / "secrets.toml").exists() and not (Path.home() / ".streamlit" / "secrets.toml").exists():
        return
    if secrets is None:
        return
    try:
        key = secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = str(key)


def parse_chat(prompt: str, current: dict[str, Any]) -> tuple[dict[str, Any], str, list[str], list[str]]:
    text = prompt.lower()
    updates = dict(current)
    notes: list[str] = []
    inferred_filters: list[str] = []
    ranking_preferences: list[str] = []
    match = re.search(r"under\s*\$?\s*(\d+(?:\.\d+)?)\s*([mk])?", text)
    if match:
        number = float(match.group(1))
        updates["max_price"] = int(number * (1_000_000 if (match.group(2) or "m") == "m" else 1_000))
        notes.append(f"I will keep the search under {money(updates['max_price'])}.")
        inferred_filters.append(f"Max price: {money(updates['max_price'])}")
    if any(x in text for x in ["good school", "strong school", "top school"]):
        updates["min_school"] = max(float(updates.get("min_school", 0)), 8.0)
        notes.append("I will prioritize stronger schools.")
        ranking_preferences.append("Stronger school catchments")
    if any(x in text for x in ["busy road", "highway", "noise", "quiet"]):
        updates["avoid_noise"] = True
        notes.append("I will treat road/highway noise as a major concern.")
        inferred_filters.append("Avoid high-noise homes")
        ranking_preferences.append("Quiet streets")
    if any(x in text for x in ["toddler", "yard", "backyard", "play"]):
        updates["yard_focus"] = True
        notes.append("I will give more weight to usable outdoor space.")
        ranking_preferences.append("Usable yard for a toddler")
    if any(x in text for x in ["steep", "driveway", "slope"]):
        updates["avoid_slope"] = True
        notes.append("I will flag slope or steep driveway risk for verification.")
        ranking_preferences.append("Avoid steep driveway or unusable slope")
    if any(x in text for x in ["size", "sqft", "square", "interior", "small"]):
        notes.append("For size questions, select a home and use Ask AI about this home so I can answer from that listing.")
        ranking_preferences.append("Interior size and usable layout")
    if "compare" in text:
        notes.append("I will compare the named homes when they are visible in the current results.")
    if not notes:
        notes.append("I saved that as a preference note. I can translate price, school, yard, noise, and comparison requests into search behavior.")
    return updates, " ".join(notes), inferred_filters, ranking_preferences


def visible_listing_context(visible: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    context_cols = ["Address", "City", "detected_area", "price_numeric", "Bedrooms", "Size", "match_score", "noise_risk", "yard_playability", "final_school", "final_fraser_score", "interior_size_rating", "family_evaluation_note"]
    if visible.empty:
        return []
    return visible.head(limit)[[col for col in context_cols if col in visible.columns]].to_dict("records")


def normalize_chat_response(value: dict[str, Any], fallback_answer: str) -> dict[str, Any]:
    return {
        "answer": str(value.get("answer") or fallback_answer),
        "inferred_filters": value.get("inferred_filters") if isinstance(value.get("inferred_filters"), list) else [],
        "ranking_preferences": value.get("ranking_preferences") if isinstance(value.get("ranking_preferences"), list) else [],
        "homes_referenced": value.get("homes_referenced") if isinstance(value.get("homes_referenced"), list) else [],
        "needs_user_followup": bool(value.get("needs_user_followup", False)),
    }


def fallback_chat_response(answer: str, inferred_filters: list[str], ranking_preferences: list[str]) -> dict[str, Any]:
    return {
        "answer": answer,
        "inferred_filters": inferred_filters,
        "ranking_preferences": ranking_preferences,
        "homes_referenced": [],
        "needs_user_followup": False,
    }

def call_openai_chat(prompt: str, visible: pd.DataFrame, fallback_answer: str, inferred_filters: list[str], ranking_preferences: list[str]) -> dict[str, Any] | None:
    if not openai_ready():
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        context = visible_listing_context(visible)
        requested_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        fallback_models = [requested_model]
        if requested_model != "gpt-4o-mini":
            fallback_models.append("gpt-4o-mini")
        payload = {
            "question": prompt,
            "visible_listings": context,
            "rule_based_interpretation": fallback_answer,
            "inferred_filters": inferred_filters,
            "ranking_preferences": ranking_preferences,
        }
        for model_name in fallback_models:
            try:
                result = client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "system", "content": "You are a concise AI home-buying assistant. Use only supplied listing facts. Return JSON only."},
                        {"role": "user", "content": json.dumps(payload, default=str)},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "family_home_chat_response",
                            "schema": CHAT_RESPONSE_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                parsed = json.loads(result.output_text)
                if isinstance(parsed, dict):
                    return normalize_chat_response(parsed, fallback_answer)
            except Exception:
                continue
    except Exception:
        return None
    return None


def answer_chat(prompt: str, visible: pd.DataFrame, current_rules: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    rules, note, inferred_filters, ranking_preferences = parse_chat(prompt, current_rules or {})
    response = call_openai_chat(prompt, visible, note, inferred_filters, ranking_preferences)
    if response is None:
        response = fallback_chat_response(note, inferred_filters, ranking_preferences)
    return rules, response


def answer_home_question(prompt: str, row: pd.Series, visible: pd.DataFrame, current_rules: dict[str, Any] | None = None) -> str:
    text = prompt.lower()
    address = card_title(row)
    size = size_sqft_value(row)
    if any(word in text for word in ["size", "sqft", "square", "small", "interior", "space"]):
        if pd.isna(size):
            return f"For {address}, I do not have a reliable interior sqft value in the current listing data, so this needs manual verification."
        if size < 1600:
            return f"{address} is about {size:,.0f} sqft. For a family, that is a serious constraint, so I cap its score and treat it as worth considering only if the layout works unusually well or expansion is realistic."
        if size < 1900:
            return f"{address} is about {size:,.0f} sqft. That is modest for family life, so I would verify bedroom layout, storage, and main-floor flow before prioritizing it."
        return f"{address} is about {size:,.0f} sqft, which is more comfortable from an interior-size perspective. I would still verify layout because sqft alone does not prove family usability."
    if any(word in text for word in ["yard", "backyard", "toddler", "play"]):
        yard = str(row.get("yard_playability", "Unknown"))
        yard_score = pd.to_numeric(row.get("backyard_component", 0), errors="coerce")
        return f"For {address}, yard status is {yard}. The current yard signal score is {yard_score if pd.notna(yard_score) else 'unknown'}. This is still estimated unless photos/manual review confirm a flat usable play area."
    if any(word in text for word in ["noise", "highway", "road", "quiet"]):
        return f"For {address}, noise risk is {row.get('noise_risk', 'Unknown')}. The app also tracks highway distance as {row.get('distance_to_highway_m', 'unknown')} m and major-road distance as {row.get('distance_to_major_road_m', 'unknown')} m. This should be verified outside and in the backyard."
    if any(word in text for word in ["school", "catchment"]):
        return f"For {address}, the current school is {row.get('final_school', 'Unknown')} with Fraser score {row.get('final_fraser_score', 'unknown')}."
    if any(word in text for word in ["price", "value", "assessment", "deal"]):
        return f"For {address}, price is {money(row.get('price_numeric', row.get('Price')))}. BC Assessment total is {money(row.get('bc_assessment_total_value'))}, if entered. Value confidence is limited when assessment data is missing."
    _, response = answer_chat(f"About {row.get('Address')}: {prompt}", visible, current_rules)
    return response.get("answer", "I saved that as a preference note.")


def get_cached_or_fallback_listing_evaluation(row: pd.Series, preferences: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    key = cache_key(row, preferences)
    cache = load_cache(cache_path)
    cached = cache.get(key)
    if isinstance(cached, dict):
        cached.setdefault("known_facts", known_facts(row))
        cached.setdefault("rule_evidence", score_breakdown(row))
        return cached
    evaluation = fallback_evaluation(row, preferences)
    evaluation["source"] = "Instant rule-based review"
    return evaluation


def get_listing_evaluation(row: pd.Series, preferences: dict[str, Any], cache_path: Path, force: bool = False, model: str | None = None) -> dict[str, Any]:
    return get_home_evaluation(row, preferences, cache_path, force=force, model=model)
