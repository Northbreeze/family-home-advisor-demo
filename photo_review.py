from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from data_cleaning import clean_address
from review_store import review_key


PHOTO_REVIEW_COLUMNS = [
    "Address",
    "photo_review_status",
    "ai_yard_playability",
    "ai_yard_type",
    "ai_flatness",
    "ai_fenced",
    "ai_privacy",
    "ai_yard_noise_clues",
    "ai_layout_notes",
    "ai_confidence",
    "ai_photo_notes",
    "ai_yard_score",
    "ai_layout_score",
    "ai_privacy_score",
    "ai_fence_score",
    "ai_slope_score",
    "ai_noise_clue_score",
    "ai_photo_url_count",
]

DEFAULT_PHOTO_REVIEW_VALUES = {
    "photo_review_status": "Not reviewed",
    "ai_yard_playability": "Unknown",
    "ai_yard_type": "Unknown",
    "ai_flatness": "Unknown",
    "ai_fenced": "Unknown",
    "ai_privacy": "Unknown",
    "ai_yard_noise_clues": "Unknown",
    "ai_layout_notes": "",
    "ai_confidence": "",
    "ai_photo_notes": "",
    "ai_yard_score": 50,
    "ai_layout_score": 50,
    "ai_privacy_score": 50,
    "ai_fence_score": 50,
    "ai_slope_score": 50,
    "ai_noise_clue_score": 50,
    "ai_photo_url_count": 0,
}

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REALTOR_CA_SEARCH_ENDPOINT = "https://api2.realtor.ca/Listing.svc/PropertySearch_Post"
REALTOR_CA_DETAILS_ENDPOINT = "https://api2.realtor.ca/Listing.svc/PropertyDetails"
REALTOR_CA_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://www.realtor.ca",
    "Referer": "https://www.realtor.ca/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Host": "api2.realtor.ca",
}


def safe_folder_name(address: Any) -> str:
    key = review_key(address)
    key = re.sub(r"[^A-Z0-9]+", "_", key).strip("_")
    return key[:80] or "UNKNOWN_LISTING"


def listing_photo_dir(root: Path, address: Any) -> Path:
    return root / "listing_photos" / safe_folder_name(address)


def save_uploaded_photos(root: Path, address: Any, uploaded_files: list[Any]) -> list[Path]:
    folder = listing_photo_dir(root, address)
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, uploaded in enumerate(uploaded_files, start=1):
        suffix = Path(uploaded.name).suffix.lower() or ".jpg"
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            continue
        filename = f"photo_{index:02d}{suffix}"
        path = folder / filename
        path.write_bytes(uploaded.getbuffer())
        saved.append(path)
    return saved


def realtor_cookie_session() -> requests.Session:
    session = requests.Session()
    response = requests.post(
        "https://www.realtor.ca/dnight-Exit-shall-Braith-Then-why-vponst-is-proc",
        json={
            "solution": {"interrogation": None, "version": "beta"},
            "old_token": None,
            "error": None,
            "performance": {"interrogation": 1897},
        },
        params={"d": "www.realtor.ca"},
        timeout=15,
    )
    response.raise_for_status()
    token = response.json().get("token")
    if token:
        session.headers.update({"Cookie": f"reese84={token};"})
    return session


def fetch_realtor_photo_urls(mls_number: Any, max_photos: int = 12) -> list[str]:
    """Fetch listing photo URLs from Realtor.ca using the MLS number."""
    mls = str(mls_number or "").strip()
    if not mls:
        return []

    payload = {
        "Version": "7.0",
        "ApplicationId": "1",
        "CultureId": "1",
        "Currency": "CAD",
        "PropertyTypeGroupID": "1",
        "RecordsPerPage": "1",
        "MaximumResults": "1",
        "TransactionTypeId": "2",
        "ReferenceNumber": mls,
        "IncludeTombstones": "1",
        "IncludePins": "1",
    }
    session = realtor_cookie_session()
    search_response = session.post(REALTOR_CA_SEARCH_ENDPOINT, headers=REALTOR_CA_HEADERS, data=payload, timeout=20)
    search_response.raise_for_status()
    results = search_response.json().get("Results") or []
    if not results:
        return []

    listing_id = results[0].get("Id")
    property_payload = results[0].get("Property") or {}
    photos = property_payload.get("Photo") or []
    if listing_id:
        detail_response = session.get(
            REALTOR_CA_DETAILS_ENDPOINT,
            headers=REALTOR_CA_HEADERS,
            params={"PropertyID": listing_id, "ReferenceNumber": mls, "ApplicationId": "1", "CultureId": "1"},
            timeout=20,
        )
        detail_response.raise_for_status()
        photos = (detail_response.json().get("Property") or {}).get("Photo") or photos

    urls = []
    for photo in photos:
        url = photo.get("HighResPath") or photo.get("MedResPath") or photo.get("LowResPath")
        if url:
            urls.append(url)
    return urls[:max_photos]


def download_listing_photos(root: Path, address: Any, mls_number: Any, max_photos: int = 12) -> tuple[list[Path], str]:
    folder = listing_photo_dir(root, address)
    folder.mkdir(parents=True, exist_ok=True)
    urls = fetch_realtor_photo_urls(mls_number, max_photos=max_photos)
    if not urls:
        return [], "No Realtor.ca photos were returned for this MLS number."

    saved: list[Path] = []
    for index, url in enumerate(urls, start=1):
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            suffix = ".jpg"
        path = folder / f"realtor_photo_{index:02d}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            saved.append(path)
            continue
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        path.write_bytes(response.content)
        saved.append(path)
    return saved, f"Fetched {len(saved)} Realtor.ca photo(s)."


def list_listing_photos(root: Path, address: Any) -> list[Path]:
    folder = listing_photo_dir(root, address)
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES)


def load_photo_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PHOTO_REVIEW_COLUMNS + ["review_key"])
    reviews = pd.read_csv(path)
    for column in PHOTO_REVIEW_COLUMNS:
        if column not in reviews.columns:
            reviews[column] = DEFAULT_PHOTO_REVIEW_VALUES.get(column, "")
    reviews["review_key"] = reviews["Address"].map(review_key)
    return reviews[PHOTO_REVIEW_COLUMNS + ["review_key"]]


def save_photo_reviews(path: Path, reviews: pd.DataFrame) -> None:
    out = reviews.copy()
    for column in PHOTO_REVIEW_COLUMNS:
        if column not in out.columns:
            out[column] = DEFAULT_PHOTO_REVIEW_VALUES.get(column, "")
    out = out[PHOTO_REVIEW_COLUMNS].drop_duplicates(subset=["Address"], keep="last")
    out.to_csv(path, index=False)


def merge_photo_reviews(listings: pd.DataFrame, photo_reviews: pd.DataFrame) -> pd.DataFrame:
    data = listings.copy()
    if "review_key" not in data.columns:
        data["review_key"] = data["Address"].map(review_key)
    if photo_reviews.empty:
        for column, value in DEFAULT_PHOTO_REVIEW_VALUES.items():
            data[column] = value
        return data
    merged = data.merge(photo_reviews.drop(columns=["Address"], errors="ignore"), on="review_key", how="left")
    for column, value in DEFAULT_PHOTO_REVIEW_VALUES.items():
        if column not in merged.columns:
            merged[column] = value
        merged[column] = merged[column].fillna(value)
    return merged


def upsert_photo_review(path: Path, address: str, values: dict[str, Any]) -> None:
    reviews = load_photo_reviews(path)
    key = review_key(address)
    row = {"Address": clean_address(address)}
    row.update(DEFAULT_PHOTO_REVIEW_VALUES)
    row.update(values)
    if "review_key" not in reviews.columns:
        reviews["review_key"] = reviews["Address"].map(review_key)
    reviews = reviews[reviews["review_key"] != key]
    updated = pd.concat([reviews.drop(columns=["review_key"], errors="ignore"), pd.DataFrame([row])], ignore_index=True)
    save_photo_reviews(path, updated)


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def openai_ready() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def coerce_score(value: Any, default: int = 50) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, score))


def parse_ai_photo_response(text: str, photo_count: int) -> dict[str, Any]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        parsed = {"notes": text, "yard_playability": "Unknown", "confidence": ""}
    return {
        "photo_review_status": "AI reviewed",
        "ai_yard_playability": parsed.get("yard_playability", "Unknown"),
        "ai_yard_type": parsed.get("yard_type", "Unknown"),
        "ai_flatness": parsed.get("flatness", "Unknown"),
        "ai_fenced": parsed.get("fenced", "Unknown"),
        "ai_privacy": parsed.get("privacy", "Unknown"),
        "ai_yard_noise_clues": parsed.get("yard_noise_clues", "Unknown"),
        "ai_layout_notes": parsed.get("layout_notes", ""),
        "ai_confidence": parsed.get("confidence", ""),
        "ai_photo_notes": parsed.get("notes", ""),
        "ai_yard_score": coerce_score(parsed.get("yard_score")),
        "ai_layout_score": coerce_score(parsed.get("layout_score")),
        "ai_privacy_score": coerce_score(parsed.get("privacy_score")),
        "ai_fence_score": coerce_score(parsed.get("fence_score")),
        "ai_slope_score": coerce_score(parsed.get("slope_score")),
        "ai_noise_clue_score": coerce_score(parsed.get("noise_clue_score")),
        "ai_photo_url_count": photo_count,
    }


def photo_analysis_prompt() -> str:
    return (
        "You are helping a realtor screen family homes from listing photos. "
        "Return only JSON with these keys: yard_playability, yard_type, flatness, fenced, privacy, "
        "yard_noise_clues, layout_notes, confidence, notes, yard_score, layout_score, privacy_score, "
        "fence_score, slope_score, noise_clue_score. "
        "Scores must be integers from 0 to 100. yard_score means toddler-playable outdoor space; "
        "layout_score means visible family-friendly layout; privacy_score means private/secluded yard; "
        "fence_score means fenced/contained play area; slope_score means flat usable outdoor space where 100 is flat and 0 is steep; "
        "noise_clue_score means visual quietness where 100 has no visible road/highway concern and 0 has strong visible road/highway concern. "
        "Use cautious labels. yard_playability must be Great, Maybe, Poor, or Unknown. "
        "If photos do not show a category clearly, use 50 and explain uncertainty in notes."
    )


def analyze_listing_photo_urls(photo_urls: list[str], model: str = "gpt-4.1-mini") -> dict[str, Any]:
    if not openai_ready():
        raise RuntimeError("OPENAI_API_KEY is not configured. Add a key to run AI photo review.")
    if not photo_urls:
        raise RuntimeError("No listing photo URLs are available for this listing.")

    from openai import OpenAI

    client = OpenAI()
    content: list[dict[str, Any]] = [{"type": "input_text", "text": photo_analysis_prompt()}]
    for url in photo_urls[:12]:
        content.append({"type": "input_image", "image_url": url, "detail": "low"})

    response = client.responses.create(model=model, input=[{"role": "user", "content": content}])
    return parse_ai_photo_response(response.output_text, photo_count=len(photo_urls[:12]))


def analyze_listing_photos(photo_paths: list[Path], model: str = "gpt-4.1-mini") -> dict[str, Any]:
    if not openai_ready():
        raise RuntimeError("OPENAI_API_KEY is not configured. Add a key to run AI photo review.")
    if not photo_paths:
        raise RuntimeError("No listing photos are available for this listing.")

    from openai import OpenAI

    client = OpenAI()
    content: list[dict[str, Any]] = [{"type": "input_text", "text": photo_analysis_prompt()}]
    for path in photo_paths[:8]:
        content.append({"type": "input_image", "image_url": image_to_data_url(path), "detail": "low"})

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )
    return parse_ai_photo_response(response.output_text, photo_count=len(photo_paths[:8]))
