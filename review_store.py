from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_cleaning import clean_address


REVIEW_COLUMNS = [
    "Address",
    "client_status",
    "noise_verified",
    "yard_playability",
    "yard_noise",
    "layout_fit",
    "photo_review_status",
    "photo_yard_playability",
    "photo_yard_type",
    "photo_flatness",
    "photo_fenced",
    "photo_privacy",
    "photo_notes",
    "bc_assessment_total_value",
    "bc_assessment_land_value",
    "bc_assessment_building_value",
    "review_notes",
]

DEFAULT_REVIEW_VALUES = {
    "client_status": "Unreviewed",
    "noise_verified": "Unknown",
    "yard_playability": "Unknown",
    "yard_noise": "Unknown",
    "layout_fit": "Unknown",
    "photo_review_status": "Not reviewed",
    "photo_yard_playability": "Unknown",
    "photo_yard_type": "Unknown",
    "photo_flatness": "Unknown",
    "photo_fenced": "Unknown",
    "photo_privacy": "Unknown",
    "photo_notes": "",
    "bc_assessment_total_value": pd.NA,
    "bc_assessment_land_value": pd.NA,
    "bc_assessment_building_value": pd.NA,
    "review_notes": "",
}


def review_key(address: Any) -> str:
    return clean_address(address).upper().strip()


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    reviews = pd.read_csv(path)
    for column in REVIEW_COLUMNS:
        if column not in reviews.columns:
            reviews[column] = DEFAULT_REVIEW_VALUES.get(column, "")
    reviews["client_status"] = reviews["client_status"].replace({"New": "Unreviewed"})
    reviews["review_key"] = reviews["Address"].map(review_key)
    return reviews[REVIEW_COLUMNS + ["review_key"]]


def save_reviews(path: Path, reviews: pd.DataFrame) -> None:
    out = reviews.copy()
    for column in REVIEW_COLUMNS:
        if column not in out.columns:
            out[column] = DEFAULT_REVIEW_VALUES.get(column, "")
    out = out[REVIEW_COLUMNS].drop_duplicates(subset=["Address"], keep="last")
    out.to_csv(path, index=False)


def merge_reviews(listings: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    data = listings.copy()
    data["review_key"] = data["Address"].map(review_key)

    assessment_columns = ["bc_assessment_total_value", "bc_assessment_land_value", "bc_assessment_building_value"]
    source_assessment = {
        column: pd.to_numeric(data[column], errors="coerce") if column in data.columns else pd.Series(pd.NA, index=data.index)
        for column in assessment_columns
    }

    # Remove stale/default review columns from listing workbooks before merging
    # saved reviews. Otherwise pandas creates _x/_y columns and the saved values
    # can be hidden behind blank placeholders.
    base = data.drop(columns=[column for column in REVIEW_COLUMNS if column != "Address" and column in data.columns], errors="ignore")

    if reviews.empty:
        for column, value in DEFAULT_REVIEW_VALUES.items():
            base[column] = value
        for column in assessment_columns:
            base[column] = source_assessment[column]
        return base

    merged = base.merge(reviews.drop(columns=["Address"], errors="ignore"), on="review_key", how="left")
    for column, value in DEFAULT_REVIEW_VALUES.items():
        if column not in merged.columns:
            merged[column] = value
        merged[column] = merged[column].fillna(value)

    for column in assessment_columns:
        reviewed_values = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = reviewed_values.combine_first(source_assessment[column])
    return merged


def upsert_review(path: Path, address: str, values: dict[str, Any]) -> None:
    reviews = load_reviews(path)
    key = review_key(address)
    row = {"Address": clean_address(address)}
    row.update(DEFAULT_REVIEW_VALUES)
    row.update(values)
    if "review_key" not in reviews.columns:
        reviews["client_status"] = reviews["client_status"].replace({"New": "Unreviewed"})
    reviews["review_key"] = reviews["Address"].map(review_key)
    reviews = reviews[reviews["review_key"] != key]
    updated = pd.concat([reviews.drop(columns=["review_key"], errors="ignore"), pd.DataFrame([row])], ignore_index=True)
    save_reviews(path, updated)
