from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from area_filters import add_area_columns
from data_cleaning import add_noise_columns, add_open_house_columns, choose_listing_sheet, find_default_input, normalize_columns
from photo_review import load_photo_reviews, merge_photo_reviews
from review_store import load_reviews, merge_reviews

BC_ASSESSMENT_SEARCH_URL = "https://www.bcassessment.ca/Property/AssessmentSearch?sp=1"


@dataclass(frozen=True)
class DataPaths:
    root: Path
    reviews_path: Path
    photo_reviews_path: Path
    listing_change_log_path: Path


def default_data_paths(root: Path) -> DataPaths:
    return DataPaths(
        root=root,
        reviews_path=root / "manual_reviews.csv",
        photo_reviews_path=root / "photo_reviews.csv",
        listing_change_log_path=root / "listing_change_log.csv",
    )


def read_workbook(path_text: str, modified_time: float) -> tuple[pd.DataFrame, str, list[str]]:
    path = Path(path_text)
    sheet = choose_listing_sheet(path)
    return pd.read_excel(path, sheet_name=sheet), sheet, []


def source_path(root: Path) -> Path:
    packaged = root / "family_home_advisor_client_report.xlsx"
    if packaged.exists():
        return packaged
    found = find_default_input(root)
    if found is None:
        raise FileNotFoundError("No listing workbook found.")
    return found


def normalize_url(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.replace("https://www.realtor.ca", "", regex=False).str.split("?").str[0].str.rstrip("/")


def addr_key(df: pd.DataFrame) -> pd.Series:
    return df.get("Address", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def add_change_flags(df: pd.DataFrame, listing_change_log_path: Path) -> pd.DataFrame:
    out = df.copy()
    out["is_new_since_last_refresh"] = False
    out["listing_change_status"] = "Existing"
    if not listing_change_log_path.exists():
        return out
    try:
        changes = pd.read_csv(listing_change_log_path)
    except Exception:
        return out
    mask = pd.Series(False, index=out.index)
    if "Listing URL" in out.columns:
        refs = set()
        for col_name in ["Current Listing URL", "Listing URL", "Website"]:
            if col_name in changes.columns:
                refs.update(normalize_url(changes[col_name]))
        refs = {value for value in refs if value and value not in {"nan", "none"}}
        if refs:
            mask = mask | normalize_url(out["Listing URL"]).isin(refs)
    if "Address" in changes.columns:
        mask = mask | addr_key(out).isin(set(addr_key(changes)))
        if "Change Type" in changes.columns:
            status = changes.drop_duplicates("Address", keep="last").set_index(addr_key(changes))["Change Type"].to_dict()
            mapped = addr_key(out).map(status)
            out.loc[mapped.notna(), "listing_change_status"] = mapped[mapped.notna()]
    out["is_new_since_last_refresh"] = mask
    out.loc[mask & out["listing_change_status"].eq("Existing"), "listing_change_status"] = "Changed Since Last Refresh"
    return out


def load_listing_data(paths: DataPaths) -> tuple[pd.DataFrame, Path, list[str]]:
    path = source_path(paths.root)
    raw, _, _ = read_workbook(str(path), path.stat().st_mtime)
    df, warnings = normalize_columns(raw)
    df = add_open_house_columns(df)
    df = add_noise_columns(df)
    df = merge_reviews(df, load_reviews(paths.reviews_path))
    df = merge_photo_reviews(df, load_photo_reviews(paths.photo_reviews_path))
    df = add_change_flags(df, paths.listing_change_log_path)
    df["BC Assessment Search Link"] = BC_ASSESSMENT_SEARCH_URL
    df = add_area_columns(df)
    return df, path, warnings
