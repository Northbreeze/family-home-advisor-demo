from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd


TODAY = pd.Timestamp.today().normalize()

DEFAULT_INPUT_CANDIDATES = [
    "family_home_advisor_client_report.xlsx",
    "houses_client_ready_school_scores.xlsx",
    "houses_with_school_catchments_latest_scores.xlsx",
]

MAJOR_ROAD_TERMS = {
    "HIGHWAY": "Highway",
    "HWY": "Highway",
    "MARINE DRIVE": "Marine Drive",
    "MARINE DR": "Marine Drive",
    "TAYLOR WAY": "Taylor Way",
    "CAPILANO ROAD": "Capilano Road",
    "CAPILANO RD": "Capilano Road",
    "LONSDALE": "Lonsdale Avenue",
    "KEITH ROAD": "Keith Road",
    "KEITH RD": "Keith Road",
    "15TH STREET": "15th Street",
    "15TH ST": "15th Street",
    "3RD STREET": "3rd Street",
    "3RD ST": "3rd Street",
}


def parse_number(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    match = re.findall(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match[0]) if match else float("nan")


def money(value: Any) -> str:
    number = parse_number(value)
    return "N/A" if pd.isna(number) else f"${number:,.0f}"


def clean_address(address: Any) -> str:
    if pd.isna(address):
        return ""
    return str(address).replace("|", ", ")


def google_maps_link(row: pd.Series) -> str:
    lat = row.get("Latitude")
    lon = row.get("Longitude")
    if pd.notna(lat) and pd.notna(lon):
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(clean_address(row.get('Address', '')))}"


def extract_coords_from_map_link(value: Any) -> tuple[float, float]:
    if pd.isna(value):
        return float("nan"), float("nan")
    match = re.search(r"query=(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)", str(value))
    if not match:
        return float("nan"), float("nan")
    return float(match.group(1)), float(match.group(2))


def find_default_input(root: Path) -> Path | None:
    for filename in DEFAULT_INPUT_CANDIDATES:
        path = root / filename
        if path.exists():
            return path

    raw_files = sorted(
        root.glob("North_West_Vancouver_Houses_Open_Houses_WORKING_URLS_*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return raw_files[0] if raw_files else None


def choose_listing_sheet(path: Path) -> str:
    xl = pd.ExcelFile(path)
    for sheet in ["All_Listings", "Personalized_Ranking", "Sheet1"]:
        if sheet in xl.sheet_names:
            return sheet
    return xl.sheet_names[0]


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    warnings: list[str] = []

    required_defaults: dict[str, Any] = {
        "Address": "",
        "City": "",
        "Price": float("nan"),
        "Bedrooms": float("nan"),
        "Bathrooms": float("nan"),
        "Size": float("nan"),
        "final_school": "Unknown",
        "final_fraser_score": float("nan"),
        "school_confidence": "Unknown",
        "school_source": "",
        "Open House": "",
        "Listing URL": "",
        "Latitude": float("nan"),
        "Longitude": float("nan"),
        "bc_assessment_total_value": float("nan"),
        "bc_assessment_land_value": float("nan"),
        "bc_assessment_building_value": float("nan"),
        "bc_assessment_year": float("nan"),
        "bc_assessment_status": "Not available",
        "BC Assessment Search Link": "",
    }

    if "Price" not in data.columns and "_price_numeric" in data.columns:
        data["Price"] = data["_price_numeric"]
    if "Map Link" in data.columns and "Google Maps Link" not in data.columns:
        data["Google Maps Link"] = data["Map Link"]

    for column, default in required_defaults.items():
        if column not in data.columns:
            data[column] = default
            warnings.append(f"Missing `{column}`; using a safe default.")

    data["Address"] = data["Address"].apply(clean_address)
    data["price_numeric"] = data["Price"].apply(parse_number)
    data["bedrooms_numeric"] = data["Bedrooms"].apply(parse_number)
    data["bathrooms_numeric"] = data["Bathrooms"].apply(parse_number)
    data["size_sqft"] = data["Size"].apply(parse_number)
    data["fraser_score_numeric"] = data["final_fraser_score"].apply(parse_number)
    data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
    data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")

    map_source = "Google Maps Link" if "Google Maps Link" in data.columns else "Map Link" if "Map Link" in data.columns else None
    if map_source:
        missing_coords = data["Latitude"].isna() | data["Longitude"].isna()
        if missing_coords.any():
            coords = data.loc[missing_coords, map_source].apply(extract_coords_from_map_link)
            data.loc[missing_coords, "Latitude"] = coords.apply(lambda item: item[0]).astype(float)
            data.loc[missing_coords, "Longitude"] = coords.apply(lambda item: item[1]).astype(float)

    if "Google Maps Link" not in data.columns:
        data["Google Maps Link"] = data.apply(google_maps_link, axis=1)
    else:
        missing_maps = data["Google Maps Link"].isna() | (data["Google Maps Link"].astype(str).str.len() == 0)
        data.loc[missing_maps, "Google Maps Link"] = data[missing_maps].apply(google_maps_link, axis=1)

    return data, warnings


def parse_open_house_events(raw_value: Any, today: pd.Timestamp = TODAY) -> dict[str, Any]:
    raw = "" if pd.isna(raw_value) else str(raw_value).strip()
    if not raw:
        return {"open_house_raw": "", "next_open_house": pd.NaT, "last_open_house": pd.NaT, "open_house_status": "None"}

    events: list[pd.Timestamp] = []
    for part in re.split(r";|\n", raw):
        match = re.search(r"([A-Za-z]{3,9})\s+(\d{1,2})/(\d{2,4})", part)
        if not match:
            continue
        month, day, year = match.groups()
        year_int = int(year) + 2000 if int(year) < 100 else int(year)
        parsed = pd.to_datetime(f"{month} {int(day)} {year_int}", errors="coerce")
        if pd.notna(parsed):
            events.append(parsed.normalize())

    if not events:
        return {"open_house_raw": raw, "next_open_house": pd.NaT, "last_open_house": pd.NaT, "open_house_status": "None"}

    upcoming = sorted([event for event in events if event >= today])
    past = sorted([event for event in events if event < today])
    return {
        "open_house_raw": raw,
        "next_open_house": upcoming[0] if upcoming else pd.NaT,
        "last_open_house": past[-1] if past else pd.NaT,
        "open_house_status": "Upcoming" if upcoming else "Past",
    }


def add_open_house_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    parsed = data["Open House"].apply(parse_open_house_events).apply(pd.Series)
    for column in ["open_house_raw", "next_open_house", "last_open_house", "open_house_status"]:
        data[column] = parsed[column]
    return data


def estimate_noise_risk(row: pd.Series) -> tuple[str, str, bool]:
    existing = row.get("noise_risk")
    if pd.notna(existing) and str(existing).strip():
        corridor = row.get("nearest_noise_corridor", "")
        return str(existing).title(), str(corridor) if pd.notna(corridor) else "", False

    distance = parse_number(row.get("distance_to_noise_corridor_m"))
    corridor = row.get("nearest_noise_corridor", "")
    if pd.notna(distance):
        if distance <= 150:
            return "High", str(corridor), False
        if distance <= 350:
            return "Medium", str(corridor), False
        return "Low", str(corridor), False

    address = clean_address(row.get("Address", "")).upper()
    for term, label in MAJOR_ROAD_TERMS.items():
        if term in address:
            risk = "High" if label in {"Highway", "Marine Drive", "Taylor Way"} else "Medium"
            return risk, label, True

    if pd.isna(row.get("Latitude")) or pd.isna(row.get("Longitude")):
        return "Unknown", "", True
    return "Low", "", True


def add_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    estimates = data.apply(estimate_noise_risk, axis=1, result_type="expand")
    data["noise_risk"] = estimates[0]
    data["nearest_noise_corridor"] = estimates[1]
    data["noise_estimated"] = estimates[2]
    return data
