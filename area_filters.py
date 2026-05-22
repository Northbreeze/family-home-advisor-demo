from __future__ import annotations

import math
from typing import Any

import pandas as pd


AREA_KEYWORDS: dict[str, list[str]] = {
    "Ambleside": ["ambleside"],
    "Dundarave": ["dundarave"],
    "Caulfeild": ["caulfeild", "caulfeild village"],
    "Sentinel Hill": ["sentinel hill"],
    "Edgemont": ["edgemont", "edgemont village"],
    "Delbrook": ["delbrook"],
    "Lynn Valley": ["lynn valley"],
    "Canyon Heights": ["canyon heights"],
    "Upper Lonsdale": ["upper lonsdale"],
    "Central Lonsdale": ["central lonsdale"],
    "Lower Lonsdale": ["lower lonsdale"],
    "British Properties": ["british properties"],
}


LANDMARKS: dict[str, tuple[float, float]] = {
    "None": (math.nan, math.nan),
    "Edgemont Village": (49.3349, -123.1027),
    "Ambleside": (49.3282, -123.1579),
    "Dundarave Village": (49.3354, -123.1836),
    "Caulfeild Village": (49.3495, -123.2548),
    "Lynn Valley Centre": (49.3367, -123.0394),
    "Park Royal": (49.3266, -123.1377),
}


def detect_area(row: pd.Series) -> str:
    text_parts = [
        row.get("Address", ""),
        row.get("Description", ""),
        row.get("Nearby Ammenities", ""),
        row.get("location_flags", ""),
    ]
    text = " ".join("" if pd.isna(part) else str(part) for part in text_parts).lower()
    for area, keywords in AREA_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return area
    city = str(row.get("City", "") or "").strip()
    return city if city else "Unknown"


def haversine_km(lat1: Any, lon1: Any, lat2: float, lon2: float) -> float:
    try:
        lat1_f = float(lat1)
        lon1_f = float(lon1)
    except (TypeError, ValueError):
        return math.nan
    if any(math.isnan(value) for value in [lat1_f, lon1_f, lat2, lon2]):
        return math.nan
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1_f)
    dlon = math.radians(lon2 - lon1_f)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1_f)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def add_area_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["detected_area"] = data.apply(detect_area, axis=1)
    for landmark, (lat, lon) in LANDMARKS.items():
        if landmark == "None":
            continue
        column = f"distance_to_{landmark.lower().replace(' ', '_')}_km"
        data[column] = data.apply(lambda row: haversine_km(row.get("Latitude"), row.get("Longitude"), lat, lon), axis=1)
    return data


def filter_by_area(df: pd.DataFrame, preferred_area: str, landmark: str, radius_km: float) -> pd.DataFrame:
    data = df.copy()
    if preferred_area and preferred_area != "All":
        data = data[data["detected_area"].eq(preferred_area)]
    if landmark and landmark != "None":
        column = f"distance_to_{landmark.lower().replace(' ', '_')}_km"
        if column in data.columns:
            data = data[pd.to_numeric(data[column], errors="coerce").le(radius_km)]
    return data
