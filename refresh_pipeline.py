from __future__ import annotations

"""Refresh and enrich the Family Home Advisor listing workbook.

This script pulls Realtor.ca listings, joins them to local school catchment
data, adds practical buyer-facing signals, and writes a multi-sheet Excel
report used by the Streamlit app.
"""

import argparse
import re
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from urllib.parse import quote_plus

import numpy as np
import pandas as pd


PROJECTED_CRS = "EPSG:26910"
SCHOOL_COL_NV = "PRIMARY_NA"
TARGET_CITIES = ["North Vancouver", "West Vancouver"]
RAW_DIR_NAME = "_pyrealtor_raw"
MANUAL_LISTINGS_FILE = "manual_listings.csv"

NOISE_CORRIDORS = [
    {
        "name": "Highway 1 / Upper Levels",
        "type": "highway",
        "coords": [
            (-123.286, 49.373), (-123.255, 49.356), (-123.215, 49.348),
            (-123.175, 49.345), (-123.135, 49.340), (-123.095, 49.334),
            (-123.055, 49.325), (-123.020, 49.315),
        ],
    },
    {
        "name": "Marine Drive",
        "type": "major_road",
        "coords": [
            (-123.292, 49.373), (-123.260, 49.355), (-123.215, 49.340),
            (-123.170, 49.333), (-123.135, 49.323), (-123.095, 49.316),
            (-123.060, 49.309),
        ],
    },
    {"name": "Taylor Way", "type": "major_road", "coords": [(-123.150, 49.333), (-123.145, 49.340), (-123.140, 49.350), (-123.135, 49.360)]},
    {"name": "Capilano Road", "type": "major_road", "coords": [(-123.115, 49.305), (-123.116, 49.325), (-123.118, 49.345)]},
    {"name": "Lonsdale Avenue", "type": "major_road", "coords": [(-123.073, 49.307), (-123.073, 49.325), (-123.073, 49.345)]},
]

NOISE_OVERRIDES = [
    {
        "address_contains": "2762 NEWMARKET DRIVE",
        "noise_risk": "Low",
        "note": (
            "Manual geography review: property is north/uphill from Highway 1 with residential/terrain buffer; "
            "verify during showing, especially backyard."
        ),
    },
]

NOISE_BUFFER_TERMS = [
    "quiet street",
    "quiet, family-friendly",
    "quiet family-friendly",
    "cul-de-sac",
    "no-through",
    "residential",
    "edgemont village",
    "family-friendly neighborhood",
    "family friendly neighborhood",
    "trees",
    "greenbelt",
]

NOISE_NEGATIVE_TERMS = [
    "backs onto highway",
    "beside highway",
    "highway noise",
    "traffic noise",
    "highway 1",
    "hwy 1",
    "upper levels",
    "easy access to hwy",
    "easy access to highway",
    "on marine drive",
    "fronts marine drive",
]


def log_default(message: str) -> None:
    """Default status callback used by CLI runs."""
    print(message)


def _norm_col(value: object) -> str:
    """Normalize a column label so loose source-file matching is reliable."""
    return str(value).replace("\xa0", " ").strip().lower()


def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return the first DataFrame column whose normalized name contains any keyword."""
    norm_map = {_norm_col(column): column for column in df.columns}
    for keyword in [str(k).lower() for k in keywords]:
        for normalized, original in norm_map.items():
            if keyword in normalized:
                return original
    return None


def clean_school_name(value: object) -> str | float:
    """Create a comparable school-name key across catchment and score files."""
    if pd.isna(value):
        return np.nan
    text = str(value).upper().strip()
    text = text.replace("É", "E").replace("È", "E").replace("Ê", "E")
    text = re.sub(r"\bECOLE\b", "", text)
    text = re.sub(r"\bELEMENTARY\b", "", text)
    text = re.sub(r"\bCOMMUNITY\b", "", text)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_number(value: object) -> float:
    """Extract the first numeric value from text such as square-footage labels."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"([0-9][0-9,\.]+)", str(value))
    return float(match.group(1).replace(",", "")) if match else np.nan


def clean_price(series: pd.Series) -> pd.Series:
    """Convert Realtor.ca price text into numeric dollar values."""
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def extract_zip_if_needed(zip_path: Path, out_dir: Path) -> Path:
    """Extract a vector-data zip once and reuse the existing extracted folder."""
    if out_dir.exists() and (any(out_dir.rglob("*.shp")) or any(out_dir.rglob("*.geojson"))):
        return out_dir
    if not zip_path.exists():
        raise FileNotFoundError(f"Cannot find {zip_path}.")
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    return out_dir


def read_first_vector_file(path: Path):
    """Read the first supported geospatial file from a path or extracted folder."""
    import geopandas as gpd

    if path.is_file():
        return gpd.read_file(path)
    shp_files = sorted(path.rglob("*.shp"))
    if shp_files:
        return gpd.read_file(shp_files[0])
    geojson_files = sorted(path.rglob("*.geojson"))
    if geojson_files:
        return gpd.read_file(geojson_files[0])
    raise FileNotFoundError(f"No .shp or .geojson file found inside {path}")


def fetch_realtor_listings(
    root: Path,
    cities: list[str],
    max_price: int,
    open_houses_only: bool,
    status: Callable[[str], None] = log_default,
) -> Path:
    """Fetch and lightly filter Realtor.ca listings before saving raw workbook output."""
    import pyRealtor

    raw_dir = root / RAW_DIR_NAME
    raw_dir.mkdir(exist_ok=True)
    output_file = root / f"North_West_Vancouver_Houses_Open_Houses_WORKING_URLS_{date.today().isoformat()}.xlsx"
    house_obj = pyRealtor.HousesFacade()
    all_results: list[pd.DataFrame] = []

    for city in cities:
        status(f"Fetching Realtor.ca listings for {city}...")
        raw_file = raw_dir / f"{city.replace(' ', '_')}_raw.xlsx"
        house_obj.search_save_houses(search_area=city, country="Canada", report_file_name=str(raw_file))
        if not raw_file.exists():
            status(f"No raw file was created for {city}.")
            continue

        df = pd.read_excel(raw_file, sheet_name=1)
        price_col = find_col(df, ["price"])
        open_col = find_col(df, ["open house", "openhouse", "open"])
        website_col = find_col(df, ["website"])
        url_col = find_col(df, ["url", "link"])
        category_col = find_col(df, ["house category", "property type", "ownership category", "ownership"])

        if not price_col:
            status(f"Skipping {city}: no price column found.")
            continue

        # Keep the raw export mostly intact, but remove listings outside the
        # configured buyer budget before downstream enrichment work begins.
        df["_price_numeric"] = clean_price(df[price_col])
        df = df[df["_price_numeric"].notna() & (df["_price_numeric"] <= max_price)].copy()

        if open_houses_only and open_col:
            df = df[df[open_col].notna()].copy()
        if category_col:
            house_mask = df[category_col].astype(str).str.strip().str.lower().eq("house")
            if house_mask.any():
                df = df[house_mask].copy()

        if df.empty:
            status(f"No matching listings kept for {city}.")
            continue

        # Prefer direct Realtor.ca links when the raw export includes relative
        # website paths; fall back to any URL/link column the package provided.
        if website_col:
            website = df[website_col].astype(str).str.strip()
            df["Listing URL"] = np.where(website.str.startswith("http"), website, "https://www.realtor.ca" + website)
        elif url_col:
            df["Listing URL"] = df[url_col].astype(str).str.strip()
        else:
            df["Listing URL"] = ""

        df["City"] = city
        df["Fetched On"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        preferred = ["Listing URL", "City", "Fetched On"]
        df = df[preferred + [column for column in df.columns if column not in preferred]]
        all_results.append(df)
        status(f"Kept {len(df)} listings for {city}.")

    if not all_results:
        raise RuntimeError("No listings were fetched from Realtor.ca.")

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_excel(output_file, index=False)
    status(f"Saved raw refreshed listings: {output_file.name}")
    return output_file


def normalize_change_text(value: object) -> str:
    """Normalize listing identity fields for stable refresh-to-refresh comparisons."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def normalize_change_url(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().split("?")[0].rstrip("/")
    for prefix in ["https://www.realtor.ca", "http://www.realtor.ca", "https://realtor.ca", "http://realtor.ca"]:
        text = text.replace(prefix, "")
    return text


def listing_address_key(df: pd.DataFrame) -> pd.Series:
    if "Address" not in df.columns:
        return pd.Series("", index=df.index)
    return df["Address"].map(normalize_change_text)


def listing_url_key(df: pd.DataFrame) -> pd.Series:
    if "Listing URL" in df.columns:
        return df["Listing URL"].map(normalize_change_url)
    if "Website" in df.columns:
        return df["Website"].map(normalize_change_url)
    return pd.Series("", index=df.index)


def listing_mls_key(df: pd.DataFrame) -> pd.Series:
    if "MLS" not in df.columns:
        return pd.Series("", index=df.index)
    return df["MLS"].map(normalize_change_text)


def find_previous_listing_file(root: Path, current_file: Path) -> Path | None:
    files = sorted(root.glob("North_West_Vancouver_Houses_Open_Houses_WORKING_URLS_*.xlsx"), key=lambda path: path.stat().st_mtime)
    previous = [path for path in files if path.resolve() != current_file.resolve()]
    return previous[-1] if previous else None


def write_listing_change_log(root: Path, current_file: Path, status: Callable[[str], None] = log_default) -> Path:
    """Write a compact change log so the deployed app does not need old Excel workbooks."""
    output_path = root / "listing_change_log.csv"
    current = pd.read_excel(current_file)
    previous_file = find_previous_listing_file(root, current_file)

    current = current.copy()
    current["_address_key"] = listing_address_key(current)
    current["_url_key"] = listing_url_key(current)
    current["_mls_key"] = listing_mls_key(current)
    current["_price_numeric"] = pd.to_numeric(clean_price(current["Price"] if "Price" in current.columns else pd.Series(index=current.index, dtype=object)), errors="coerce")

    rows: list[dict[str, object]] = []
    if previous_file is None:
        status("No previous listing workbook found; writing empty listing_change_log.csv.")
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return output_path

    previous = pd.read_excel(previous_file).copy()
    previous["_address_key"] = listing_address_key(previous)
    previous["_url_key"] = listing_url_key(previous)
    previous["_mls_key"] = listing_mls_key(previous)
    previous["_price_numeric"] = pd.to_numeric(clean_price(previous["Price"] if "Price" in previous.columns else pd.Series(index=previous.index, dtype=object)), errors="coerce")
    previous_by_address = previous.drop_duplicates("_address_key", keep="last").set_index("_address_key", drop=False)
    previous_urls = set(previous["_url_key"].dropna().astype(str))
    previous_urls.discard("")
    previous_mls = set(previous["_mls_key"].dropna().astype(str))
    previous_mls.discard("")

    for _, row in current.iterrows():
        address_key_value = row.get("_address_key", "")
        url_key_value = str(row.get("_url_key", "") or "")
        mls_key_value = str(row.get("_mls_key", "") or "")
        previous_row = previous_by_address.loc[address_key_value] if address_key_value in previous_by_address.index else None
        same_listing_identity = (url_key_value and url_key_value in previous_urls) or (mls_key_value and mls_key_value in previous_mls)

        change_type = "Existing"
        previous_price = pd.NA
        previous_mls = ""
        previous_url = ""
        price_delta = pd.NA
        if previous_row is None:
            change_type = "New Address"
        else:
            previous_price = previous_row.get("_price_numeric", pd.NA)
            previous_mls = previous_row.get("MLS", "")
            previous_url = previous_row.get("Listing URL", previous_row.get("Website", ""))
            current_price = row.get("_price_numeric", pd.NA)
            price_changed = pd.notna(current_price) and pd.notna(previous_price) and float(current_price) != float(previous_price)
            identity_changed = not same_listing_identity
            if identity_changed and price_changed:
                change_type = "Relisted / Price Changed"
            elif identity_changed:
                change_type = "Relisted / Changed"
            elif price_changed:
                change_type = "Price Changed"
            if price_changed:
                price_delta = float(current_price) - float(previous_price)

        if change_type == "Existing":
            continue

        rows.append({
            "Address": row.get("Address", ""),
            "City": row.get("City", ""),
            "Current Price": row.get("Price", ""),
            "Previous Price": previous_price,
            "Price Delta": price_delta,
            "Current MLS": row.get("MLS", ""),
            "Previous MLS": previous_mls,
            "Current Listing URL": row.get("Listing URL", row.get("Website", "")),
            "Previous Listing URL": previous_url,
            "Change Type": change_type,
            "Current Workbook": current_file.name,
            "Previous Workbook": previous_file.name,
        })

    change_log = pd.DataFrame(rows)
    change_log.to_csv(output_path, index=False)
    status(f"Saved listing change log: {output_path.name} ({len(change_log)} changed listing(s)).")
    return output_path
def load_manual_listings(root: Path, status: Callable[[str], None] = log_default) -> pd.DataFrame:
    """Load manually confirmed listings that the automated Realtor.ca pull missed."""
    manual_file = root / MANUAL_LISTINGS_FILE
    if not manual_file.exists():
        return pd.DataFrame()
    manual = pd.read_csv(manual_file)
    if manual.empty:
        return manual
    if "manual_listing_source" not in manual.columns:
        manual["manual_listing_source"] = "Manual Realtor.ca URL"
    manual["Fetched On"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    status(f"Loaded {len(manual)} manually confirmed listing(s).")
    return manual


def append_manual_listings(listings_file: Path, root: Path, status: Callable[[str], None] = log_default) -> Path:
    """Append manual listings to the refreshed workbook before enrichment."""
    manual = load_manual_listings(root, status)
    if manual.empty:
        return listings_file

    listings = pd.read_excel(listings_file)
    combined = pd.concat([listings, manual], ignore_index=True, sort=False)
    if "Listing URL" in combined.columns:
        combined = combined.drop_duplicates(subset=["Listing URL"], keep="last")
    if "Address" in combined.columns:
        combined = combined.drop_duplicates(subset=["Address"], keep="last")
    combined.to_excel(listings_file, index=False)
    status(f"Appended manual listings to {listings_file.name}; total rows now {len(combined)}.")
    return listings_file


def find_latest_listing_file(root: Path) -> Path:
    """Find the newest previously generated Realtor.ca listing workbook."""
    files = sorted(root.glob("North_West_Vancouver_Houses_Open_Houses_WORKING_URLS_*.xlsx"))
    if not files:
        raise FileNotFoundError("No Realtor output Excel file found.")
    return files[-1]


def ensure_client_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the standard report columns exist, using loose source aliases."""
    aliases = {
        "MLS": ["MLS", "MLS Number", "MlsNumber", "Mls Number", "MLS® Number", "Reference Number"],
        "Listing URL": ["Listing URL", "URL", "Link"],
        "Address": ["Address"],
        "Postal Code": ["Postal Code", "PostalCode"],
        "City": ["City"],
        "Latitude": ["Latitude", "Lat"],
        "Longitude": ["Longitude", "Lon", "Lng"],
        "Price": ["Price", "_price_numeric"],
        "Open House": ["Open House", "OpenHouse", "Open"],
        "Description": ["Description", "Public Remarks", "Remarks"],
        "Bedrooms": ["Bedrooms", "Bedroom"],
        "Bathrooms": ["Bathrooms", "Bathroom"],
        "Size": ["Size", "Living Area", "Floor Area", "Area"],
        "Stories": ["Stories", "Storeys"],
        "House Category": ["House Category", "Property Type"],
        "Ammenities": ["Ammenities", "Amenities"],
        "Ownership Category": ["Ownership Category", "Ownership"],
        "Nearby Ammenities": ["Nearby Ammenities", "Nearby Amenities", "Nearby"],
        "Website": ["Website"],
    }
    out = df.copy()
    for desired, terms in aliases.items():
        if desired not in out.columns:
            src = find_col(out, terms)
            out[desired] = out[src] if src is not None else np.nan
    for column in ["geometry_wkt", "final_school", "final_fraser_score", "final_fraser_rank", "school_source"]:
        if column not in out.columns:
            out[column] = np.nan
    return out


def load_catchments(root: Path):
    """Load North and West Vancouver catchment polygons into one projected layer."""
    import geopandas as gpd

    # North Vancouver uses official catchment files; West Vancouver is loaded
    # from the local approximate catchment package.
    nv_dir = extract_zip_if_needed(root / "RegSchoolCatchmentAreas_shp.zip", root / "catchments")
    nv_raw = read_first_vector_file(nv_dir)
    if SCHOOL_COL_NV not in nv_raw.columns:
        raise ValueError(f"Cannot find {SCHOOL_COL_NV} in North Vancouver catchment file.")
    nv = nv_raw[[SCHOOL_COL_NV, "geometry"]].rename(columns={SCHOOL_COL_NV: "catchment_school"})
    nv = gpd.GeoDataFrame(nv, geometry="geometry", crs=nv_raw.crs).to_crs(PROJECTED_CRS)
    nv["catchment_district"] = "North Vancouver"
    nv["school_source"] = "North Vancouver official catchment polygon"

    wv_dir = extract_zip_if_needed(root / "west_van_approx_catchments.zip", root / "west_van_approx_catchments")
    wv_raw = read_first_vector_file(wv_dir)
    wv_school_col = "school" if "school" in wv_raw.columns else find_col(wv_raw, ["school", "name"])
    if wv_school_col is None:
        raise ValueError("Cannot find school/name column in West Vancouver catchment file.")
    wv = wv_raw[[wv_school_col, "geometry"]].rename(columns={wv_school_col: "catchment_school"})
    wv = gpd.GeoDataFrame(wv, geometry="geometry", crs=wv_raw.crs).to_crs(PROJECTED_CRS)
    wv["catchment_district"] = "West Vancouver"
    wv["school_source"] = "West Vancouver approximate catchment polygon"

    all_catchments = pd.concat([nv, wv], ignore_index=True)
    all_catchments["catchment_school_key"] = all_catchments["catchment_school"].apply(clean_school_name)
    # Match local catchment naming to the Fraser score workbook naming.
    school_name_corrections = {
        "HIGHLAND": "HIGHLANDS",
        "QUEENMARY": "QUEEN MARY",
    }
    all_catchments["catchment_school_key"] = all_catchments["catchment_school_key"].replace(school_name_corrections)
    return gpd.GeoDataFrame(all_catchments, geometry="geometry", crs=PROJECTED_CRS)


def load_school_scores(root: Path) -> pd.DataFrame:
    """Load elementary Fraser scores for the two target school districts."""
    score_file = root / "Public_Elementary_Schools_Master_2026-05-01.xlsx"
    if not score_file.exists():
        raise FileNotFoundError(f"Cannot find {score_file.name}.")

    scores = pd.read_excel(score_file).rename(
        columns={
            "School": "school",
            "Fraser Score": "fraser_score",
            "Fraser Provincial Rank": "fraser_rank",
            "District": "district",
            "Level_Type": "level_type",
        }
    )
    scores["school_key"] = scores["school"].apply(clean_school_name)
    scores["district_clean"] = (
        scores["district"]
        .astype(str)
        .str.upper()
        .str.replace(r"\bSCHOOL\b|\bDISTRICT\b|#\d+|[^A-Z ]+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    scores = scores[scores["district_clean"].isin(["NORTH VANCOUVER", "WEST VANCOUVER"])].copy()
    if "level_type" in scores.columns:
        scores = scores[scores["level_type"].astype(str).str.lower().eq("elementary")].copy()
    return scores[["school_key", "district_clean", "school", "fraser_score", "fraser_rank"]].drop_duplicates()


def enrich_schools(root: Path, listings_file: Path, status: Callable[[str], None] = log_default) -> pd.DataFrame:
    """Attach catchment school names and Fraser scores to each geocoded listing."""
    import geopandas as gpd

    status("Loading refreshed listings and school catchments...")
    listings = ensure_client_columns(pd.read_excel(listings_file))
    listings["Latitude"] = pd.to_numeric(listings["Latitude"], errors="coerce")
    listings["Longitude"] = pd.to_numeric(listings["Longitude"], errors="coerce")
    listings = listings.dropna(subset=["Latitude", "Longitude"]).copy()
    if listings.empty:
        raise RuntimeError("No refreshed listings have latitude/longitude for catchment enrichment.")

    catchments = load_catchments(root)
    scores = load_school_scores(root)
    gdf = gpd.GeoDataFrame(
        listings,
        geometry=gpd.points_from_xy(listings["Longitude"], listings["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)

    # Spatially join each listing point to the catchment polygon that contains it.
    joined = gpd.sjoin(gdf, catchments, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    if "school_source_right" in joined.columns:
        joined["school_source"] = joined["school_source_right"]
        joined = joined.drop(columns=["school_source_left", "school_source_right"], errors="ignore")
    joined["catchment_district_clean"] = joined["catchment_district"].astype(str).str.upper().str.strip()
    joined = joined.merge(
        scores,
        left_on=["catchment_school_key", "catchment_district_clean"],
        right_on=["school_key", "district_clean"],
        how="left",
    ).drop(columns=["school_key", "district_clean", "catchment_district_clean"], errors="ignore")
    joined["final_school"] = joined["school"].fillna(joined["catchment_school"])
    joined["final_fraser_score"] = joined["fraser_score"]
    joined["final_fraser_rank"] = joined["fraser_rank"]

    final_geo = joined.to_crs("EPSG:4326").copy()
    final_geo["geometry_wkt"] = final_geo.geometry.to_wkt()
    final = pd.DataFrame(final_geo.drop(columns=["geometry"]))
    final = ensure_client_columns(final)
    status(f"School enrichment complete for {len(final)} listings.")
    return final


def add_numeric_fields(final: pd.DataFrame) -> pd.DataFrame:
    """Add numeric fields needed by ranking, filtering, and profile scoring."""
    out = final.copy()
    for column in ["Price", "Bedrooms", "Bathrooms", "Latitude", "Longitude", "final_fraser_score"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["size_sqft_numeric"] = out["Size"].apply(parse_number)
    out["price_per_sqft"] = out["Price"] / out["size_sqft_numeric"]
    out["City"] = np.where(
        out["Address"].astype(str).str.contains("West Vancouver", case=False, na=False),
        "West Vancouver",
        np.where(out["Address"].astype(str).str.contains("North Vancouver", case=False, na=False), "North Vancouver", out["City"]),
    )
    return out


def add_noise_features(final: pd.DataFrame, status: Callable[[str], None] = log_default) -> pd.DataFrame:
    """Estimate listing noise exposure from distance to major local corridors."""
    import geopandas as gpd
    from shapely.geometry import LineString

    roads = gpd.GeoDataFrame(
        [{"road_name": item["name"], "road_type": item["type"], "geometry": LineString(item["coords"])} for item in NOISE_CORRIDORS],
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    props = gpd.GeoDataFrame(
        final.copy(),
        geometry=gpd.points_from_xy(final["Longitude"], final["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)

    def nearest_corridor(point):
        """Return distance and metadata for the nearest configured corridor."""
        distances = roads.geometry.distance(point)
        idx = distances.idxmin()
        return pd.Series([float(distances.loc[idx]), roads.loc[idx, "road_name"], roads.loc[idx, "road_type"]])

    def nearest_distance_by_type(point, road_type: str) -> float:
        subset = roads[roads["road_type"].eq(road_type)]
        if subset.empty:
            return np.nan
        return float(subset.geometry.distance(point).min())

    props[["distance_to_noise_corridor_m", "nearest_noise_corridor", "nearest_noise_corridor_type"]] = props.geometry.apply(nearest_corridor)
    props["distance_to_highway_m"] = props.geometry.apply(lambda point: nearest_distance_by_type(point, "highway"))
    props["distance_to_major_road_m"] = props.geometry.apply(lambda point: nearest_distance_by_type(point, "major_road"))
    props["noise_risk"] = np.select(
        [
            props["distance_to_noise_corridor_m"].isna(),
            (props["distance_to_noise_corridor_m"] <= 250) | (props["distance_to_highway_m"] <= 450),
            (props["distance_to_noise_corridor_m"] <= 600) | (props["distance_to_highway_m"] <= 850),
        ],
        ["Unknown", "High", "Medium"],
        default="Low",
    )
    highway_is_driver = props["distance_to_highway_m"].le(850) & props["noise_risk"].isin(["High", "Medium"])
    props.loc[highway_is_driver, "nearest_noise_corridor"] = "Highway 1 / Upper Levels"
    props.loc[highway_is_driver, "nearest_noise_corridor_type"] = "highway"
    props["noise_model_risk"] = props["noise_risk"]
    props["noise_override_note"] = ""
    props["noise_context_note"] = ""
    props["noise_verification_needed"] = props["noise_risk"].isin(["High", "Medium"])
    props = apply_contextual_noise_adjustment(props)
    for override in NOISE_OVERRIDES:
        mask = props["Address"].astype(str).str.upper().str.contains(override["address_contains"].upper(), regex=False, na=False)
        props.loc[mask, "noise_risk"] = override["noise_risk"]
        props.loc[mask, "noise_override_note"] = override["note"]
        props.loc[mask, "noise_verification_needed"] = True
    status("Noise-risk enrichment complete.")
    return pd.DataFrame(props.drop(columns=["geometry"]))


def apply_contextual_noise_adjustment(props: pd.DataFrame) -> pd.DataFrame:
    out = props.copy()
    text_cols = [column for column in ["Address", "Description", "Nearby Ammenities"] if column in out.columns]
    combined = out[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower() if text_cols else pd.Series("", index=out.index)
    has_buffer = combined.apply(lambda text: any(term in text for term in NOISE_BUFFER_TERMS))
    has_negative = combined.apply(lambda text: any(term in text for term in NOISE_NEGATIVE_TERMS))
    is_highway = out["nearest_noise_corridor"].astype(str).str.contains("Highway", case=False, na=False)
    high_to_medium = out["noise_risk"].eq("High") & is_highway & has_buffer & ~has_negative
    medium_to_low = out["noise_risk"].eq("Medium") & has_buffer & ~has_negative
    highway_text_warning = has_negative & combined.str.contains(r"\b(?:hwy|highway|upper levels)\b", regex=True, na=False)
    highway_text_medium_to_high = highway_text_warning & out["noise_risk"].eq("Medium") & ~has_buffer

    out.loc[high_to_medium, "noise_risk"] = "Medium"
    out.loc[high_to_medium, "noise_context_note"] = (
        "Distance model flagged High, adjusted to Medium because listing/neighbourhood text suggests residential/terrain/amenity buffering; verify during showing."
    )
    out.loc[medium_to_low, "noise_risk"] = "Low"
    out.loc[medium_to_low, "noise_context_note"] = (
        "Distance model flagged Medium, adjusted to Low because listing/neighbourhood text suggests quiet residential buffering; verify during showing."
    )
    out.loc[highway_text_medium_to_high, "noise_risk"] = "High"
    out.loc[highway_text_medium_to_high, "noise_context_note"] = (
        "Listing text references Highway 1/Upper Levels access and the distance model is already near a major corridor; verify road noise from home and yard."
    )
    out.loc[high_to_medium | medium_to_low | highway_text_warning, "noise_verification_needed"] = True
    return out


def add_assessment_placeholders(final: pd.DataFrame) -> pd.DataFrame:
    """Add BC Assessment columns and search links for manual value lookup."""
    out = final.copy()
    out["bc_assessment_total_value"] = np.nan
    out["bc_assessment_land_value"] = np.nan
    out["bc_assessment_building_value"] = np.nan
    out["bc_assessment_year"] = np.nan
    out["bc_assessment_status"] = "Not fetched - verify manually"
    out["bc_assessment_source"] = "BC Assessment public website, manual/optional lookup required"
    out["BC Assessment Search Link"] = "https://www.bcassessment.ca/Property/AssessmentSearch?sp=1"
    return out


def add_basic_ranking(final: pd.DataFrame) -> pd.DataFrame:
    """Create a default family-fit score before buyer-profile scoring is applied."""
    out = final.copy()
    out["school_confidence"] = out["school_source"].map(
        lambda value: (
            "High"
            if "official" in str(value).lower() or "baragar" in str(value).lower()
            else "Medium"
            if "approx" in str(value).lower()
            else "Low"
        )
    )
    school = (pd.to_numeric(out["final_fraser_score"], errors="coerce") * 10).clip(0, 100).fillna(50)
    price = (100 * (2_500_000 - pd.to_numeric(out["Price"], errors="coerce")) / 2_500_000).clip(0, 100).fillna(50)
    noise = out["noise_risk"].map({"Low": 100, "Medium": 70, "High": 35, "Unknown": 50}).fillna(50)
    size = (pd.to_numeric(out["Bedrooms"], errors="coerce").fillna(0) / 3).clip(0, 1) * 100
    # Weighted default score: schools matter most, then quietness, price, and bedroom fit.
    out["AI Family Fit Score"] = (school * 0.35 + price * 0.2 + noise * 0.25 + size * 0.2).round(1)
    out = out.sort_values("AI Family Fit Score", ascending=False).reset_index(drop=True)
    out["AI Rank"] = np.arange(1, len(out) + 1)
    out["Map Link"] = "https://www.google.com/maps/search/?api=1&query=" + out["Latitude"].astype(str) + "," + out["Longitude"].astype(str)
    out["AI Explanation"] = out.apply(
        lambda row: (
            f"School: {row.get('final_school', 'Unknown')} ({row.get('final_fraser_score', 'N/A')}/10). "
            f"Noise risk: {row.get('noise_risk', 'Unknown')}. "
            f"BC Assessment: {row.get('bc_assessment_status', 'Not fetched')}."
        ),
        axis=1,
    )
    return out


def write_client_report(root: Path, final: pd.DataFrame, status: Callable[[str], None] = log_default) -> Path:
    """Write the refreshed workbook used by clients and the Streamlit app."""
    out_file = root / "family_home_advisor_client_report.xlsx"
    ranking_cols = [
        "AI Rank", "AI Family Fit Score", "Address", "City", "Price", "Bedrooms", "Bathrooms", "Size",
        "final_school", "final_fraser_score", "school_confidence", "school_source",
        "noise_risk", "noise_model_risk", "noise_override_note", "noise_verification_needed",
        "noise_context_note",
        "distance_to_noise_corridor_m", "distance_to_highway_m", "distance_to_major_road_m", "nearest_noise_corridor",
        "Open House", "AI Explanation", "Map Link", "Listing URL",
        "bc_assessment_total_value", "bc_assessment_land_value", "bc_assessment_building_value",
        "bc_assessment_year", "bc_assessment_status", "BC Assessment Search Link",
    ]
    all_cols = [column for column in final.columns if column not in ["school", "fraser_score", "fraser_rank"]]
    ranking = final[[column for column in ranking_cols if column in final.columns]].copy()
    all_listings = final[all_cols].copy()
    # Summarize each school catchment so clients can compare price and noise patterns.
    school_summary = (
        final.groupby(["final_school", "final_fraser_score", "school_confidence"], dropna=False)
        .agg(
            listings_count=("Address", "count"),
            min_price=("Price", "min"),
            median_price=("Price", "median"),
            max_score=("AI Family Fit Score", "max"),
            low_noise_count=("noise_risk", lambda s: (s == "Low").sum()),
            high_noise_count=("noise_risk", lambda s: (s == "High").sum()),
        )
        .reset_index()
        .sort_values(["final_fraser_score", "max_score"], ascending=False)
    )
    client_preferences = pd.DataFrame(
        {
            "Setting": ["refresh_date", "source", "bc_assessment_note"],
            "Value": [datetime.now().strftime("%Y-%m-%d %H:%M"), "pyRealtor + local catchment files", "Assessment values are placeholders unless manually/optionally fetched."],
        }
    )
    scoring_method = pd.DataFrame(
        {
            "Component": ["School", "Price", "Noise", "Size", "BC Assessment"],
            "Description": [
                "Fraser score and catchment assignment from local school files.",
                "Basic default ranking favors lower price under $2.5M; Streamlit app applies buyer-specific scoring.",
                "Approximate distance to major corridors.",
                "Bedroom fit in the generated workbook; Streamlit app also uses size.",
                "Search links are provided; values are not bulk-fetched by default.",
            ],
        }
    )

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="Personalized_Ranking", index=False)
        all_listings.to_excel(writer, sheet_name="All_Listings", index=False)
        client_preferences.to_excel(writer, sheet_name="Client_Preferences", index=False)
        school_summary.to_excel(writer, sheet_name="School_Summary", index=False)
        scoring_method.to_excel(writer, sheet_name="Scoring_Method", index=False)
    write_profile_ranking_sheets(out_file, final, status)
    status(f"Saved refreshed client report: {out_file.name}")
    return out_file


def write_profile_ranking_sheets(out_file: Path, final: pd.DataFrame, status: Callable[[str], None] = log_default) -> None:
    """Append one ranking sheet per preset buyer profile."""
    from buyer_profile import PRESET_PROFILES
    from data_cleaning import add_noise_columns, add_open_house_columns, normalize_columns
    from scoring import score_listings

    profile_columns = [
        "match_score", "buyer_fit_flags", "lifestyle_component", "rancher_component",
        "backyard_component", "mortgage_helper_component", "layout_component",
        "location_component", "location_flags",
        "condition_component", "condition_flags",
        "price_component", "budget_price_component", "sqft_value_component",
        "assessment_value_component", "quiet_component", "size_component",
        "school_component", "Address", "City", "Price", "Bedrooms", "Bathrooms", "Size",
        "price_per_sqft", "final_school", "final_fraser_score", "school_confidence",
        "noise_risk", "noise_model_risk", "noise_override_note", "noise_verification_needed",
        "noise_context_note",
        "distance_to_noise_corridor_m", "distance_to_highway_m", "distance_to_major_road_m", "nearest_noise_corridor",
        "open_house_status", "Open House", "explanation", "Listing URL", "Google Maps Link",
        "bc_assessment_total_value", "bc_assessment_land_value", "bc_assessment_building_value",
        "bc_assessment_status", "BC Assessment Search Link",
    ]

    data, _ = normalize_columns(final)
    data = add_open_house_columns(data)
    data = add_noise_columns(data)
    # Let profile scorers use the highest refreshed price as the session budget ceiling.
    max_price = int(pd.to_numeric(data["price_numeric"], errors="coerce").max(skipna=True) or 2_500_000)

    rankings: dict[str, pd.DataFrame] = {}
    for profile_name, profile in PRESET_PROFILES.items():
        prefs = profile.defaults()
        prefs["max_price"] = max_price
        scored = score_listings(data, prefs)
        selected = [column for column in profile_columns if column in scored.columns]
        profile_df = scored[selected].copy()
        profile_df.insert(0, "profile", profile_name)
        rankings[profile_name] = profile_df

    with pd.ExcelWriter(out_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        main_profile = "Rancher Backyard Profile" if "Rancher Backyard Profile" in rankings else next(iter(rankings))
        rankings[main_profile].to_excel(writer, sheet_name="Personalized_Ranking", index=False)
        for profile_name, profile_df in rankings.items():
            sheet_name = {
                "Quiet Family Profile": "Profile_Quiet_Family",
                "Top School Profile": "Profile_Top_School",
                "Value Buyer Profile": "Profile_Value_Buyer",
                "Rancher Backyard Profile": "Profile_Rancher_Backyard",
            }.get(profile_name, profile_name[:31])
            profile_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    status("Added profile-specific ranking sheets.")


def refresh_family_home_advisor(
    root: Path | str = ".",
    max_price: int = 2_500_000,
    open_houses_only: bool = False,
    use_existing_listing_file: bool = False,
    status: Callable[[str], None] = log_default,
) -> Path:
    """Run the full refresh pipeline and return the final report path."""
    root = Path(root).resolve()
    if use_existing_listing_file:
        listings_file = find_latest_listing_file(root)
        status(f"Using existing listing file: {listings_file.name}")
    else:
        listings_file = fetch_realtor_listings(root, TARGET_CITIES, max_price, open_houses_only, status)
    listings_file = append_manual_listings(listings_file, root, status)
    write_listing_change_log(root, listings_file, status)

    # Enrichment order matters: later ranking fields depend on schools, numeric
    # values, noise categories, and assessment placeholders all being present.
    final = enrich_schools(root, listings_file, status)
    final = add_numeric_fields(final)
    final = add_noise_features(final, status)
    final = add_assessment_placeholders(final)
    final = add_basic_ranking(final)
    return write_client_report(root, final, status)


def main() -> None:
    """Parse CLI arguments and launch the refresh pipeline."""
    parser = argparse.ArgumentParser(description="Refresh Family Home Advisor listing workbook.")
    parser.add_argument("--root", default=".", help="Housing_app folder path.")
    parser.add_argument("--max-price", type=int, default=2_500_000)
    parser.add_argument("--open-houses-only", action="store_true")
    parser.add_argument("--use-existing-listing-file", action="store_true")
    args = parser.parse_args()
    refresh_family_home_advisor(
        root=args.root,
        max_price=args.max_price,
        open_houses_only=args.open_houses_only,
        use_existing_listing_file=args.use_existing_listing_file,
    )


if __name__ == "__main__":
    main()
