from __future__ import annotations

from html import escape
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from area_filters import AREA_KEYWORDS, add_area_columns
from data_cleaning import add_noise_columns, add_open_house_columns, choose_listing_sheet, find_default_input, money, normalize_columns
from photo_review import load_photo_reviews, merge_photo_reviews, openai_ready
from review_store import load_reviews, merge_reviews
from scoring import score_listings
from v2_product import append_listing_event, card_title, concerns, evidence_table, listing_links, load_family_profile, load_listing_events, recommendation_sentence, verification_steps, why_it_may_work

try:
    import folium
except ImportError:
    folium = None
try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium = None

APP_TITLE = "Family Home Advisor"
ROOT = Path(__file__).resolve().parent
REVIEWS_PATH = ROOT / "manual_reviews.csv"
PHOTO_REVIEWS_PATH = ROOT / "photo_reviews.csv"
FAMILY_PROFILE_PATH = ROOT / "family_profile.json"
LISTING_EVENTS_PATH = ROOT / "listing_events.csv"
BC_ASSESSMENT_SEARCH_URL = "https://www.bcassessment.ca/Property/AssessmentSearch?sp=1"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")


def configure_openai_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    if not (ROOT / ".streamlit" / "secrets.toml").exists() and not (Path.home() / ".streamlit" / "secrets.toml").exists():
        return
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = str(key)


configure_openai_key()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {background:#ffffff; color:#0f172a;}
        .block-container {padding-top:.85rem; max-width:1540px;}
        header[data-testid="stHeader"] {background:rgba(255,255,255,.92);}
        section[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #edf1ee; box-shadow:8px 0 24px rgba(15,45,30,.03);}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {font-size:17px; margin-bottom:.25rem;}
        section[data-testid="stSidebar"] label {font-weight:750; color:#111827;}
        div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px!important; border-color:#e7ede9!important; box-shadow:0 12px 30px rgba(18,52,34,.07);}
        .brand {display:flex; gap:12px; align-items:center; min-width:245px;}
        .logo {width:43px;height:43px;border-radius:14px;background:#eaf8ef;color:#07915a;display:grid;place-items:center;font-weight:900;font-size:23px;border:1px solid #cfeadb;}
        .title {font-size:24px;font-weight:850;color:#0f172a;line-height:1;}
        .sub {font-size:13px;color:#64748b;margin-top:4px;}
        .avatar {width:42px;height:42px;border-radius:50%;background:#07915a;color:white;display:grid;place-items:center;font-weight:850;margin-top:2px;}
        .panel-title {font-size:20px;font-weight:850;color:#0f172a;margin:2px 0;}
        .panel-sub {font-size:13px;color:#64748b;margin-bottom:12px;}
        .toolbar {display:flex;align-items:center;justify-content:space-between;margin:0 0 10px;gap:10px;}
        .toolbar-left {display:flex;align-items:center;gap:10px;}
        .seg {display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid #e5ece7;border-radius:13px;box-shadow:0 7px 18px rgba(18,52,34,.08);padding:9px 13px;font-weight:800;color:#0f172a;}
        .seg-muted {color:#475569;font-weight:750;}
        .home-count {display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid #e5ece7;border-radius:13px;box-shadow:0 7px 18px rgba(18,52,34,.08);padding:9px 14px;font-weight:850;}
        .home-count-dot {width:8px;height:8px;border-radius:50%;background:#07915a;display:inline-block;}
        .right-head {display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
        .view-all {font-size:13px;color:#07915a;font-weight:850;}
        .badge {display:inline-flex;border-radius:999px;font-size:12px;font-weight:850;padding:5px 10px;margin-bottom:8px;}
        .green {background:#e7f6ec;color:#17633d}.yellow {background:#fff7dc;color:#765800}.red {background:#fff0ee;color:#9f351f}
        .score {display:inline-grid;place-items:center;min-width:48px;height:32px;border-radius:10px;color:white;font-weight:900;font-size:15px;}
        .score.green-bg {background:#07915a}.score.yellow-bg {background:#f3b700}.score.red-bg {background:#ff6b5f}
        .photo {height:124px;border-radius:14px;background:linear-gradient(135deg,#dfeee5,#c5dfcf,#a7ccb6);display:flex;align-items:flex-end;padding:12px;color:#17462f;font-weight:850;border:1px solid #d7e7dc;margin-bottom:10px;}
        .rec-photo .photo {height:168px; margin-bottom:0;}
        .reason {font-size:13px;color:#27362e;margin:3px 0}.reason:before{content:'✓ ';color:#07915a;font-weight:900}.concern {font-size:13px;color:#86451d;margin-top:7px;}.concern:before{content:'⚠ ';color:#e2a100;font-weight:900}
        .sidebar-hint {font-size:12px;color:#64748b;margin-top:-4px;margin-bottom:8px;}
        .chat-box {border:1px solid #cfeadb;border-radius:18px;background:#ffffff;box-shadow:0 12px 30px rgba(18,52,34,.08);padding:14px;margin-top:12px;}
        .chat-title {font-weight:850;color:#0f172a;margin-bottom:8px;}
        .chat-bubble {background:#f8fafc;border:1px solid #edf2ef;border-radius:14px;padding:11px 12px;color:#334155;font-size:13px;margin-bottom:10px;}
        .prompt-row {display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;}
        .prompt-pill {border:1px solid #e5ece7;border-radius:999px;padding:6px 9px;font-size:11px;color:#334155;background:#fff;}
        div.stButton>button{border-radius:12px;font-weight:800;border:1px solid #dfe8e2;background:white;}
        div.stButton>button:hover{border-color:#07915a;color:#07915a;}
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {background:#e7f6ec;color:#17633d;border-radius:999px;}
        @media(max-width:1100px){.toolbar{align-items:flex-start;flex-direction:column}.brand{min-width:0}}
        </style>
        """,
        unsafe_allow_html=True,
    )
def read_workbook(path_text: str, modified_time: float) -> tuple[pd.DataFrame, str, list[str]]:
    path = Path(path_text)
    sheet = choose_listing_sheet(path)
    return pd.read_excel(path, sheet_name=sheet), sheet, []


def source_path() -> Path:
    packaged = ROOT / "family_home_advisor_client_report.xlsx"
    if packaged.exists():
        return packaged
    found = find_default_input(ROOT)
    if found is None:
        raise FileNotFoundError("No listing workbook found.")
    return found


def normalize_url(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.replace("https://www.realtor.ca", "", regex=False).str.split("?").str[0].str.rstrip("/")


def addr_key(df: pd.DataFrame) -> pd.Series:
    return df.get("Address", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def add_change_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_new_since_last_refresh"] = False
    out["listing_change_status"] = "Existing"
    path = ROOT / "listing_change_log.csv"
    if not path.exists():
        return out
    try:
        changes = pd.read_csv(path)
    except Exception:
        return out
    mask = pd.Series(False, index=out.index)
    if "Listing URL" in out.columns:
        refs = set()
        for col in ["Current Listing URL", "Listing URL", "Website"]:
            if col in changes.columns:
                refs.update(normalize_url(changes[col]))
        refs = {x for x in refs if x and x not in {"nan", "none"}}
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


def load_data() -> tuple[pd.DataFrame, Path, list[str]]:
    path = source_path()
    raw, _, _ = read_workbook(str(path), path.stat().st_mtime)
    df, warnings = normalize_columns(raw)
    df = add_open_house_columns(df)
    df = add_noise_columns(df)
    df = merge_reviews(df, load_reviews(REVIEWS_PATH))
    df = merge_photo_reviews(df, load_photo_reviews(PHOTO_REVIEWS_PATH))
    df = add_change_flags(df)
    df["BC Assessment Search Link"] = BC_ASSESSMENT_SEARCH_URL
    df = add_area_columns(df)
    return df, path, warnings


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
def home_ai_answer(prompt: str, row: pd.Series, visible: pd.DataFrame) -> str:
    text = prompt.lower()
    address = title(row)
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
    return ai_answer(f"About {row.get('Address')}: {prompt}", visible)
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


def badge(row: pd.Series) -> None:
    st.markdown(f"<span class='badge {tone(row)}'>{escape(category(row))}</span>", unsafe_allow_html=True)


def title(row: pd.Series) -> str:
    return card_title(row)


def slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).upper()).strip("_")[:80]


def photo_path(row: pd.Series) -> Path | None:
    folder = ROOT / "listing_photos"
    if not folder.exists():
        return None
    matches = sorted(folder.glob(f"{slug(row.get('Address', ''))[:24]}*"))
    for match in matches:
        photos = sorted(match.glob("*.jpg")) + sorted(match.glob("*.png"))
        if photos:
            return photos[0]
    return None


def photo(row: pd.Series) -> None:
    path = photo_path(row)
    if path and path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.markdown(f"<div class='photo'>{escape(title(row))}</div>", unsafe_allow_html=True)


def marker_html(row: pd.Series) -> str:
    colors = {"green": ("#2f8f5b", "#17633d"), "yellow": ("#f2c94c", "#8a6a00"), "red": ("#dc6b57", "#9f351f")}
    bg, border = colors[tone(row)]
    return f"""
    <div style="width:42px;height:42px;border-radius:50%;background:{bg};border:3px solid {border};box-shadow:0 5px 14px rgba(0,0,0,.22);display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:900;">{score(row)}</div>
    """


def app_map(df: pd.DataFrame):
    if folium is None:
        return None
    mapped = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if mapped.empty:
        return None
    fmap = folium.Map(location=[mapped["Latitude"].mean(), mapped["Longitude"].mean()], zoom_start=12, tiles="CartoDB positron", control_scale=True)
    for _, row in mapped.iterrows():
        reasons = why_it_may_work(row, {}, limit=1)
        risk = family_concern_items(row, {}, limit=1)[0]
        url = str(row.get("Listing URL", ""))
        html = f"""
        <div style='font-family:Arial,sans-serif;width:260px;padding:4px;'>
          <div style='font-weight:800;font-size:15px;margin-bottom:4px;'>{escape(title(row))}</div>
          <div style='color:#17633d;font-weight:800;margin-bottom:6px;'>{money(row.get('price_numeric', row.get('Price')))} | Score {score(row)}/100</div>
          <div style='font-size:13px;margin-bottom:6px;'>{escape(category(row))}</div>
          <div style='font-size:12px;color:#34443b;'>+ {escape(reasons[0])}</div>
          <div style='font-size:12px;color:#86451d;margin-top:4px;'>Check: {escape(risk)}</div>
          <a href='{escape(url)}' target='_blank' style='display:block;margin-top:8px;color:#17633d;font-weight:700;'>Open Realtor.ca</a>
        </div>
        """
        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            tooltip=f"{title(row)} - {score(row)}/100",
            popup=folium.Popup(html, max_width=300),
            icon=folium.DivIcon(html=marker_html(row), icon_size=(42, 42), icon_anchor=(21, 21)),
        ).add_to(fmap)
    legend_html = """
    <div style="position: fixed; left: 18px; bottom: 26px; z-index: 9999; background: white; border: 1px solid #e5ece7; border-radius: 14px; padding: 14px 16px; box-shadow: 0 10px 26px rgba(18,52,34,.16); font-family: Arial, sans-serif; min-width: 205px;">
      <div style="font-weight: 800; margin-bottom: 10px; color: #0f172a;">AI Match Score</div>
      <div style="display:flex; align-items:center; gap:9px; margin:7px 0; font-size:12px;"><span style="width:11px;height:11px;border-radius:50%;background:#07915a;display:inline-block;"></span>80 - 100&nbsp;&nbsp; Strong Match</div>
      <div style="display:flex; align-items:center; gap:9px; margin:7px 0; font-size:12px;"><span style="width:11px;height:11px;border-radius:50%;background:#f3b700;display:inline-block;"></span>60 - 79&nbsp;&nbsp; Worth Visiting</div>
      <div style="display:flex; align-items:center; gap:9px; margin:7px 0; font-size:12px;"><span style="width:11px;height:11px;border-radius:50%;background:#ff6b5f;display:inline-block;"></span>0 - 59&nbsp;&nbsp; Needs Verification</div>
      <div style="font-size:12px; color:#0b72b9; font-weight:700; margin-top:10px;">About scores</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


def clicked_address(df: pd.DataFrame, click: dict[str, Any] | None) -> str | None:
    if not click or df.empty:
        return None
    lat, lon = num(click.get("lat"), float("nan")), num(click.get("lng"), float("nan"))
    if pd.isna(lat) or pd.isna(lon):
        return None
    mapped = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if mapped.empty:
        return None
    d = (mapped["Latitude"].astype(float) - lat).abs() + (mapped["Longitude"].astype(float) - lon).abs()
    return str(mapped.loc[d.idxmin(), "Address"])


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


def filter_data(scored: pd.DataFrame, search: str, price_range: tuple[int, int], locations: list[str], min_beds: int, min_school: float, yard: str, more: dict[str, bool]) -> pd.DataFrame:
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
def col(df: pd.DataFrame, name: str, default: object = "") -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


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
def reason_lines(row: pd.Series, profile: dict[str, Any], limit: int = 3) -> None:
    for reason in why_it_may_work(row, profile, limit=limit):
        st.markdown(f"<div class='reason'>+ {escape(reason)}</div>", unsafe_allow_html=True)


def recommendation_card(row: pd.Series, profile: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        left, right = st.columns([0.42, 0.58], gap="small")
        with left:
            st.markdown("<div class='rec-photo'>", unsafe_allow_html=True)
            photo(row)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>"
                f"<span class='score {tone(row)}-bg'>{score(row)}</span>"
                f"<span class='badge {tone(row)}'>{escape(category(row))}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{title(row)}**")
            st.caption(f"{money(row.get('price_numeric', row.get('Price')))}")
            st.caption(interior_size_summary(row))
            reason_lines(row, profile, 3)
            st.markdown(f"<div class='concern'>{escape(family_concern_items(row, profile, 1)[0])}</div>", unsafe_allow_html=True)
            if st.button("♡ Save", key=f"save_{key}", use_container_width=True):
                append_listing_event(LISTING_EVENTS_PATH, "Saved", str(row.get("Address", "")))
                st.toast("Saved home")
            if st.button("View Details", key=f"detail_{key}", use_container_width=True):
                st.session_state["selected_address"] = str(row.get("Address", ""))
                st.rerun()


def listing_card(row: pd.Series, profile: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        photo(row)
        top = st.columns([0.72, 0.28])
        with top[0]:
            st.markdown(f"**{title(row)}**")
            st.caption(f"{money(row.get('price_numeric', row.get('Price')))} | {row.get('Bedrooms', '')} bd | {row.get('Bathrooms', '')} ba | {row.get('Size', '')}")
            st.caption(interior_size_summary(row))
        with top[1]:
            st.markdown(f"<span class='score {tone(row)}-bg'>{score(row)}</span>", unsafe_allow_html=True)
        badge(row)
        a, b = st.columns(2)
        with a:
            if st.button("View", key=f"view_{key}"):
                st.session_state["selected_address"] = str(row.get("Address", ""))
                st.rerun()
        with b:
            if st.button("Save", key=f"card_save_{key}"):
                append_listing_event(LISTING_EVENTS_PATH, "Saved", str(row.get("Address", "")))
                st.toast("Saved home")


def selected_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    selected = st.session_state.get("selected_address")
    if selected:
        match = df[df["Address"].astype(str).eq(str(selected))]
        if not match.empty:
            return match.iloc[0]
    return df.iloc[0]


def detail_panel(row: pd.Series, visible: pd.DataFrame, profile: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>Property Detail</div>", unsafe_allow_html=True)
        photo(row)
        badge(row)
        st.markdown(f"### {title(row)}")
        st.caption(f"{row.get('City', '')} | {money(row.get('price_numeric', row.get('Price')))} | Score {score(row)}/100")
        st.info(interior_size_summary(row))
        st.write(recommendation_sentence(row, profile))
        st.markdown("**Score breakdown**")
        components = pd.DataFrame([
            {"Factor": "Location", "Score": row.get("location_component", 0)},
            {"Factor": "Yard", "Score": row.get("backyard_component", 0)},
            {"Factor": "Layout", "Score": row.get("layout_component", 0)},
            {"Factor": "Quiet", "Score": row.get("quiet_component", 0)},
            {"Factor": "Interior size", "Score": row.get("size_component", 0)},
            {"Factor": "School", "Score": row.get("school_component", 0)},
            {"Factor": "Value", "Score": row.get("price_component", 0)},
        ])
        st.dataframe(components, use_container_width=True, hide_index=True, height=285)
        st.markdown("**What we like**")
        for item in why_it_may_work(row, profile, 4):
            st.write(f"- {item}")
        st.markdown("**Concerns**")
        for item in family_concern_items(row, profile, 4):
            st.write(f"- {item}")
        st.markdown("**What to verify in person**")
        for item in verification_steps(row, profile, 5):
            st.write(f"- {item}")
        with st.expander("Evidence details", expanded=False):
            st.dataframe(evidence_table(row, profile), use_container_width=True, hide_index=True)
            links = listing_links(row)
            pieces = [f"[{name}]({url})" for name, url in links.items() if str(url).strip()]
            if pieces:
                st.markdown(" | ".join(pieces))
        similar = visible[visible["Address"].astype(str).ne(str(row.get("Address", "")))].copy()
        area = str(row.get("detected_area", ""))
        if area and "detected_area" in similar.columns:
            same = similar[similar["detected_area"].astype(str).eq(area)]
            if not same.empty:
                similar = same
        similar = similar.sort_values("match_score", ascending=False).head(3)
        if not similar.empty:
            st.markdown("**Similar homes**")
            for _, item in similar.iterrows():
                st.caption(f"{title(item)} | {money(item.get('price_numeric'))} | {score(item)}/100")
        question = st.text_input("Ask AI about this home", key=f"ask_{slug(row.get('Address', ''))}")
        if question:
            st.info(home_ai_answer(question, row, visible))


def parse_chat(prompt: str, current: dict[str, Any]) -> tuple[dict[str, Any], str]:
    text = prompt.lower()
    updates = dict(current)
    notes = []
    match = re.search(r"under\s*\$?\s*(\d+(?:\.\d+)?)\s*([mk])?", text)
    if match:
        number = float(match.group(1))
        updates["max_price"] = int(number * (1_000_000 if (match.group(2) or "m") == "m" else 1_000))
        notes.append(f"I will keep the search under {money(updates['max_price'])}.")
    if any(x in text for x in ["good school", "strong school", "top school"]):
        updates["min_school"] = max(float(updates.get("min_school", 0)), 8.0)
        notes.append("I will prioritize stronger schools.")
    if any(x in text for x in ["busy road", "highway", "noise", "quiet"]):
        updates["avoid_noise"] = True
        notes.append("I will treat road/highway noise as a major concern.")
    if any(x in text for x in ["toddler", "yard", "backyard", "play"]):
        updates["yard_focus"] = True
        notes.append("I will give more weight to usable outdoor space.")
    if any(x in text for x in ["steep", "driveway", "slope"]):
        updates["avoid_slope"] = True
        notes.append("I will flag slope or steep driveway risk for verification.")
    if any(x in text for x in ["size", "sqft", "square", "interior", "small"]):
        notes.append("For size questions, select a home and use Ask AI about this home so I can answer from that listing.")
    if "compare" in text:
        notes.append("I will compare the named homes when they are visible in the current results.")
    if not notes:
        notes.append("I saved that as a preference note. I can translate price, school, yard, noise, and comparison requests into search behavior.")
    return updates, " ".join(notes)


def ai_answer(prompt: str, visible: pd.DataFrame) -> str:
    rules, note = parse_chat(prompt, st.session_state.get("ai_rules", {}))
    st.session_state["ai_rules"] = rules
    if not openai_ready():
        return note
    try:
        from openai import OpenAI
        context_cols = ["Address", "City", "price_numeric", "Bedrooms", "Size", "match_score", "noise_risk", "final_school", "final_fraser_score"]
        context = visible.head(8)[[c for c in context_cols if c in visible.columns]].to_dict("records")
        client = OpenAI()
        requested_model = os.getenv("OPENAI_MODEL", "gpt-5.5")
        fallback_models = [requested_model]
        if requested_model != "gpt-4o-mini":
            fallback_models.append("gpt-4o-mini")
        for model_name in fallback_models:
            try:
                result = client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "system", "content": "You are a concise family home-buying assistant. Explain trade-offs and do not invent facts."},
                        {"role": "user", "content": f"Question: {prompt}\nVisible listings: {context}\nAlso state what filters or preferences you inferred."},
                    ],
                )
                return result.output_text or note
            except Exception:
                continue
        return note
    except Exception:
        return note
def top_header(saved_count: int) -> str:
    c1, c2, c3, c4, c5 = st.columns([0.25, 0.43, 0.12, 0.13, 0.07], gap="medium")
    with c1:
        st.markdown("""
        <div class='brand'><div class='logo'>⌂</div><div><div class='title'>Family Home Advisor</div><div class='sub'>AI-powered home search for families</div></div></div>
        """, unsafe_allow_html=True)
    with c2:
        search = st.text_input("Search by address, neighbourhood, or school", placeholder="Search by address, neighbourhood, or school", label_visibility="collapsed")
    with c3:
        if st.button("AI Chat ✨", use_container_width=True):
            st.session_state["show_chat"] = not st.session_state.get("show_chat", True)
    with c4:
        if st.button(f"♡ Saved Homes", use_container_width=True):
            st.session_state["saved_only"] = not st.session_state.get("saved_only", False)
    with c5:
        st.markdown("<div class='avatar'>NN</div>", unsafe_allow_html=True)
    st.divider()
    return search
def metrics(scored: pd.DataFrame, visible: pd.DataFrame) -> dict[str, int]:
    strong = int(visible.apply(lambda r: category(r) == "Strong Match", axis=1).sum()) if not visible.empty else 0
    open_houses = int(visible.get("open_house_status", pd.Series(index=visible.index)).eq("Upcoming").sum()) if not visible.empty else 0
    changed = int(visible.get("is_new_since_last_refresh", pd.Series(False, index=visible.index)).fillna(False).sum()) if not visible.empty else 0
    return {"all": len(scored), "visible": len(visible), "strong": strong, "open_houses": open_houses, "changed": changed}


def map_toolbar(counts: dict[str, int]) -> None:
    st.markdown(
        f"""
        <div class='toolbar'>
          <div class='toolbar-left'>
            <div class='seg'>▣ Map</div>
            <div class='seg seg-muted'>☷ List</div>
            <div class='home-count'><span class='home-count-dot'></span>{counts['visible']} homes</div>
          </div>
          <div class='seg seg-muted'>▨</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_panel(visible: pd.DataFrame) -> None:
    st.markdown("""
    <div class='chat-box'>
      <div class='chat-title'>✦ Ask AI Assistant</div>
      <div class='chat-bubble'>Hi, I'm your AI home advisor. Ask me about homes, neighbourhoods, schools, or what is best for your family.</div>
      <div class='prompt-row'>
        <span class='prompt-pill'>Homes under $2M with good schools</span>
        <span class='prompt-pill'>Best for a toddler</span>
        <span class='prompt-pill'>Compare Kilkeel and Ballantree</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    for message in st.session_state.get("chat_messages", [])[-3:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.text_input("Ask a question", placeholder="Ask a question...", label_visibility="collapsed", key="rail_ai_question")
    if st.button("Send to AI", key="rail_ai_send", use_container_width=True) and question.strip():
        st.session_state.setdefault("chat_messages", []).append({"role": "user", "content": question})
        answer = ai_answer(question, visible)
        st.session_state.setdefault("chat_messages", []).append({"role": "assistant", "content": answer})
        st.rerun()
def main() -> None:
    inject_css()
    try:
        df, path, warnings = load_data()
    except Exception as exc:
        st.error("Could not load listings.")
        st.exception(exc)
        return

    events = load_listing_events(LISTING_EVENTS_PATH)
    saved = set(events.loc[events["event_type"].astype(str).eq("Saved"), "address"].astype(str)) if not events.empty else set()
    profile = load_family_profile(FAMILY_PROFILE_PATH)
    search = top_header(len(saved))

    prices = pd.to_numeric(df["price_numeric"], errors="coerce")
    min_price = int(max(0, prices.min(skipna=True) // 50000 * 50000)) if prices.notna().any() else 0
    max_price = int((prices.max(skipna=True) // 50000 + 1) * 50000) if prices.notna().any() else 3500000
    default_high = min(max_price, max(2500000, int(prices.quantile(0.85)) if prices.notna().any() else 2500000))

    st.sidebar.markdown("### Filters")
    price_range = st.sidebar.slider("Price range", min_price, max_price, (min_price, default_high), step=50000, format="$%d", key="filter_price_range")
    location_options = ["North Vancouver", "West Vancouver"] + sorted(AREA_KEYWORDS.keys())
    locations = st.sidebar.multiselect("Location", location_options, default=[], key="filter_locations")
    bed_choice = st.sidebar.radio("Bedrooms", ["Any", "2+", "3+", "4+", "5+"], index=2, horizontal=True, key="filter_bedrooms")
    min_beds = 0 if bed_choice == "Any" else int(bed_choice.replace("+", ""))
    school_choice = st.sidebar.selectbox("School quality", ["Any", "7.0+", "8.0+", "9.0+"], key="filter_school_quality")
    min_school = 0.0 if school_choice == "Any" else float(school_choice.replace("+", ""))
    yard = st.sidebar.selectbox("Yard size", ["Any", "Usable yard", "Large yard signal", "Exclude known poor yard"], key="filter_yard_size")
    recommended_first = st.sidebar.checkbox("Show recommended homes first", value=True, key="filter_recommended_first")

    chip_options = ["Quiet street", "Good for toddlers", "Large backyard", "Move-in ready", "Strong resale", "Avoid busy roads", "Avoid steep driveway"]
    chips = st.sidebar.multiselect("AI preference chips", chip_options, default=["Good for toddlers", "Avoid busy roads", "Large backyard"], key="filter_preference_chips")

    with st.sidebar.expander("More Filters", expanded=False):
        open_house_only = st.checkbox("Open houses only", value=False, key="filter_open_house_only")
        new_only = st.checkbox("Changed/new since last refresh", value=False, key="filter_new_only")
        avoid_high_noise = st.checkbox("Exclude high-noise homes", value="Avoid busy roads" in chips, key="filter_avoid_high_noise")
        assessment_only = st.checkbox("Only homes with BC Assessment entered", value=False, key="filter_assessment_only")
        st.caption(f"Data source: {path.name}")
        if warnings:
            st.caption("Some missing columns were handled automatically.")
        if st.session_state.get("saved_only"):
            st.info("Saved Homes mode is on. Click Saved Homes again to show all.")

    rules = st.session_state.get("ai_rules", {})
    max_for_score = int(rules.get("max_price", price_range[1]))
    if rules.get("min_school"):
        min_school = max(min_school, float(rules["min_school"]))
    if rules.get("avoid_noise"):
        avoid_high_noise = True
        if "Avoid busy roads" not in chips:
            chips = chips + ["Avoid busy roads"]
    if rules.get("yard_focus") and "Good for toddlers" not in chips:
        chips = chips + ["Good for toddlers"]

    prefs = base_prefs(max_for_score, min_beds, min_school, chips)
    scored = apply_family_evaluation(add_area_columns(score_listings(df, prefs)), chips)
    more = {"open_house_only": open_house_only, "new_only": new_only, "avoid_high_noise": avoid_high_noise, "assessment_only": assessment_only}
    visible = filter_data(scored, search, (price_range[0], min(price_range[1], max_for_score)), locations, min_beds, min_school, yard, more)
    if st.session_state.get("saved_only") and saved:
        visible = visible[visible["Address"].astype(str).isin(saved)]
    visible = sort_visible(visible, recommended_first)

    counts = metrics(scored, visible)

    map_col, rec_col = st.columns([0.68, 0.32], gap="large")
    with map_col:
        map_toolbar(counts)
        fmap = app_map(visible)
        if fmap is None:
            st.info("No listings with map coordinates match the current filters.")
        elif st_folium is None:
            st.components.v1.html(fmap._repr_html_(), height=560)
        else:
            map_data = st_folium(fmap, height=560, use_container_width=True, returned_objects=["last_object_clicked"], key="consumer_map")
            picked = clicked_address(visible, (map_data or {}).get("last_object_clicked"))
            if picked:
                st.session_state["selected_address"] = picked

    with rec_col:
        st.markdown("<div class='right-head'><div class='panel-title'>✦ AI Recommendations</div><div class='view-all'>View all</div></div>", unsafe_allow_html=True)
        recs = sort_visible(visible, True).head(3)
        if recs.empty:
            st.info("No recommendations match these filters yet.")
        for i, (_, row) in enumerate(recs.iterrows(), start=1):
            recommendation_card(row, profile, f"rec_{i}_{slug(row.get('Address', ''))}")
        render_chat_panel(visible)

    selected = selected_row(visible if not visible.empty else scored)
    if selected is not None:
        st.divider()
        detail_panel(selected, visible if not visible.empty else scored, profile)

    st.divider()
    st.markdown("<div class='panel-title'>All Listings</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-sub'>Clean property cards replace the old spreadsheet-style table.</div>", unsafe_allow_html=True)
    if visible.empty:
        st.info("No homes match the current search. Try clearing filters or asking AI for a broader search.")
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(visible.head(18).iterrows()):
            with cols[i % 3]:
                listing_card(row, profile, f"listing_{i}_{slug(row.get('Address', ''))}")



if __name__ == "__main__":
    main()






