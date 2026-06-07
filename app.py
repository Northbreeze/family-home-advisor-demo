from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from area_filters import AREA_KEYWORDS, LANDMARKS, add_area_columns, filter_by_area
from buyer_profile import PRESET_PROFILES, get_profile, parse_buyer_profile
from data_cleaning import (
    DEFAULT_INPUT_CANDIDATES,
    add_noise_columns,
    add_open_house_columns,
    choose_listing_sheet,
    find_default_input,
    money,
    normalize_columns,
)
from photo_review import (
    analyze_listing_photo_urls,
    fetch_realtor_photo_urls,
    load_photo_reviews,
    merge_photo_reviews,
    openai_ready,
    upsert_photo_review,
)
from review_store import load_reviews, merge_reviews, upsert_review
from scoring import display_columns, export_excel, filter_by_preferences, marker_color, score_listings
from v2_product import (
    PROFILE_DEFAULTS,
    append_listing_event,
    card_subtitle,
    card_title,
    concerns,
    confidence_level,
    evidence_table,
    listing_links,
    load_family_profile,
    load_listing_events,
    mortgage_helper_need,
    recommendation_category,
    recommendation_sentence,
    save_family_profile,
    verification_steps,
    why_it_may_work,
)
try:
    import folium
    from folium.plugins import Draw
except ImportError:  # pragma: no cover - handled in Streamlit UI
    folium = None
    Draw = None

try:
    from streamlit_folium import st_folium
except ImportError:  # pragma: no cover - handled in Streamlit UI
    st_folium = None


APP_TITLE = "Family Home Advisor V2"
ROOT = Path(__file__).resolve().parent
REVIEWS_PATH = ROOT / "manual_reviews.csv"
PHOTO_REVIEWS_PATH = ROOT / "photo_reviews.csv"
FAMILY_PROFILE_PATH = ROOT / "family_profile.json"
LISTING_EVENTS_PATH = ROOT / "listing_events.csv"
st.set_page_config(page_title=APP_TITLE, layout="wide")

try:
    if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
except Exception:
    pass

STATUS_COLORS = {
    "Liked": "blue",
    "Offered": "purple",
    "Rejected": "darkred",
    "Disliked": "red",
    "Needs Review": "orange",
}


def get_app_mode() -> str:
    env_mode = os.getenv("APP_MODE")
    if env_mode:
        return env_mode.lower()
    local_secret = ROOT / ".streamlit" / "secrets.toml"
    user_secret = Path.home() / ".streamlit" / "secrets.toml"
    if local_secret.exists() or user_secret.exists():
        try:
            return str(st.secrets.get("APP_MODE", "local")).lower()
        except Exception:
            return "local"
    # Streamlit Cloud runs on Linux and should use the packaged workbook, not the local refresh pipeline.
    if os.name != "nt":
        return "demo"
    return "local"


APP_MODE = get_app_mode()
DEMO_MODE = APP_MODE in {"demo", "static", "deployed"}


@st.cache_data(show_spinner=False)
def load_excel(path_text: str, modified_time: float) -> tuple[pd.DataFrame, str, list[str]]:
    path = Path(path_text)
    sheet = choose_listing_sheet(path)
    df = pd.read_excel(path, sheet_name=sheet)
    return df, sheet, list(df.columns)


def safe_int(value: object, default: int = 0) -> int:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return default
    return int(number)

def review_marker_color(row: pd.Series) -> str:
    status = str(row.get("client_status", "Unreviewed"))
    if status in STATUS_COLORS:
        return STATUS_COLORS[status]
    return marker_color(row)


def make_map(df: pd.DataFrame, enable_draw: bool = False) -> folium.Map | None:
    if folium is None:
        return None

    mapped = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if mapped.empty:
        return None

    fmap = folium.Map(location=[mapped["Latitude"].mean(), mapped["Longitude"].mean()], zoom_start=12, tiles="OpenStreetMap")
    for _, row in mapped.iterrows():
        new_badge = "<br><b>New since last refresh</b>" if bool(row.get("is_new_since_last_refresh", False)) else ""
        popup_html = f"""
        <b>{row.get('Address', '')}</b><br>
        Review status: {row.get('client_status', 'Unreviewed')}<br>
        Score: {row.get('match_score', 0)}<br>
        Bucket: {row.get('recommendation_bucket', 'Review')}<br>
        Area: {row.get('detected_area', 'Unknown')}<br>
        Price: {money(row.get('price_numeric'))}<br>
        School: {row.get('final_school', 'Unknown')} ({row.get('final_fraser_score', 'N/A')})<br>
        Noise: {row.get('noise_risk', 'Unknown')} / yard {row.get('yard_noise', 'Unknown')}<br>
        Yard: {row.get('yard_playability', 'Unknown')}<br>
        Layout: {row.get('layout_fit', 'Unknown')}<br>
        Why: {row.get('final_verdict', row.get('explanation', ''))}{new_badge}<br>
        <a href="{row.get('Listing URL', '')}" target="_blank">Listing URL</a><br>
        <a href="{row.get('Google Maps Link', '')}" target="_blank">Google Maps</a>
        """
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            tooltip=f"{card_title(row)} - {recommendation_category(row)}",
            popup=folium.Popup(popup_html, max_width=420),
            icon=folium.Icon(color=v2_marker_color(row)),
        ).add_to(fmap)

    if enable_draw and Draw is not None:
        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polygon": True,
                "rectangle": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(fmap)
    return fmap


def map_html(fmap: folium.Map | None) -> str:
    return "" if fmap is None else fmap._repr_html_()


def extract_drawn_bounds(map_data: dict | None) -> tuple[float, float, float, float] | None:
    drawings = (map_data or {}).get("all_drawings") or []
    if not drawings:
        return None
    geometry = drawings[-1].get("geometry", {})
    coords = geometry.get("coordinates") or []
    points: list[tuple[float, float]] = []

    def collect(value: object) -> None:
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]):
            lon, lat = float(value[0]), float(value[1])
            points.append((lat, lon))
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(coords)
    if not points:
        return None
    lats = [point[0] for point in points]
    lons = [point[1] for point in points]
    return min(lats), max(lats), min(lons), max(lons)


def filter_by_bounds(df: pd.DataFrame, bounds: tuple[float, float, float, float] | None) -> pd.DataFrame:
    if bounds is None:
        return df
    min_lat, max_lat, min_lon, max_lon = bounds
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    return df[lat.between(min_lat, max_lat) & lon.between(min_lon, max_lon)].copy()


def normalize_listing_url_key(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace("https://www.realtor.ca", "", regex=False)
        .str.replace("http://www.realtor.ca", "", regex=False)
        .str.replace("https://realtor.ca", "", regex=False)
        .str.replace("http://realtor.ca", "", regex=False)
        .str.split("?").str[0]
        .str.rstrip("/")
    )


def has_url_key(df: pd.DataFrame) -> bool:
    return any(column in df.columns and df[column].notna().any() for column in ["Listing URL", "Website"])


def url_key(df: pd.DataFrame) -> pd.Series:
    if "Listing URL" in df.columns and df["Listing URL"].notna().any():
        return "URL:" + normalize_listing_url_key(df["Listing URL"])
    if "Website" in df.columns and df["Website"].notna().any():
        return "URL:" + normalize_listing_url_key(df["Website"])
    return pd.Series("", index=df.index)


def address_key(df: pd.DataFrame) -> pd.Series:
    if "Address" not in df.columns:
        return pd.Series("", index=df.index)
    return "ADDR:" + df["Address"].fillna("").astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def listing_identity_key(df: pd.DataFrame, key_type: str) -> pd.Series:
    if key_type == "url":
        return url_key(df)
    if key_type == "mls" and "MLS" in df.columns:
        return "MLS:" + df["MLS"].fillna("").astype(str).str.upper().str.strip()
    return address_key(df)


def shared_listing_key_type(current: pd.DataFrame, previous: pd.DataFrame) -> str:
    if has_url_key(current) and has_url_key(previous):
        return "url"
    if "MLS" in current.columns and "MLS" in previous.columns and current["MLS"].notna().any() and previous["MLS"].notna().any():
        return "mls"
    return "address"


BC_ASSESSMENT_SEARCH_URL = "https://www.bcassessment.ca/Property/AssessmentSearch?sp=1"


def fix_bc_assessment_links(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["BC Assessment Search Link"] = BC_ASSESSMENT_SEARCH_URL
    return data


def add_new_since_last_refresh(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    data = df.copy()
    data["is_new_since_last_refresh"] = False
    data["listing_change_status"] = "Existing"

    def apply_change_log() -> pd.Series:
        change_path = root / "listing_change_log.csv"
        if not change_path.exists():
            return pd.Series(False, index=data.index)
        try:
            changes = pd.read_csv(change_path)
        except Exception:
            return pd.Series(False, index=data.index)
        if changes.empty:
            return pd.Series(False, index=data.index)

        ref_urls = set()
        for column in ["Current Listing URL", "Listing URL", "Website"]:
            if column in changes.columns:
                ref_urls.update(normalize_listing_url_key(changes[column]).dropna().astype(str))
        ref_urls = {value for value in ref_urls if value and value.lower() not in {"nan", "none"}}

        ref_mls = set()
        for column in ["Current MLS", "MLS"]:
            if column in changes.columns:
                ref_mls.update(changes[column].fillna("").astype(str).str.upper().str.strip())
        ref_mls = {value for value in ref_mls if value and value not in {"NAN", "NONE"}}

        ref_addresses = set(address_key(changes).dropna().astype(str)) if "Address" in changes.columns else set()
        ref_addresses = {value for value in ref_addresses if len(value) > 8}

        mask = pd.Series(False, index=data.index)
        if ref_urls:
            current_url = url_key(data).str.replace("URL:", "", regex=False)
            mask = mask | current_url.isin(ref_urls)
        if ref_mls and "MLS" in data.columns:
            current_mls = data["MLS"].fillna("").astype(str).str.upper().str.strip()
            mask = mask | current_mls.isin(ref_mls)
        if ref_addresses and "Address" in data.columns:
            mask = mask | address_key(data).isin(ref_addresses)

        if "Change Type" in changes.columns:
            change_by_address = changes.drop_duplicates("Address", keep="last").set_index(address_key(changes))["Change Type"].to_dict() if "Address" in changes.columns else {}
            if change_by_address and "Address" in data.columns:
                mapped_status = address_key(data).map(change_by_address)
                data.loc[mapped_status.notna(), "listing_change_status"] = mapped_status[mapped_status.notna()]
        return mask

    is_new = apply_change_log()

    # Local fallback for development: compare dated workbooks when the change log is absent.
    if not is_new.any():
        files = sorted(root.glob("North_West_Vancouver_Houses_Open_Houses_WORKING_URLS_*.xlsx"), key=lambda path: path.stat().st_mtime)
        if len(files) >= 2:
            previous = pd.read_excel(files[-2])
            key_type = shared_listing_key_type(data, previous)
            previous_keys = set(listing_identity_key(previous, key_type).dropna().astype(str))
            current_keys = listing_identity_key(data, key_type).astype(str)
            valid_keys = current_keys.str.len().gt(5)
            is_new = valid_keys & ~current_keys.isin(previous_keys)
            data.loc[is_new, "listing_change_status"] = "New Address"

    data["is_new_since_last_refresh"] = is_new
    data.loc[is_new & data["listing_change_status"].eq("Existing"), "listing_change_status"] = "Changed Since Last Refresh"
    return data

def listing_label(row: pd.Series) -> str:
    return f"{row.get('match_score', 0):.1f} | {row.get('client_status', 'Unreviewed')} | {row.get('Address', '')}"


def needs_ai_photo_score(row: pd.Series) -> bool:
    status = str(row.get("photo_review_status", "")).strip().lower()
    if status in {"ai reviewed", "no realtor photos", "ai photo failed"}:
        return False
    score = pd.to_numeric(row.get("ai_photo_url_count", 0), errors="coerce")
    return pd.isna(score) or int(score) <= 0


def auto_score_photo_candidates(candidates: pd.DataFrame, limit: int = 5) -> tuple[int, list[str]]:
    if not openai_ready() or candidates.empty:
        return 0, []

    scored_count = 0
    messages: list[str] = []
    pending = candidates[candidates.apply(needs_ai_photo_score, axis=1)].head(limit)
    for _, row in pending.iterrows():
        address = str(row.get("Address", ""))
        mls = row.get("MLS", "")
        if not str(mls).strip():
            upsert_photo_review(
                PHOTO_REVIEWS_PATH,
                address,
                {"photo_review_status": "AI photo failed", "ai_photo_notes": "Missing MLS number for Realtor.ca photo lookup."},
            )
            continue
        try:
            photo_urls = fetch_realtor_photo_urls(mls, max_photos=12)
            if not photo_urls:
                upsert_photo_review(
                    PHOTO_REVIEWS_PATH,
                    address,
                    {"photo_review_status": "No Realtor photos", "ai_photo_notes": "No Realtor.ca photo URLs returned for this MLS.", "ai_photo_url_count": 0},
                )
                messages.append(f"No photos: {address}")
                continue
            result = analyze_listing_photo_urls(photo_urls)
            upsert_photo_review(PHOTO_REVIEWS_PATH, address, result)
            scored_count += 1
            messages.append(f"AI scored: {address}")
        except Exception as exc:
            error_text = str(exc)
            upsert_photo_review(
                PHOTO_REVIEWS_PATH,
                address,
                {"photo_review_status": "AI photo failed", "ai_photo_notes": f"AI photo scoring failed: {error_text}"},
            )
            messages.append(f"Failed: {address}")
            if "insufficient_quota" in error_text or "exceeded your current quota" in error_text or "Error code: 429" in error_text:
                st.session_state["ai_photo_quota_blocked"] = True
                messages.append("OpenAI quota/billing limit reached. Automatic photo scoring paused.")
                break
    return scored_count, messages




def inject_v2_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e8ece8;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 1px 8px rgba(20, 46, 35, 0.04);
        }
        div[data-testid="stMetricLabel"] { color: #54645c; }
        div[data-testid="stMetricValue"] { color: #0f5132; font-weight: 700; }
        section[data-testid="stSidebar"] { background: #f7faf7; }
        .v2-soft-note {
            background: #eef8f1;
            border: 1px solid #d8ecdc;
            border-radius: 10px;
            padding: 12px 14px;
            color: #17462f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def v2_marker_color(row: pd.Series) -> str:
    category = recommendation_category(row)
    status = str(row.get("client_status", "Unreviewed"))
    if status in STATUS_COLORS:
        return STATUS_COLORS[status]
    return {
        "Excellent Match": "green",
        "Worth Touring": "orange",
        "Significant Trade-offs": "lightred",
        "Excluded by Deal Breaker": "red",
        "Needs More Information": "blue",
    }.get(category, "blue")


def render_reason_list(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    for item in items:
        st.write(f"- {item}")


def render_listing_actions(row: pd.Series, key_prefix: str = "action") -> None:
    address = str(row.get("Address", ""))
    cols = st.columns(4)
    if cols[0].button("Save", key=f"{key_prefix}_save_{address}"):
        append_listing_event(LISTING_EVENTS_PATH, "saved", address)
        upsert_review(REVIEWS_PATH, address, {"client_status": "Liked"})
        st.success("Saved this home.")
        st.rerun()
    if cols[1].button("Add to tour", key=f"{key_prefix}_tour_{address}"):
        append_listing_event(LISTING_EVENTS_PATH, "planned_visit", address)
        upsert_review(REVIEWS_PATH, address, {"client_status": "Needs Review"})
        st.success("Added to the open house plan.")
        st.rerun()
    if cols[2].button("Reject", key=f"{key_prefix}_reject_{address}"):
        st.session_state["rejecting_address"] = address
    links = listing_links(row)
    realtor_url = links.get("Realtor.ca", "")
    if realtor_url:
        cols[3].link_button("Realtor.ca", realtor_url)
    else:
        cols[3].button("Realtor.ca", disabled=True, key=f"{key_prefix}_disabled_realtor_{address}")

    if st.session_state.get("rejecting_address") == address:
        with st.form(f"reject_form_{key_prefix}_{address}"):
            reason = st.selectbox(
                "Why reject this home?",
                ["Too small", "Bad location", "Close to highway / noisy", "Yard not usable", "Too expensive", "No mortgage helper", "Too much renovation", "Bad layout", "Bad schools", "Other"],
            )
            details = st.text_input("Optional note")
            if st.form_submit_button("Save rejection"):
                append_listing_event(LISTING_EVENTS_PATH, "rejected_online", address, reason, details)
                upsert_review(REVIEWS_PATH, address, {"client_status": "Rejected", "review_notes": f"Rejected online: {reason}. {details}".strip()})
                st.session_state.pop("rejecting_address", None)
                st.success("Rejection saved.")
                st.rerun()


def render_selected_home_summary(row: pd.Series, profile: dict[str, object], key_prefix: str = "selected") -> None:
    category = recommendation_category(row)
    confidence = confidence_level(row)
    st.markdown(f"### {card_title(row)}")
    st.caption(card_subtitle(row))
    school = row.get("final_school", "Unknown")
    fraser = row.get("final_fraser_score", "N/A")
    st.markdown(f"**Recommendation:** {category}")
    st.caption(f"Confidence: {confidence} | School: {school} {fraser}")
    st.write(recommendation_sentence(row, profile))
    render_reason_list("Why it may work for your family", why_it_may_work(row, profile, limit=4))
    render_reason_list("What needs checking", concerns(row, profile, limit=4))
    render_reason_list("Recommended next step", verification_steps(row, profile, limit=3))
    render_listing_actions(row, key_prefix=key_prefix)


def render_listing_report(row: pd.Series, profile: dict[str, object]) -> None:
    st.markdown(f"## {card_title(row)}")
    st.caption(card_subtitle(row))
    st.markdown(f"**{recommendation_category(row)}** | Confidence: **{confidence_level(row)}**")
    st.markdown("#### Plain-English Recommendation")
    st.write(recommendation_sentence(row, profile))
    cols = st.columns(2)
    with cols[0]:
        render_reason_list("Why It May Work", why_it_may_work(row, profile, limit=5))
    with cols[1]:
        render_reason_list("Concerns / Trade-offs", concerns(row, profile, limit=5))
    render_reason_list("What to Verify in Person", verification_steps(row, profile, limit=5))
    st.markdown("#### Evidence")
    st.dataframe(evidence_table(row, profile), use_container_width=True, hide_index=True)
    st.markdown("#### Links")
    link_cols = st.columns(4)
    for idx, (label, url) in enumerate(listing_links(row).items()):
        if url:
            link_cols[idx % 4].link_button(label, url)


def render_shortlist_card(row: pd.Series, profile: dict[str, object], idx: int) -> None:
    with st.container(border=True):
        st.markdown(f"### {card_title(row)}")
        st.caption(card_subtitle(row))
        st.markdown(f"**{recommendation_category(row)}** | Confidence: **{confidence_level(row)}**")
        open_house = str(row.get("open_house_raw", row.get("Open House", "")) or "").strip()
        if open_house and open_house.lower() != "nan":
            st.caption(f"Open house: {open_house}")
        render_reason_list("Why it made your shortlist", why_it_may_work(row, profile, limit=3))
        biggest = concerns(row, profile, limit=1)[0]
        st.markdown(f"**Biggest uncertainty:** {biggest}")
        render_listing_actions(row, key_prefix=f"shortlist_{idx}")


def render_feedback_after_visit(row: pd.Series) -> None:
    address = str(row.get("Address", ""))
    with st.expander("After-open-house feedback", expanded=False):
        with st.form(f"visited_feedback_{address}"):
            outcome = st.selectbox("How was it?", ["Better than expected", "As expected", "Worse than expected"])
            reason = st.selectbox(
                "What changed your mind?",
                ["Noise worse than expected", "Yard smaller / not usable", "Layout felt cramped", "House felt too small", "Photos were misleading", "Renovation quality poor", "Neighbourhood not right", "Loved the location", "Loved the yard", "Good family fit", "Other"],
            )
            details = st.text_area("Notes", height=80)
            if st.form_submit_button("Save visit feedback"):
                append_listing_event(LISTING_EVENTS_PATH, "visited_open_house", address, outcome, f"{reason}. {details}".strip())
                upsert_review(REVIEWS_PATH, address, {"client_status": "Needs Review", "review_notes": f"Visited: {outcome}. {reason}. {details}".strip()})
                st.success("Open house feedback saved.")
                st.rerun()


def render_family_profile_editor(profile: dict[str, object]) -> None:
    st.subheader("Family Profile")
    st.caption("A simple decision profile for weekend touring. Edit the high-impact rules first; everything else is tucked away.")
    budget = dict(profile.get("budget_logic", PROFILE_DEFAULTS["budget_logic"]))

    with st.container(border=True):
        st.markdown("### Budget & mortgage-helper rule")
        st.caption("This is the most important rule because helper need changes with price.")
        cols = st.columns(2)
        with cols[0]:
            helper_optional_max = st.number_input("Helper optional up to", min_value=0, value=int(budget.get("helper_optional_max", 2100000)), step=50000)
        with cols[1]:
            helper_helpful_max = st.number_input("Helper helpful up to", min_value=helper_optional_max, value=int(budget.get("helper_helpful_max", 2500000)), step=50000)
        st.markdown(f"Under ${helper_optional_max:,.0f}, a suite is optional. Above ${helper_helpful_max:,.0f}, it becomes important.")

    location_col, rules_col = st.columns(2)
    with location_col:
        with st.container(border=True):
            st.markdown("### Locations")
            preferred_cities = st.multiselect("Preferred cities", ["West Vancouver", "North Vancouver"], default=list(profile.get("preferred_cities", ["West Vancouver", "North Vancouver"])))
            preferred_neighbourhoods = st.multiselect("Preferred neighbourhoods", sorted(AREA_KEYWORDS.keys()), default=[x for x in profile.get("preferred_neighbourhoods", []) if x in AREA_KEYWORDS])
            avoided_areas = st.multiselect("Avoided areas", sorted(AREA_KEYWORDS.keys()), default=[x for x in profile.get("avoided_areas", []) if x in AREA_KEYWORDS])
    with rules_col:
        with st.container(border=True):
            st.markdown("### Deal breakers")
            deal_breakers = st.multiselect(
                "What should clearly flag or exclude a home?",
                ["Highway noise", "Busy arterial road", "Steep or unusable yard", "Too small interior", "Bad location", "Major structural renovation", "Below minimum school threshold"],
                default=list(profile.get("deal_breakers", [])),
            )

    with st.expander("Important preferences", expanded=False):
        important = st.text_area("One preference per line", value="\n".join(profile.get("important_preferences", [])), height=130)
    with st.expander("Flexible preferences", expanded=False):
        flexible = st.text_area("What are you willing to compromise on?", value="\n".join(profile.get("flexible_preferences", [])), height=100)
    with st.expander("What the app has learned", expanded=True):
        learned = st.text_area("Editable learned rules", value="\n".join(profile.get("learned_rules", [])), height=130)

    if st.button("Save Family Profile", type="primary"):
        updated = dict(profile)
        updated["budget_logic"] = {**budget, "helper_optional_max": helper_optional_max, "helper_helpful_max": helper_helpful_max}
        updated["preferred_cities"] = preferred_cities
        updated["preferred_neighbourhoods"] = preferred_neighbourhoods
        updated["avoided_areas"] = avoided_areas
        updated["deal_breakers"] = deal_breakers
        updated["important_preferences"] = [line.strip() for line in important.splitlines() if line.strip()]
        updated["flexible_preferences"] = [line.strip() for line in flexible.splitlines() if line.strip()]
        updated["learned_rules"] = [line.strip() for line in learned.splitlines() if line.strip()]
        save_family_profile(FAMILY_PROFILE_PATH, updated)
        st.success("Family profile saved.")
        st.rerun()

def render_advisor_chat_placeholder(profile: dict[str, object]) -> None:
    st.subheader("Advisor Chat")
    st.caption("V2 placeholder: the app suggests structured rules and asks before saving them. Full LLM integration can come later.")
    note = st.text_area("Tell the advisor what you learned", placeholder="Example: We can renovate, but highway noise is a deal breaker.", height=120)
    if note.strip():
        suggestions = []
        lowered = note.lower()
        if "highway" in lowered or "noise" in lowered:
            suggestions.append("Highway or traffic noise should be treated as a deal breaker.")
        if "renovat" in lowered:
            suggestions.append("Cosmetic renovation is flexible if the location and yard work.")
        if "mortgage" in lowered or "helper" in lowered:
            suggestions.append("Mortgage helper importance should depend on price thresholds.")
        if not suggestions:
            suggestions.append("Save this as a learned family preference for future review.")
        st.markdown("**Suggested memory/rule**")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
        cols = st.columns(3)
        if cols[0].button("Save rule"):
            updated = dict(profile)
            learned = list(updated.get("learned_rules", []))
            learned.extend(suggestions)
            updated["learned_rules"] = list(dict.fromkeys(learned))
            save_family_profile(FAMILY_PROFILE_PATH, updated)
            append_listing_event(LISTING_EVENTS_PATH, "advisor_rule_saved", "", "; ".join(suggestions), note)
            st.success("Rule saved to Family Profile.")
            st.rerun()
        if cols[1].button("Ignore"):
            st.info("Nothing saved.")
def render_listing_card(row: pd.Series) -> None:
    st.subheader(str(row.get("Address", "Selected Listing")))
    metric_cols = st.columns(5)
    metric_cols[0].metric("Match", f"{row.get('match_score', 0):.1f}")
    metric_cols[1].metric("Bucket", str(row.get("recommendation_bucket", "Review")))
    metric_cols[2].metric("Price", money(row.get("price_numeric")))
    metric_cols[3].metric("Beds / Baths", f"{row.get('Bedrooms', 'N/A')} / {row.get('Bathrooms', 'N/A')}")
    metric_cols[4].metric("Area", str(row.get("detected_area", "Unknown")))

    st.markdown(f"**Verdict:** {row.get('final_verdict', row.get('explanation', ''))}")
    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.write("**Family Fit**")
        st.write(f"Yard: `{row.get('yard_playability', 'Unknown')}`")
        st.write(f"Layout: `{row.get('layout_fit', 'Unknown')}`")
        st.write(f"Flags: {row.get('buyer_fit_flags', '')}")
    with detail_cols[1]:
        st.write("**Noise / Location**")
        st.write(f"Noise: `{row.get('noise_risk', 'Unknown')}`")
        st.write(f"Yard noise: `{row.get('yard_noise', 'Unknown')}`")
        st.write(f"Highway distance: `{row.get('distance_to_highway_m', 'N/A')}` m")
    with detail_cols[2]:
        st.write("**School / Value**")
        st.write(f"School: {row.get('final_school', 'Unknown')}")
        st.write(f"Fraser: {row.get('final_fraser_score', 'N/A')}")
        st.write(f"BC Assessment: {money(row.get('bc_assessment_total_value'))}")
        st.write(row.get("assessment_interpretation", ""))

    component_data = pd.DataFrame(
        [
            {"Component": "Location", "Score": row.get("location_component", 0)},
            {"Component": "Yard", "Score": row.get("backyard_component", 0)},
            {"Component": "Layout", "Score": row.get("layout_component", 0)},
            {"Component": "Interior Size", "Score": row.get("size_component", 0)},
            {"Component": "Quiet", "Score": row.get("quiet_component", 0)},
            {"Component": "School", "Score": row.get("school_component", 0)},
            {"Component": "Value", "Score": row.get("price_component", 0)},
        ]
    )
    st.write("**Component Scores**")
    st.dataframe(component_data, use_container_width=True, hide_index=True)
    st.write("**Verification Checklist**")
    for item in str(row.get("verification_checklist", "")).split(" | "):
        st.write(f"- {item}")

    with st.expander("BC Assessment Detail", expanded=False):
        st.write(f"Total: {money(row.get('bc_assessment_total_value'))}")
        st.write(f"Land: {money(row.get('bc_assessment_land_value'))}")
        st.write(f"Building: {money(row.get('bc_assessment_building_value'))}")
        st.write(f"Price / Assessment: {row.get('assessment_price_ratio', 'N/A')}")
        st.write(f"Land share: {row.get('land_value_share', 'N/A')}%")
        st.write(f"Building share: {row.get('building_value_share', 'N/A')}%")

    render_photo_review(row)

    links = []
    if str(row.get("Listing URL", "")).strip():
        links.append(f"[Listing]({row.get('Listing URL')})")
    if str(row.get("Google Maps Link", "")).strip():
        links.append(f"[Google Maps]({row.get('Google Maps Link')})")
    if str(row.get("BC Assessment Search Link", "")).strip():
        links.append(f"[BC Assessment]({row.get('BC Assessment Search Link')})")
    if links:
        st.markdown(" | ".join(links))

    with st.expander("Listing Description", expanded=False):
        st.write(row.get("Description", "No description available."))


def render_photo_review(row: pd.Series) -> None:
    address = str(row.get("Address", ""))
    mls = row.get("MLS", "")
    with st.expander("AI Photo Scoring", expanded=False):
        st.write("Analyze Realtor.ca photo URLs directly with OpenAI. The app stores scores only, not photos.")
        st.caption(f"MLS used for photo lookup: {mls or 'missing'}")

        score_data = pd.DataFrame(
            [
                {"Photo Signal": "Yard", "Score": row.get("ai_yard_score", 50)},
                {"Photo Signal": "Layout", "Score": row.get("ai_layout_score", 50)},
                {"Photo Signal": "Privacy", "Score": row.get("ai_privacy_score", 50)},
                {"Photo Signal": "Fence", "Score": row.get("ai_fence_score", 50)},
                {"Photo Signal": "Flat / usable slope", "Score": row.get("ai_slope_score", 50)},
                {"Photo Signal": "Quiet visual clues", "Score": row.get("ai_noise_clue_score", 50)},
            ]
        )
        st.dataframe(score_data, use_container_width=True, hide_index=True)

        st.write("**Current AI Photo Notes**")
        st.write(f"Status: `{row.get('photo_review_status', 'Not reviewed')}`")
        st.write(f"Yard label: `{row.get('photo_yard_playability', row.get('ai_yard_playability', 'Unknown'))}`")
        st.write(f"Yard type: `{row.get('photo_yard_type', row.get('ai_yard_type', 'Unknown'))}`")
        st.write(f"Flatness: `{row.get('photo_flatness', row.get('ai_flatness', 'Unknown'))}`")
        st.write(f"Fenced: `{row.get('photo_fenced', row.get('ai_fenced', 'Unknown'))}`")
        st.write(f"Privacy: `{row.get('photo_privacy', row.get('ai_privacy', 'Unknown'))}`")
        st.write(f"Photo URLs analyzed: `{row.get('ai_photo_url_count', 0)}`")
        if str(row.get('ai_photo_notes', '')).strip():
            st.write(f"Notes: {row.get('ai_photo_notes')}")

        if not openai_ready():
            st.warning("OPENAI_API_KEY is not configured. Add it to `.streamlit/secrets.toml` to run AI photo scoring.")

        ai_cols = st.columns(2)
        with ai_cols[0]:
            if st.button("Run AI photo scoring", disabled=not openai_ready(), key=f"ai_photo_{address}"):
                try:
                    with st.spinner("Fetching Realtor.ca photo URLs and scoring yard/layout/privacy/noise..."):
                        photo_urls = fetch_realtor_photo_urls(mls, max_photos=12)
                        if not photo_urls:
                            st.warning("No Realtor.ca photo URLs were returned for this MLS number.")
                            return
                        result = analyze_listing_photo_urls(photo_urls)
                    upsert_photo_review(PHOTO_REVIEWS_PATH, address, result)
                    st.success("AI photo scores saved. They will be included in ranking and Excel export.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"AI photo scoring failed: {exc}")
        with ai_cols[1]:
            if st.button("Apply AI photo scores", key=f"apply_photo_{address}"):
                suggested_yard = row.get("ai_yard_playability") or row.get("photo_yard_playability") or "Unknown"
                values = {
                    "yard_playability": suggested_yard if suggested_yard in {"Great", "Maybe", "Poor", "Unknown"} else "Unknown",
                    "photo_review_status": row.get("photo_review_status", "Reviewed"),
                    "photo_yard_playability": suggested_yard,
                    "photo_yard_type": row.get("ai_yard_type", row.get("photo_yard_type", "Unknown")),
                    "photo_flatness": row.get("ai_flatness", row.get("photo_flatness", "Unknown")),
                    "photo_fenced": row.get("ai_fenced", row.get("photo_fenced", "Unknown")),
                    "photo_privacy": row.get("ai_privacy", row.get("photo_privacy", "Unknown")),
                    "photo_notes": row.get("ai_photo_notes", row.get("photo_notes", "")),
                }
                upsert_review(REVIEWS_PATH, address, values)
                st.success("AI photo scores applied to Realtor Review.")
                st.rerun()


def render_bulk_assessment_editor(top: pd.DataFrame) -> None:
    st.subheader("Top 20 BC Assessment Entry")
    if top.empty:
        st.info("No top candidates available for BC Assessment entry.")
        return

    columns = [
        "Address",
        "Price",
        "match_score",
        "bc_assessment_total_value",
        "bc_assessment_land_value",
        "bc_assessment_building_value",
        "BC Assessment Search Link",
    ]
    editor_df = top[[column for column in columns if column in top.columns]].head(20).copy()
    for column in ["bc_assessment_total_value", "bc_assessment_land_value", "bc_assessment_building_value"]:
        if column not in editor_df.columns:
            editor_df[column] = pd.NA
        editor_df[column] = pd.to_numeric(editor_df[column], errors="coerce")

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Address": st.column_config.TextColumn("Address", disabled=True),
            "Price": st.column_config.TextColumn("Price", disabled=True),
            "match_score": st.column_config.NumberColumn("Current Score", disabled=True, format="%.1f"),
            "bc_assessment_total_value": st.column_config.NumberColumn("Assessment Total", min_value=0, step=50000, format="$%d"),
            "bc_assessment_land_value": st.column_config.NumberColumn("Land Value", min_value=0, step=50000, format="$%d"),
            "bc_assessment_building_value": st.column_config.NumberColumn("Building Value", min_value=0, step=25000, format="$%d"),
            "BC Assessment Search Link": st.column_config.LinkColumn("BC Assessment", disabled=True),
        },
        key="bulk_bc_assessment_editor",
    )

    if st.button("Save BC Assessment values and rescore"):
        saved_count = 0
        for _, row in edited.iterrows():
            address = str(row.get("Address", "")).strip()
            if not address:
                continue
            values = {
                "bc_assessment_total_value": pd.to_numeric(row.get("bc_assessment_total_value"), errors="coerce"),
                "bc_assessment_land_value": pd.to_numeric(row.get("bc_assessment_land_value"), errors="coerce"),
                "bc_assessment_building_value": pd.to_numeric(row.get("bc_assessment_building_value"), errors="coerce"),
            }
            if all(pd.isna(value) for value in values.values()):
                continue
            values = {key: (pd.NA if pd.isna(value) else int(value)) for key, value in values.items()}
            upsert_review(REVIEWS_PATH, address, values)
            saved_count += 1
        st.success(f"Saved BC Assessment values for {saved_count} listing(s). Scores will refresh now.")
        st.rerun()


def render_review_form(row: pd.Series) -> None:
    st.subheader("Realtor Review")
    with st.form("manual_review_form"):
        status = st.selectbox(
            "Client status",
            ["Unreviewed", "Liked", "Needs Review", "Offered", "Disliked", "Rejected"],
            index=["Unreviewed", "Liked", "Needs Review", "Offered", "Disliked", "Rejected"].index(str(row.get("client_status", "Unreviewed")) if str(row.get("client_status", "Unreviewed")) in ["Unreviewed", "Liked", "Needs Review", "Offered", "Disliked", "Rejected"] else "Unreviewed"),
        )
        form_cols = st.columns(3)
        with form_cols[0]:
            noise_verified = st.selectbox("Noise verified", ["Unknown", "Low", "Medium", "High"], index=["Unknown", "Low", "Medium", "High"].index(str(row.get("noise_verified", "Unknown")) if str(row.get("noise_verified", "Unknown")) in ["Unknown", "Low", "Medium", "High"] else "Unknown"))
            yard_noise = st.selectbox("Yard noise", ["Unknown", "Low", "Medium", "High"], index=["Unknown", "Low", "Medium", "High"].index(str(row.get("yard_noise", "Unknown")) if str(row.get("yard_noise", "Unknown")) in ["Unknown", "Low", "Medium", "High"] else "Unknown"))
        with form_cols[1]:
            yard_playability = st.selectbox("Yard playability", ["Unknown", "Great", "Maybe", "Poor"], index=["Unknown", "Great", "Maybe", "Poor"].index(str(row.get("yard_playability", "Unknown")) if str(row.get("yard_playability", "Unknown")) in ["Unknown", "Great", "Maybe", "Poor"] else "Unknown"))
            layout_fit = st.selectbox("Layout fit", ["Unknown", "Great", "Good", "Concern"], index=["Unknown", "Great", "Good", "Concern"].index(str(row.get("layout_fit", "Unknown")) if str(row.get("layout_fit", "Unknown")) in ["Unknown", "Great", "Good", "Concern"] else "Unknown"))
        with form_cols[2]:
            assessment_total = st.number_input("BC Assessment total", min_value=0, value=safe_int(row.get("bc_assessment_total_value")), step=50000)
            assessment_land = st.number_input("BC Assessment land", min_value=0, value=safe_int(row.get("bc_assessment_land_value")), step=50000)
            assessment_building = st.number_input("BC Assessment building", min_value=0, value=safe_int(row.get("bc_assessment_building_value")), step=25000)
        notes = st.text_area("Showing / review notes", value=str(row.get("review_notes", "") or ""), height=90)
        saved = st.form_submit_button("Save review")

    if saved:
        values = {
            "client_status": status,
            "noise_verified": noise_verified,
            "yard_playability": yard_playability,
            "yard_noise": yard_noise,
            "layout_fit": layout_fit,
            "bc_assessment_total_value": assessment_total or pd.NA,
            "bc_assessment_land_value": assessment_land or pd.NA,
            "bc_assessment_building_value": assessment_building or pd.NA,
            "review_notes": notes,
        }
        upsert_review(REVIEWS_PATH, str(row.get("Address", "")), values)
        st.success("Review saved. Scores will refresh now.")
        st.rerun()


def main() -> None:
    st.title("Weekend Open House Planner")
    st.caption("Family Home Advisor V2: map-first browsing, AI shortlist cards, family profile rules, and buyer-friendly listing reports.")
    st.info("V2 is organized around four buyer sections: Map & Listings, AI Shortlist, Family Profile, and Advisor Chat. Raw tables/export tools are under Advanced / Admin.")

    default_path = find_default_input(ROOT)
    if default_path is None:
        st.error("No Excel listing file was found in this folder.")
        return

    st.sidebar.header("Buyer Profile")
    if DEMO_MODE:
        st.sidebar.info("Demo mode: listings are from the packaged workbook. Adjust scoring and download results.")
    else:
        from refresh_pipeline import refresh_family_home_advisor

        st.sidebar.header("Data Refresh")
        refresh_open_houses_only = st.sidebar.checkbox("Refresh open houses only", value=False)
        if st.sidebar.button("Refresh listings now"):
            messages: list[str] = []
            progress_box = st.empty()

            def show_status(message: str) -> None:
                messages.append(message)
                progress_box.info("\n".join(messages[-6:]))

            try:
                with st.spinner("Refreshing Realtor.ca listings and rebuilding the client report..."):
                    refreshed_path = refresh_family_home_advisor(
                        root=ROOT,
                        open_houses_only=refresh_open_houses_only,
                        status=show_status,
                    )
                load_excel.clear()
                st.success(f"Refreshed listings and saved `{refreshed_path.name}`.")
                st.rerun()
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")

    profile_name = st.sidebar.selectbox("Preset profile", list(PRESET_PROFILES.keys()))
    profile_defaults = get_profile(profile_name).defaults()
    source_path = st.sidebar.text_input("Listing workbook", value=str(default_path))

    st.sidebar.header("Map Search")
    market_view_mode = st.sidebar.radio(
        "Market view mode",
        ["Recommended matches", "All active listings"],
        help="Recommended matches applies buyer filters. All active listings shows the full inventory so nothing is hidden by scoring filters.",
    )
    show_all_active = market_view_mode == "All active listings"
    if show_all_active:
        st.sidebar.info("Showing all active listings. Buyer scoring still appears, but price/bed/school/noise/status filters will not hide homes.")
    preferred_area = st.sidebar.selectbox("Preferred area", ["All"] + sorted(AREA_KEYWORDS.keys()))
    landmark = st.sidebar.selectbox("Around landmark", list(LANDMARKS.keys()))
    radius_km = st.sidebar.slider("Landmark radius (km)", 0.5, 5.0, 2.0, 0.25)
    status_filter = st.sidebar.multiselect(
        "Client status",
        ["Unreviewed", "Liked", "Needs Review", "Offered", "Disliked", "Rejected"],
        default=["Unreviewed", "Liked", "Needs Review", "Offered"],
    )
    st.sidebar.caption("Unreviewed means not reviewed by us yet. It does not mean new to market. Use New since last refresh for newly appeared listings.")
    show_new_only = st.sidebar.checkbox("Changed/new since last refresh only", value=False)
    use_drawn_area = st.sidebar.checkbox("Filter by area drawn on map", value=False)
    if use_drawn_area:
        st.sidebar.caption("Use the rectangle/polygon tool on the map, then the app will filter listings inside the drawn box.")

    try:
        source_mtime = Path(source_path).stat().st_mtime
        raw_df, sheet, original_columns = load_excel(source_path, source_mtime)
    except Exception as exc:
        st.error(f"Could not load workbook: {exc}")
        return

    df, warnings = normalize_columns(raw_df)
    df = add_open_house_columns(df)
    df = add_noise_columns(df)
    reviews = load_reviews(REVIEWS_PATH)
    df = merge_reviews(df, reviews)
    photo_reviews = load_photo_reviews(PHOTO_REVIEWS_PATH)
    df = merge_photo_reviews(df, photo_reviews)
    df = add_new_since_last_refresh(df, ROOT)
    df = fix_bc_assessment_links(df)

    max_observed_price = int(pd.to_numeric(df["price_numeric"], errors="coerce").max(skipna=True) or 3000000)
    default_max = min(max_observed_price, 3000000)
    widget_prefix = profile_name.lower().replace(" ", "_")

    st.sidebar.header("Scoring Controls")
    max_price = st.sidebar.number_input("Max price", min_value=0, value=default_max, step=50000, format="%d", key=f"{widget_prefix}_max_price")
    min_bedrooms = st.sidebar.number_input("Minimum bedrooms", min_value=0, value=profile_defaults["min_bedrooms"], step=1, key=f"{widget_prefix}_min_bedrooms")
    min_fraser_score = st.sidebar.slider("Minimum Fraser school score", 0.0, 10.0, float(profile_defaults["min_fraser_score"]), 0.1, key=f"{widget_prefix}_min_fraser_score")
    quiet_importance = st.sidebar.slider("Quiet-location importance", 1, 5, profile_defaults["quiet_importance"], key=f"{widget_prefix}_quiet_importance")
    school_importance = st.sidebar.slider("School importance", 1, 5, profile_defaults["school_importance"], key=f"{widget_prefix}_school_importance")
    price_importance = st.sidebar.slider("Price/value importance", 1, 5, profile_defaults["price_importance"], key=f"{widget_prefix}_price_importance")
    size_importance = st.sidebar.slider("Size importance", 1, 5, profile_defaults["size_importance"], key=f"{widget_prefix}_size_importance")
    lifestyle_importance = st.sidebar.slider("Yard/layout/location importance", 1, 5, profile_defaults["lifestyle_importance"], key=f"{widget_prefix}_lifestyle_importance")
    preferred_city = st.sidebar.selectbox("Preferred city", ["Both", "North Vancouver", "West Vancouver"], index=["Both", "North Vancouver", "West Vancouver"].index(profile_defaults["preferred_city"]), key=f"{widget_prefix}_preferred_city")
    exclude_high_noise = st.sidebar.checkbox("Exclude high-noise homes", value=profile_defaults["exclude_high_noise"], key=f"{widget_prefix}_exclude_high_noise")

    st.sidebar.header("AI Photo Scoring")
    auto_photo_scoring = st.sidebar.checkbox("Auto-score photos for top candidates", value=False)
    auto_photo_limit = st.sidebar.slider("Photo scores per run", 1, 10, 1)
    if auto_photo_scoring and not openai_ready():
        st.sidebar.warning("OPENAI_API_KEY is missing, so automatic photo scoring is paused.")
    if st.sidebar.button("Reset AI photo pause"):
        st.session_state.pop("ai_photo_quota_blocked", None)
        st.rerun()

    with st.sidebar.expander("Profile Helper", expanded=False):
        buyer_text = st.text_area("Describe what this buyer cares about", height=100)
        if buyer_text.strip():
            suggestions = parse_buyer_profile(buyer_text)
            st.info(
                "Suggested weights: "
                f"quiet {suggestions['quiet_importance']}, school {suggestions['school_importance']}, "
                f"price {suggestions['price_importance']}, size {suggestions['size_importance']}, "
                f"lifestyle {suggestions['lifestyle_importance']}."
            )

    prefs = {
        "profile_name": profile_name,
        "deal_breakers": profile_defaults["deal_breakers"],
        "max_price": max_price,
        "min_bedrooms": min_bedrooms,
        "min_fraser_score": min_fraser_score,
        "quiet_importance": quiet_importance,
        "school_importance": school_importance,
        "price_importance": price_importance,
        "size_importance": size_importance,
        "lifestyle_importance": lifestyle_importance,
        "preferred_city": preferred_city,
        "exclude_high_noise": exclude_high_noise,
        "preferred_area": preferred_area,
        "landmark": landmark,
        "radius_km": radius_km,
        "show_all_active": show_all_active,
        "show_new_only": show_new_only,
        "use_drawn_area": use_drawn_area,
        "auto_photo_scoring": auto_photo_scoring,
        "auto_photo_limit": auto_photo_limit,
    }

    scored = score_listings(df, prefs)
    scored = add_area_columns(scored)
    new_since_refresh_all = scored[scored["is_new_since_last_refresh"]].copy() if "is_new_since_last_refresh" in scored else scored.iloc[0:0].copy()
    if "match_score" in new_since_refresh_all.columns:
        new_since_refresh_all = new_since_refresh_all.sort_values("match_score", ascending=False)
    if show_all_active:
        filtered = scored.copy()
        excluded = scored.iloc[0:0].copy()
    else:
        filtered, excluded = filter_by_preferences(scored, prefs)
    filtered = filter_by_area(filtered, preferred_area, landmark, radius_km)
    if status_filter and not show_all_active:
        filtered = filtered[filtered["client_status"].isin(status_filter)]
    if show_new_only:
        filtered = new_since_refresh_all.copy()
        excluded = scored.iloc[0:0].copy()

    drawn_bounds = st.session_state.get("drawn_bounds") if use_drawn_area else None
    filtered_for_map = filtered.copy()
    filtered = filter_by_bounds(filtered, drawn_bounds)
    top = filtered.head(20)
    if show_new_only:
        if filtered.empty:
            st.info("No changed/new listings are available in the current refresh comparison.")
        else:
            st.info(f"New-only mode is showing {len(filtered)} changed/new listing(s), regardless of buyer filters.")

    if auto_photo_scoring and openai_ready() and not top.empty:
        if st.session_state.get("ai_photo_quota_blocked"):
            st.warning("Automatic AI photo scoring is paused because OpenAI returned a quota/billing limit error. Check billing/credits, then click Reset AI photo pause in the sidebar.")
        else:
            pending_count = int(top.apply(needs_ai_photo_score, axis=1).sum())
            if pending_count > 0:
                with st.spinner(f"Automatically scoring listing photos for {min(auto_photo_limit, pending_count)} top candidate(s)..."):
                    scored_count, photo_messages = auto_score_photo_candidates(top, limit=auto_photo_limit)
                if st.session_state.get("ai_photo_quota_blocked"):
                    st.warning("OpenAI quota/billing limit reached. Automatic photo scoring is now paused.")
                if scored_count:
                    st.info("AI photo scoring updated the candidate set. Refreshing rankings with image scores...")
                    st.rerun()

    family_profile = load_family_profile(FAMILY_PROFILE_PATH)
    listing_events = load_listing_events(LISTING_EVENTS_PATH)

    st.markdown("### Weekend Open House Decision Assistant")
    st.caption(
        f"Loaded {len(scored)} active listings. Showing {len(filtered)} on the map. "
        f"{int(scored['is_new_since_last_refresh'].sum()) if 'is_new_since_last_refresh' in scored else 0} changed since the last refresh."
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Active listings", len(scored))
    summary_cols[1].metric("Shown on map", len(filtered))
    summary_cols[2].metric("Weekend/open houses", int(scored["open_house_status"].eq("Upcoming").sum()) if "open_house_status" in scored else 0)
    summary_cols[3].metric("Changed", int(scored["is_new_since_last_refresh"].sum()) if "is_new_since_last_refresh" in scored else 0)

    map_tab, shortlist_tab, profile_tab, chat_tab, advanced_tab = st.tabs([
        "Map & Listings",
        "AI Shortlist",
        "Family Profile",
        "Advisor Chat",
        "Advanced / Admin",
    ])

    with map_tab:
        st.subheader("Map & Listings")
        if filtered.empty:
            st.info("No listings match the current view. Switch Market view mode to All active listings or clear area filters.")
        view_options = filtered.head(120).copy()
        if view_options.empty:
            selected_row = None
        else:
            labels = [f"{recommendation_category(row)} | {card_title(row)} | {money(row.get('price_numeric'))}" for _, row in view_options.iterrows()]
            selected_label = st.selectbox("Choose a home to inspect", labels, key="v2_selected_home", label_visibility="collapsed")
            selected_row = view_options.iloc[labels.index(selected_label)]

        map_col, right_col = st.columns([2.35, 1.0], gap="large")
        with map_col:
            st.caption("All active listings stay visible unless you intentionally filter them. Click a marker or choose a home above to review details.")
            fmap = make_map(filtered_for_map if use_drawn_area else filtered, enable_draw=use_drawn_area)
            if fmap is not None and st_folium is not None:
                map_data = st_folium(fmap, height=680, use_container_width=True, returned_objects=["all_drawings"])
                bounds = extract_drawn_bounds(map_data)
                if use_drawn_area and bounds and bounds != st.session_state.get("drawn_bounds"):
                    st.session_state["drawn_bounds"] = bounds
                    st.rerun()
                if use_drawn_area and st.session_state.get("drawn_bounds"):
                    st.caption(f"Drawn-area filter active: {len(filtered)} listings inside the selected area.")
                    if st.button("Clear drawn area"):
                        st.session_state.pop("drawn_bounds", None)
                        st.rerun()
            elif fmap is not None:
                st.html(map_html(fmap))
            elif folium is None:
                st.warning("Install `folium` to show the interactive map.")
            else:
                st.warning("No listings with latitude/longitude are available for the map.")

            st.markdown("#### AI Shortlist Preview")
            preview_pool = filtered.copy() if not filtered.empty else scored.copy()
            if "match_score" in preview_pool.columns:
                preview_pool = preview_pool.sort_values("match_score", ascending=False)
            preview_cols = st.columns(5)
            for idx, (_, row) in enumerate(preview_pool.head(5).iterrows(), start=1):
                with preview_cols[idx - 1]:
                    with st.container(border=True):
                        st.caption(f"#{idx} {recommendation_category(row)}")
                        st.markdown(f"**{card_title(row)}**")
                        st.write(money(row.get("price_numeric")))
                        st.caption(card_subtitle(row))

        with right_col:
            with st.container(border=True):
                st.markdown("#### Selected Home")
                if selected_row is None:
                    st.info("Choose a home to see the recommendation.")
                else:
                    render_selected_home_summary(selected_row, family_profile, key_prefix="map_selected")
            if selected_row is not None:
                with st.expander("Open full report", expanded=False):
                    render_listing_report(selected_row, family_profile)
                render_feedback_after_visit(selected_row)

        if not new_since_refresh_all.empty:
            with st.expander("Changed since last refresh", expanded=False):
                st.caption("Includes new addresses, relisted homes, and price changes from `listing_change_log.csv`.")
                st.dataframe(display_columns(new_since_refresh_all.head(50)), use_container_width=True, hide_index=True)

    with shortlist_tab:
        st.subheader("AI Shortlist")
        st.caption("Deterministic scoring creates the shortlist. The UI leads with plain-English reasons instead of raw scores.")
        shortlist_size = st.radio("Shortlist size", [5, 10, 15], index=1, horizontal=True)
        shortlist_pool = filtered.copy() if not filtered.empty else scored.copy()
        if "match_score" in shortlist_pool.columns:
            shortlist_pool = shortlist_pool.sort_values("match_score", ascending=False)
        shortlist = shortlist_pool.head(int(shortlist_size))
        if shortlist.empty:
            st.info("No homes available for the shortlist with the current filters.")
        else:
            for idx, (_, row) in enumerate(shortlist.iterrows(), start=1):
                render_shortlist_card(row, family_profile, idx)

    with profile_tab:
        render_family_profile_editor(family_profile)
        st.divider()
        st.subheader("Recent Feedback Events")
        if listing_events.empty:
            st.info("No listing feedback has been saved yet.")
        else:
            st.dataframe(listing_events.tail(25).iloc[::-1], use_container_width=True, hide_index=True)

    with chat_tab:
        render_advisor_chat_placeholder(family_profile)

    with advanced_tab:
        st.subheader("Advanced / Admin")
        with st.expander("Input files and diagnostics", expanded=False):
            st.write(f"Workbook: `{Path(source_path).name}` sheet `{sheet}` with `{len(raw_df)}` rows.")
            st.write("Default input order:", DEFAULT_INPUT_CANDIDATES)
            st.write("Columns found:", original_columns)
            for warning in warnings:
                st.warning(warning)
            if df["noise_estimated"].any():
                st.warning("Some noise risk values are estimated because road GIS/proximity fields were missing.")

        with st.expander("Manual review and BC Assessment", expanded=False):
            render_bulk_assessment_editor(top)
            st.divider()
            if filtered.empty:
                st.info("No listing matches the current filters.")
            else:
                labels = [listing_label(row) for _, row in filtered.iterrows()]
                selected_label = st.selectbox("Choose a listing to review", labels, key="advanced_review_listing")
                selected_row = filtered.iloc[labels.index(selected_label)]
                render_listing_card(selected_row)
                render_review_form(selected_row)

        with st.expander("AI photo scoring", expanded=False):
            ai_action_cols = st.columns([1, 2])
            with ai_action_cols[0]:
                if st.button("Score photos for top visible listings", disabled=not openai_ready() or filtered.empty):
                    if st.session_state.get("ai_photo_quota_blocked"):
                        st.warning("AI photo scoring is paused after a quota/billing error. Use Reset AI photo pause in the sidebar after fixing billing.")
                    else:
                        with st.spinner(f"Scoring photos for up to {auto_photo_limit} visible listing(s)..."):
                            scored_count, photo_messages = auto_score_photo_candidates(filtered.head(20), limit=auto_photo_limit)
                        if photo_messages:
                            st.info("\n".join(photo_messages[-5:]))
                        if scored_count:
                            st.success("AI photo scores saved. Refreshing rankings with image scores...")
                            st.rerun()
            with ai_action_cols[1]:
                if not openai_ready():
                    st.caption("AI photo scoring needs `OPENAI_API_KEY`. The app still ranks using listing data and manual review fields.")
                else:
                    pending_visible = int(filtered.head(20).apply(needs_ai_photo_score, axis=1).sum()) if not filtered.empty else 0
                    st.caption(f"{pending_visible} of the top visible listings still need image scores.")

        with st.expander("Scoring method and export", expanded=False):
            st.markdown(
                f"""
                `match_score = weighted_average(school, quiet, price/value, size, yard/layout/location) + school_confidence_bonus + upcoming_open_house_bonus`

                Current weights: school `{school_importance}`, quiet `{quiet_importance}`, price/value `{price_importance}`, size `{size_importance}`, yard/layout/location `{lifestyle_importance}`.
                Manual review fields and AI photo scores can update yard/layout/quiet components. BC Assessment values are manual until entered.
                """
            )
            export_bytes = export_excel(prefs, display_columns(top), display_columns(filtered), display_columns(excluded))
            st.download_button(
                "Download ranked Excel",
                data=export_bytes,
                file_name=f"family_home_advisor_ranked_{datetime.now():%Y%m%d_%H%M}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.markdown("##### All listings currently shown")
            st.dataframe(display_columns(filtered), use_container_width=True, hide_index=True)
            st.markdown("##### Excluded homes")
            if not excluded.empty:
                st.dataframe(display_columns(excluded.assign(explanation=excluded["excluded_reason"])), use_container_width=True, hide_index=True)
            else:
                st.info("No homes excluded by hard filters.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error("The app hit an error while loading. The details below are shown so we can fix the deployed version quickly.")
        st.exception(exc)
