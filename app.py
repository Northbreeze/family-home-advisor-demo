from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Family Home Advisor", layout="wide", initial_sidebar_state="expanded")

from area_filters import AREA_KEYWORDS
from ai_layer import configure_openai_key
from data_layer import default_data_paths, load_listing_data, source_path
from scoring_layer import base_prefs, filter_ranked_data, metrics, score_family_fit, sort_visible
from ui_layer import (
    app_map,
    clicked_address,
    inject_css,
    render_floating_chat,
    render_listing_grid,
    render_map_header,
    render_profile_chips,
    render_search_header,
    render_selected_home_review,
    render_sidebar_profile,
    selected_row,
    set_selected_home,
)
from v2_product import load_family_profile, load_listing_events

try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium = None

APP_TITLE = "Family Home Advisor"
ROOT = Path(__file__).resolve().parent
DATA_PATHS = default_data_paths(ROOT)
FAMILY_PROFILE_PATH = ROOT / "family_profile.json"
LISTING_EVENTS_PATH = ROOT / "listing_events.csv"

configure_openai_key(ROOT, st.secrets)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def cached_load_listing_data(root_text: str, source_mtime: float, reviews_mtime: float, photo_reviews_mtime: float, change_log_mtime: float):
    return load_listing_data(default_data_paths(Path(root_text)))


@st.cache_data(show_spinner=False)
def cached_score_family_fit(data: pd.DataFrame, prefs_json: str, chips_tuple: tuple[str, ...]) -> pd.DataFrame:
    return score_family_fit(data, json.loads(prefs_json), list(chips_tuple))


def inferred_chips(school_importance: int, yard_importance: int, quiet_importance: int) -> list[str]:
    chips: list[str] = []
    if quiet_importance >= 4:
        chips.extend(["Quiet street", "Avoid busy roads"])
    if yard_importance >= 4:
        chips.extend(["Good for toddlers", "Large backyard"])
    if school_importance >= 4:
        chips.append("Strong resale")
    return list(dict.fromkeys(chips))


def main() -> None:
    inject_css()
    try:
        source = source_path(ROOT)
        df, path, warnings = cached_load_listing_data(
            str(ROOT),
            file_mtime(source),
            file_mtime(DATA_PATHS.reviews_path),
            file_mtime(DATA_PATHS.photo_reviews_path),
            file_mtime(DATA_PATHS.listing_change_log_path),
        )
    except Exception as exc:
        st.error("Could not load listings.")
        st.exception(exc)
        return

    events = load_listing_events(LISTING_EVENTS_PATH)
    saved = set(events.loc[events["event_type"].astype(str).eq("Saved"), "address"].astype(str)) if not events.empty else set()
    profile = load_family_profile(FAMILY_PROFILE_PATH)

    prices = pd.to_numeric(df["price_numeric"], errors="coerce")
    min_price = int(max(0, prices.min(skipna=True) // 50000 * 50000)) if prices.notna().any() else 0
    max_price = int((prices.max(skipna=True) // 50000 + 1) * 50000) if prices.notna().any() else 3500000
    default_high = min(max_price, max(2500000, int(prices.quantile(0.85)) if prices.notna().any() else 2500000))

    st.sidebar.markdown("# Family Home Advisor")
    st.sidebar.caption("AI buyer's agent for North and West Vancouver family homes.")
    st.sidebar.markdown("### Filters")
    price_range = st.sidebar.slider("Budget", min_price, max_price, (min_price, default_high), step=50000, format="$%d", key="filter_price_range")
    bed_choice = st.sidebar.radio("Bedrooms", ["Any", "2+", "3+", "4+", "5+"], index=2, horizontal=True, key="filter_bedrooms")
    min_beds = 0 if bed_choice == "Any" else int(bed_choice.replace("+", ""))
    st.sidebar.markdown("**Location**")
    north_selected = st.sidebar.checkbox("North Vancouver", value=True, key="filter_city_north")
    west_selected = st.sidebar.checkbox("West Vancouver", value=True, key="filter_city_west")
    area_choice = st.sidebar.selectbox("Neighbourhood / village", ["All"] + sorted(AREA_KEYWORDS.keys()), key="filter_area_choice")
    locations: list[str] = []
    if north_selected and not west_selected:
        locations.append("North Vancouver")
    elif west_selected and not north_selected:
        locations.append("West Vancouver")
    elif not north_selected and not west_selected:
        locations.append("__NO_CITY_SELECTED__")
    if area_choice != "All":
        locations.append(area_choice)
    school_importance = st.sidebar.slider("School importance", 1, 5, 4, key="school_importance")
    yard_importance = st.sidebar.slider("Yard importance", 1, 5, 5, key="yard_importance")
    quiet_importance = st.sidebar.slider("Quiet street importance", 1, 5, 5, key="quiet_importance")
    recommended_first = st.sidebar.checkbox("Show recommended homes first", value=True, key="filter_recommended_first")
    saved_only = st.sidebar.checkbox("Saved homes only", value=st.session_state.get("saved_only", False), key="saved_only")

    chips = inferred_chips(school_importance, yard_importance, quiet_importance)
    st.session_state["active_preference_chips"] = chips
    render_sidebar_profile(profile, chips)

    with st.sidebar.expander("More filters", expanded=False):
        open_house_only = st.checkbox("Open houses only", value=False, key="filter_open_house_only")
        new_only = st.checkbox("Changed/new since last refresh", value=False, key="filter_new_only")
        avoid_high_noise = st.checkbox("Exclude high-noise homes", value=False, key="filter_avoid_high_noise")
        yard_filter = st.selectbox("Yard filter", ["Any", "Usable yard", "Large yard signal", "Exclude known poor yard"], key="filter_yard_size")
        assessment_only = st.checkbox("Only homes with BC Assessment entered", value=False, key="filter_assessment_only")
        st.caption(f"Data source: {path.name}")
        if warnings:
            st.caption("Some missing columns were handled automatically.")

    search = render_search_header(len(saved))

    rules = st.session_state.get("ai_rules", {})
    max_for_score = int(rules.get("max_price", price_range[1]))
    min_school = 8.0 if school_importance >= 5 else 7.0 if school_importance >= 4 else 0.0
    if rules.get("min_school"):
        min_school = max(min_school, float(rules["min_school"]))
    if rules.get("avoid_noise"):
        avoid_high_noise = True
        if "Avoid busy roads" not in chips:
            chips.append("Avoid busy roads")
    if rules.get("yard_focus") and "Good for toddlers" not in chips:
        chips.append("Good for toddlers")

    prefs = base_prefs(max_for_score, min_beds, min_school, chips)
    prefs["school_importance"] = school_importance
    prefs["quiet_importance"] = quiet_importance
    prefs["lifestyle_importance"] = yard_importance
    prefs["size_importance"] = max(3, yard_importance)

    scored = cached_score_family_fit(df, json.dumps(prefs, sort_keys=True, default=str), tuple(chips))
    more = {"open_house_only": open_house_only, "new_only": new_only, "avoid_high_noise": avoid_high_noise, "assessment_only": assessment_only}
    visible = filter_ranked_data(scored, search, (price_range[0], min(price_range[1], max_for_score)), locations, min_beds, min_school, yard_filter, more)
    if saved_only and saved:
        visible = visible[visible["Address"].astype(str).isin(saved)]
    visible = sort_visible(visible, recommended_first)

    counts = metrics(scored, visible)
    render_profile_chips(profile, chips, counts)

    render_map_header(counts)
    fmap = app_map(visible)
    if fmap is None:
        st.info("No listings with map coordinates match the current filters.")
    elif st_folium is None:
        st.components.v1.html(fmap._repr_html_(), height=500)
    else:
        map_data = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_object_clicked"], key="consumer_map")
        picked = clicked_address(visible, (map_data or {}).get("last_object_clicked"))
        if picked and picked != st.session_state.get("selected_address"):
            match = visible[visible["Address"].astype(str).str.strip().eq(str(picked).strip())]
            if not match.empty:
                set_selected_home(match.iloc[0])

    selected_pool = scored
    selected = selected_row(selected_pool)
    render_selected_home_review(selected, selected_pool, profile)
    render_listing_grid(visible, profile, limit=12)
    render_floating_chat(selected_pool, selected)


if __name__ == "__main__":
    main()
