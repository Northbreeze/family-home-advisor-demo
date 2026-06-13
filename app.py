from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from area_filters import AREA_KEYWORDS
from ai_layer import configure_openai_key
from data_layer import default_data_paths, load_listing_data
from scoring_layer import base_prefs, filter_ranked_data, metrics, score_family_fit, sort_visible
from ui_layer import (
    app_map,
    clicked_address,
    compact_home_review,
    inject_css,
    map_toolbar,
    render_chat_panel,
    render_family_profile_step,
    render_flow_header,
    render_ranked_homes_step,
    selected_row,
    top_header,
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

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
configure_openai_key(ROOT, st.secrets)


def main() -> None:
    inject_css()
    try:
        df, path, warnings = load_listing_data(DATA_PATHS)
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
    scored = score_family_fit(df, prefs, chips)
    more = {"open_house_only": open_house_only, "new_only": new_only, "avoid_high_noise": avoid_high_noise, "assessment_only": assessment_only}
    visible = filter_ranked_data(scored, search, (price_range[0], min(price_range[1], max_for_score)), locations, min_beds, min_school, yard, more)
    if st.session_state.get("saved_only") and saved:
        visible = visible[visible["Address"].astype(str).isin(saved)]
    visible = sort_visible(visible, recommended_first)

    counts = metrics(scored, visible)
    selected_pool = visible if not visible.empty else scored

    render_family_profile_step(profile, chips, price_range, locations, min_beds, school_choice, yard, counts)
    st.markdown(
        "<div class='desktop-note'>Desktop decision workspace: map, selected home review, and AI chat stay visible together so you can choose the next home to tour.</div>",
        unsafe_allow_html=True,
    )

    map_col, review_col, chat_col = st.columns([0.50, 0.27, 0.23], gap="large")
    with map_col:
        render_flow_header("2", "Ranked Homes", "Use the map and top matches to choose what to inspect next.")
        map_toolbar(counts)
        fmap = app_map(visible)
        if fmap is None:
            st.info("No listings with map coordinates match the current filters.")
        elif st_folium is None:
            st.components.v1.html(fmap._repr_html_(), height=620)
        else:
            map_data = st_folium(fmap, height=620, use_container_width=True, returned_objects=["last_object_clicked"], key="consumer_map")
            picked = clicked_address(visible, (map_data or {}).get("last_object_clicked"))
            if picked:
                st.session_state["selected_address"] = picked
        render_ranked_homes_step(visible, profile)

    selected = selected_row(selected_pool)
    with review_col:
        render_flow_header("3", "Home Review", "Condensed decision view for the selected home.")
        if selected is not None:
            compact_home_review(selected, selected_pool, profile)
        else:
            st.info("Select a home from the map or ranked list to see the review.")

    with chat_col:
        render_flow_header("4", "Ask AI", "Persistent buyer-agent chat for trade-offs and tour decisions.")
        render_chat_panel(selected_pool)


if __name__ == "__main__":
    main()
