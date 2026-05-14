from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
from scoring import display_columns, export_excel, filter_by_preferences, marker_color, score_listings

try:
    import folium
except ImportError:  # pragma: no cover - handled in Streamlit UI
    folium = None


APP_TITLE = "Family Home Advisor"
ROOT = Path(__file__).resolve().parent


def get_app_mode() -> str:
    env_mode = os.getenv("APP_MODE")
    if env_mode:
        return env_mode.lower()
    try:
        return str(st.secrets.get("APP_MODE", "demo")).lower()
    except Exception:
        return "demo"


APP_MODE = get_app_mode()
DEMO_MODE = True


@st.cache_data(show_spinner=False)
def load_excel(path_text: str, modified_time: float) -> tuple[pd.DataFrame, str, list[str]]:
    path = Path(path_text)
    sheet = choose_listing_sheet(path)
    df = pd.read_excel(path, sheet_name=sheet)
    return df, sheet, list(df.columns)


def make_map(df: pd.DataFrame) -> str:
    if folium is None:
        return ""

    mapped = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if mapped.empty:
        return ""

    fmap = folium.Map(location=[mapped["Latitude"].mean(), mapped["Longitude"].mean()], zoom_start=12, tiles="OpenStreetMap")
    for _, row in mapped.iterrows():
        popup_html = f"""
        <b>{row.get('Address', '')}</b><br>
        Price: {money(row.get('price_numeric'))}<br>
        School: {row.get('final_school', 'Unknown')}<br>
        Fraser score: {row.get('final_fraser_score', 'N/A')}<br>
        Noise risk: {row.get('noise_risk', 'Unknown')} {'(estimated)' if row.get('noise_estimated') else ''}<br>
        Open house: {row.get('open_house_status', 'None')}<br>
        Why: {row.get('explanation', '')}<br>
        <a href="{row.get('Listing URL', '')}" target="_blank">Listing URL</a><br>
        <a href="{row.get('Google Maps Link', '')}" target="_blank">Google Maps</a>
        """
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            tooltip=f"{row.get('Address', '')} - {row.get('match_score', 0)}",
            popup=folium.Popup(popup_html, max_width=380),
            icon=folium.Icon(color=marker_color(row)),
        ).add_to(fmap)

    return fmap._repr_html_()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

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

    with st.sidebar.expander("Preset Deal-Breakers", expanded=False):
        for deal_breaker in profile_defaults["deal_breakers"]:
            st.write(f"- {deal_breaker}")

    source_path = st.sidebar.text_input("Listing workbook", value=str(default_path))

    try:
        source_mtime = Path(source_path).stat().st_mtime
        raw_df, sheet, original_columns = load_excel(source_path, source_mtime)
    except Exception as exc:
        st.error(f"Could not load workbook: {exc}")
        return

    df, warnings = normalize_columns(raw_df)
    df = add_open_house_columns(df)
    df = add_noise_columns(df)

    max_observed_price = int(pd.to_numeric(df["price_numeric"], errors="coerce").max(skipna=True) or 3000000)
    default_max = min(max_observed_price, 3000000)

    widget_prefix = profile_name.lower().replace(" ", "_")

    max_price = st.sidebar.number_input(
        "Max price",
        min_value=0,
        value=default_max,
        step=50000,
        format="%d",
        key=f"{widget_prefix}_max_price",
    )
    min_bedrooms = st.sidebar.number_input(
        "Minimum bedrooms",
        min_value=0,
        value=profile_defaults["min_bedrooms"],
        step=1,
        key=f"{widget_prefix}_min_bedrooms",
    )
    min_fraser_score = st.sidebar.slider(
        "Minimum Fraser school score",
        0.0,
        10.0,
        float(profile_defaults["min_fraser_score"]),
        0.1,
        key=f"{widget_prefix}_min_fraser_score",
    )
    quiet_importance = st.sidebar.slider(
        "Quiet-location importance", 1, 5, profile_defaults["quiet_importance"], key=f"{widget_prefix}_quiet_importance"
    )
    school_importance = st.sidebar.slider(
        "School importance", 1, 5, profile_defaults["school_importance"], key=f"{widget_prefix}_school_importance"
    )
    price_importance = st.sidebar.slider(
        "Price/value importance", 1, 5, profile_defaults["price_importance"], key=f"{widget_prefix}_price_importance"
    )
    size_importance = st.sidebar.slider(
        "Size importance", 1, 5, profile_defaults["size_importance"], key=f"{widget_prefix}_size_importance"
    )
    lifestyle_importance = st.sidebar.slider(
        "Rancher/backyard/helper/layout importance",
        1,
        5,
        profile_defaults["lifestyle_importance"],
        key=f"{widget_prefix}_lifestyle_importance",
    )
    preferred_city = st.sidebar.selectbox(
        "Preferred city",
        ["Both", "North Vancouver", "West Vancouver"],
        index=["Both", "North Vancouver", "West Vancouver"].index(profile_defaults["preferred_city"]),
        key=f"{widget_prefix}_preferred_city",
    )
    exclude_high_noise = st.sidebar.checkbox(
        "Exclude high-noise homes",
        value=profile_defaults["exclude_high_noise"],
        key=f"{widget_prefix}_exclude_high_noise",
    )

    st.sidebar.header("Optional Profile Helper")
    buyer_text = st.sidebar.text_area("Describe what this buyer cares about", height=100)
    if buyer_text.strip():
        suggestions = parse_buyer_profile(buyer_text)
        st.sidebar.info(
            "Suggested weights: "
            f"quiet {suggestions['quiet_importance']}, "
            f"school {suggestions['school_importance']}, "
            f"price {suggestions['price_importance']}, "
            f"size {suggestions['size_importance']}, "
            f"lifestyle {suggestions['lifestyle_importance']}. "
            f"Deal-breakers: {', '.join(suggestions['deal_breakers']) or 'none detected'}."
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
    }

    scored = score_listings(df, prefs)
    filtered, excluded = filter_by_preferences(scored, prefs)
    top = filtered.head(20)

    st.caption(f"Loaded `{Path(source_path).name}` sheet `{sheet}` with {len(raw_df)} listings.")
    with st.expander("Input Files And Columns", expanded=False):
        st.write("Default input order:", DEFAULT_INPUT_CANDIDATES)
        st.write("Columns found:", original_columns)
        for warning in warnings:
            st.warning(warning)
        if df["noise_estimated"].any():
            st.warning("Some noise risk values are estimated because road GIS/proximity fields were missing.")

    st.subheader("Scoring Method")
    st.markdown(
        f"""
        `match_score = weighted_average(school, quiet, price/value, size, rancher/backyard/helper/layout) + school_confidence_bonus + upcoming_open_house_bonus`

        Active profile: `{profile_name}`. Current weights: school `{school_importance}`, quiet `{quiet_importance}`, price/value `{price_importance}`, size `{size_importance}`, lifestyle `{lifestyle_importance}`.
        Open house bonus is `+3` only for `Upcoming` open houses. Noise penalties are Low `0`, Medium `18`, High `40`, Unknown `12`.
        """
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Filtered matches", len(filtered))
    metric_cols[1].metric("Excluded homes", len(excluded))
    metric_cols[2].metric("Upcoming open houses", int(filtered["open_house_status"].eq("Upcoming").sum()))
    metric_cols[3].metric("High-noise homes", int(scored["noise_risk"].eq("High").sum()))

    st.subheader("Interactive Map")
    html = make_map(filtered)
    if html:
        components.html(html, height=620)
    elif folium is None:
        st.warning("Install `folium` to show the interactive map.")
    else:
        st.warning("No listings with latitude/longitude are available for the map.")

    st.subheader("Top Matches Based on Buyer Profile")
    st.dataframe(display_columns(top), use_container_width=True, hide_index=True)

    export_bytes = export_excel(prefs, display_columns(top), display_columns(filtered), display_columns(excluded))
    st.download_button(
        "Download ranked Excel",
        data=export_bytes,
        file_name=f"family_home_advisor_ranked_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("Excluded Homes", expanded=False):
        st.dataframe(display_columns(excluded.assign(explanation=excluded["excluded_reason"])), use_container_width=True)


if __name__ == "__main__":
    main()
