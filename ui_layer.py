from __future__ import annotations

from html import escape
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from ai_layer import answer_chat, answer_home_question, evidence_dataframe, facts_dataframe, get_cached_or_fallback_listing_evaluation, get_listing_evaluation
from data_cleaning import money
from scoring_layer import category, family_concern_items, interior_size_summary, num, score, sort_visible, tone
from v2_product import append_listing_event, card_title, listing_links, why_it_may_work

try:
    import folium
except ImportError:
    folium = None

ROOT = Path(__file__).resolve().parent
LISTING_EVENTS_PATH = ROOT / "listing_events.csv"
HOME_EVALUATION_CACHE_PATH = ROOT / "home_evaluation_cache.json"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {background:#f7faf8; color:#0f172a;}
        .block-container {padding-top:1rem; padding-bottom:.75rem; max-width:1760px;}
        header[data-testid="stHeader"] {background:rgba(247,250,248,.9);}
        section[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #e5ece7; box-shadow:8px 0 24px rgba(15,45,30,.035);}
        section[data-testid="stSidebar"] label {font-weight:760; color:#111827;}
        div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px!important; border-color:#e5ece7!important; box-shadow:0 8px 24px rgba(18,52,34,.06);}
        .app-title {font-size:28px;font-weight:900;color:#0f172a;margin:0;line-height:1.05;}
        .app-subtitle {font-size:13px;color:#64748b;margin-top:4px;}
        .main-shell-note {font-size:13px;color:#64748b;margin:4px 0 14px;}
        .profile-chip-row {display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 14px;}
        .profile-chip {display:inline-flex;border:1px solid #dbe8df;background:#ffffff;color:#17633d;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:760;box-shadow:0 3px 12px rgba(18,52,34,.04);}
        .section-head {display:flex;align-items:flex-end;justify-content:space-between;margin:16px 0 8px;gap:12px;}
        .section-title {font-size:20px;font-weight:900;color:#0f172a;}
        .section-sub {font-size:13px;color:#64748b;}
        .map-toolbar {display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 8px;}
        .count-pill {display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid #e5ece7;border-radius:999px;padding:8px 12px;font-weight:850;font-size:13px;}
        .count-dot {width:8px;height:8px;border-radius:50%;background:#07915a;display:inline-block;}
        .fit-badge {display:inline-flex;border-radius:999px;font-size:12px;font-weight:900;padding:5px 10px;white-space:normal;line-height:1.15;}
        .green {background:#e7f6ec;color:#17633d}.yellow {background:#fff7dc;color:#765800}.red {background:#fff0ee;color:#9f351f}
        .score-pill {display:inline-grid;place-items:center;min-width:52px;height:34px;border-radius:11px;color:white;font-weight:950;font-size:15px;}
        .score-pill.green-bg {background:#07915a}.score-pill.yellow-bg {background:#f3b700}.score-pill.red-bg {background:#ff6b5f}
        .listing-title {font-weight:900;font-size:14px;line-height:1.2;color:#0f172a;margin-bottom:4px;}
        .listing-meta {font-size:12px;color:#475569;line-height:1.25;margin-bottom:6px;}
        .reason {font-size:12px;color:#27362e;margin:3px 0;line-height:1.28;}
        .reason:before {content:'+ ';color:#07915a;font-weight:950;}
        .concern {font-size:12px;color:#86451d;margin:3px 0 7px;line-height:1.28;}
        .concern:before {content:'! ';color:#d99000;font-weight:950;}
        .review-hero {display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:12px;}
        .review-title {font-size:24px;font-weight:950;line-height:1.1;color:#0f172a;margin:0 0 5px;}
        .review-meta {font-size:14px;color:#475569;line-height:1.4;}
        .review-card-title {font-size:12px;color:#64748b;font-weight:900;text-transform:uppercase;letter-spacing:.02em;margin-bottom:6px;}
        .review-line {font-size:14px;color:#27362e;margin:6px 0;line-height:1.4;}
        .review-line.good:before {content:'+ ';color:#07915a;font-weight:950;}
        .review-line.risk:before {content:'! ';color:#d99000;font-weight:950;}
        .final-rec {border-radius:14px;background:#edf8f1;border:1px solid #cfeadb;color:#17633d;font-weight:900;padding:12px 14px;margin-top:12px;}
        .neutral-photo {display:none;}
        .floating-chat-spacer {height:0;}
        div[data-testid="stPopover"] {position:fixed!important;right:24px!important;bottom:24px!important;z-index:9999!important;}
        div[data-testid="stPopover"] > button {background:#07915a!important;color:#fff!important;border-radius:999px!important;padding:0.75rem 1.15rem!important;font-weight:900!important;box-shadow:0 14px 34px rgba(7,145,90,.28)!important;border:0!important;}
        .review-grid {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;}
        .insight-card {background:#fff;border:1px solid #e5ece7;border-radius:14px;padding:13px 14px;box-shadow:0 7px 18px rgba(18,52,34,.05);}
        .insight-card.wide {grid-column:1 / -1;}
        .insight-card p {margin:.1rem 0 0;color:#27362e;font-size:14px;line-height:1.45;}
        .insight-list {margin:0;padding-left:0;list-style:none;}
        .insight-list li {margin:6px 0;font-size:13px;line-height:1.35;color:#27362e;}
        .insight-list.good li:before {content:'+ ';color:#07915a;font-weight:950;}
        .insight-list.risk li:before {content:'! ';color:#d99000;font-weight:950;}
        @media(max-width:900px){.review-grid{grid-template-columns:1fr}}
        .chat-box {border:1px solid #cfeadb;border-radius:16px;background:#ffffff;padding:12px;margin-bottom:10px;}
        .chat-title {font-weight:900;color:#0f172a;margin-bottom:6px;}
        .chat-bubble {background:#f8fafc;border:1px solid #edf2ef;border-radius:12px;padding:10px 11px;color:#334155;font-size:13px;margin-bottom:8px;}
        .prompt-row {display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;}
        .prompt-pill {border:1px solid #e5ece7;border-radius:999px;padding:6px 9px;font-size:11px;color:#334155;background:#fff;}
        div.stButton>button {border-radius:11px;font-weight:850;border:1px solid #dfe8e2;background:white;}
        div.stButton>button:hover {border-color:#07915a;color:#07915a;}
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {background:#e7f6ec;color:#17633d;border-radius:999px;}
        @media(max-width:1100px){.review-hero{flex-direction:column}.app-title{font-size:24px}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def title(row: pd.Series) -> str:
    return card_title(row)


def slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).upper()).strip("_")[:80]


def fit_label(row: pd.Series) -> str:
    return category(row)


def fit_badge(row: pd.Series) -> None:
    st.markdown(f"<span class='fit-badge {tone(row)}'>{escape(fit_label(row))}</span>", unsafe_allow_html=True)


def score_badge(row: pd.Series) -> str:
    return f"<span class='score-pill {tone(row)}-bg'>{score(row)}</span>"


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


def small_photo(row: pd.Series) -> None:
    path = photo_path(row)
    if path and path.exists():
        st.image(str(path), width=120)



def signal_key(value: object) -> str:
    text_value = re.sub(r"\s+", " ", str(value).lower()).strip()
    if "yard" in text_value and any(word in text_value for word in ["unknown", "verify", "photo", "showing"]):
        return "yard_verify"
    if "price" in text_value and any(word in text_value for word in ["comfort", "budget", "within"]):
        return "price_fit"
    if any(word in text_value for word in ["noise", "highway", "road"]):
        return "noise"
    if any(word in text_value for word in ["interior", "sqft", "size", "layout"]):
        return "interior"
    if "school" in text_value or "catchment" in text_value:
        return "school"
    return text_value[:80]


def unique_signals(items: list[Any], limit: int = 4, skip_keys: set[str] | None = None) -> list[str]:
    skip_keys = skip_keys or set()
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text_value = str(item).strip()
        if not text_value or text_value.lower() in {"nan", "none"}:
            continue
        key = signal_key(text_value)
        if key in skip_keys or key in seen:
            continue
        seen.add(key)
        result.append(text_value)
        if len(result) >= limit:
            break
    return result


def card_strength(row: pd.Series, profile: dict[str, Any]) -> str:
    reasons = why_it_may_work(row, profile, limit=6)
    non_price = unique_signals(reasons, limit=1, skip_keys={"price_fit"})
    if non_price:
        return non_price[0]
    return unique_signals(reasons, limit=1)[0] if reasons else "Good candidate for review."


def card_concern(row: pd.Series, profile: dict[str, Any]) -> str:
    risks = family_concern_items(row, profile, limit=6)
    unique = unique_signals(risks, limit=1)
    return unique[0] if unique else "Verify photos and showing fit."


def html_list(items: list[str], klass: str = "") -> str:
    return "".join(f"<li>{escape(str(item))}</li>" for item in items)


def set_selected_home(row: pd.Series) -> None:
    address = str(row.get("Address", "")).strip()
    st.session_state["selected_address"] = address
    st.session_state["selected_home"] = {
        "Address": address,
        "Listing URL": str(row.get("Listing URL", "")),
        "match_score": score(row),
    }
    st.session_state["scroll_to_review"] = True


def scroll_to_review_if_needed() -> None:
    if not st.session_state.pop("scroll_to_review", False):
        return
    st.components.v1.html(
        """
        <script>
        const el = window.parent.document.getElementById('selected-home-review');
        if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        </script>
        """,
        height=0,
    )

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
        reason = card_strength(row, {})
        risk = card_concern(row, {})
        html = f"""
        <div style='font-family:Arial,sans-serif;width:250px;padding:6px;'>
          <div style='font-weight:800;font-size:15px;margin-bottom:5px;color:#0f172a;'>{escape(title(row))}</div>
          <div style='color:#17633d;font-weight:800;margin-bottom:6px;'>{money(row.get('price_numeric', row.get('Price')))} | {score(row)}/100</div>
          <div style='font-size:12px;color:#34443b;margin-bottom:4px;'>+ {escape(reason)}</div>
          <div style='font-size:12px;color:#86451d;margin-bottom:9px;'>! {escape(risk)}</div>
          <div style='display:inline-block;background:#07915a;color:white;border-radius:999px;padding:7px 11px;font-weight:800;font-size:12px;'>View Review</div>
        </div>
        """
        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            tooltip=f"{title(row)} - {score(row)}/100",
            popup=folium.Popup(html, max_width=290),
            icon=folium.DivIcon(html=marker_html(row), icon_size=(42, 42), icon_anchor=(21, 21)),
        ).add_to(fmap)
    legend_html = """
    <div style="position: fixed; left: 18px; bottom: 26px; z-index: 9999; background: white; border: 1px solid #e5ece7; border-radius: 14px; padding: 14px 16px; box-shadow: 0 10px 26px rgba(18,52,34,.16); font-family: Arial, sans-serif; min-width: 230px;">
      <div style="font-weight: 800; margin-bottom: 10px; color: #0f172a;">Family Fit Score</div>
      <div style="display:flex; align-items:center; gap:9px; margin:7px 0; font-size:12px;"><span style="width:11px;height:11px;border-radius:50%;background:#07915a;display:inline-block;"></span>80 - 100&nbsp;&nbsp; Strong family fit</div>
      <div style="display:flex; align-items:center; gap:9px; margin:7px 0; font-size:12px;"><span style="width:11px;height:11px;border-radius:50%;background:#f3b700;display:inline-block;"></span>60 - 79&nbsp;&nbsp; Good fit - verify</div>
      <div style="display:flex; align-items:center; gap:9px; margin:7px 0; font-size:12px;"><span style="width:11px;height:11px;border-radius:50%;background:#ff6b5f;display:inline-block;"></span>0 - 59&nbsp;&nbsp; Possible fit - verify</div>
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


def render_search_header(saved_count: int) -> str:
    st.markdown("<div class='app-title'>Family Home Advisor</div><div class='app-subtitle'>AI buyer's agent for family homes</div>", unsafe_allow_html=True)
    return st.text_input("Search", placeholder="Search by address, neighbourhood, or school", label_visibility="collapsed", key="main_search")


def compact_items(values: Any, limit: int = 6) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        items = [values]
    else:
        items = [str(item) for item in values if str(item).strip()]
    clean = [item.strip() for item in items if item.strip() and item.strip().lower() not in {"nan", "none"}]
    return clean[:limit]


def render_profile_chips(profile: dict[str, Any], chips: list[str], counts: dict[str, int]) -> None:
    values = [f"{counts['visible']} homes shown", f"{counts['all']} active listings"]
    values.extend(compact_items(profile.get("preferred_neighbourhoods"), 4))
    values.extend(chips[:4])
    chips_html = "".join(f"<span class='profile-chip'>{escape(value)}</span>" for value in values)
    st.markdown(f"<div class='profile-chip-row'>{chips_html}</div>", unsafe_allow_html=True)


def render_sidebar_profile(profile: dict[str, Any], chips: list[str]) -> None:
    st.sidebar.markdown("### Family Profile")
    st.sidebar.caption("What the ranking is trying to optimize.")
    for value in compact_items(profile.get("important_preferences"), 5):
        st.sidebar.markdown(f"- {value}")
    if chips:
        st.sidebar.caption("Active AI preferences: " + ", ".join(chips[:5]))


def render_map_header(counts: dict[str, int]) -> None:
    st.markdown(
        f"""
        <div class='map-toolbar'>
          <div><div class='section-title'>Homes Map</div><div class='section-sub'>Click a pin or card to open the home review below.</div></div>
          <div class='count-pill'><span class='count-dot'></span>{counts['visible']} homes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def listing_card(row: pd.Series, profile: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        small_photo(row)
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:10px;'>"
            f"<div class='listing-title'>{escape(title(row))}</div>{score_badge(row)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='listing-meta'>{escape(money(row.get('price_numeric', row.get('Price'))))} | "
            f"{escape(str(row.get('Bedrooms', '')))} bd | {escape(str(row.get('Bathrooms', '')))} ba | {escape(str(row.get('Size', '')))}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='reason'>{escape(card_strength(row, profile))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='concern'>{escape(card_concern(row, profile))}</div>", unsafe_allow_html=True)
        actions = st.columns(2, gap="small")
        with actions[0]:
            if st.button("View Review", key=f"view_{key}", use_container_width=True):
                set_selected_home(row)
                st.rerun()
        with actions[1]:
            if st.button("Save", key=f"save_{key}", use_container_width=True):
                saved = append_listing_event(LISTING_EVENTS_PATH, "Saved", str(row.get("Address", "")))
                if saved:
                    st.toast("Saved home")
                else:
                    st.warning("Could not save this home because the local saved-homes file is locked.")

def render_listing_grid(visible: pd.DataFrame, profile: dict[str, Any], limit: int = 12) -> None:
    st.markdown("<div class='section-head'><div><div class='section-title'>Ranked Homes</div><div class='section-sub'>Best next-tour candidates based on the family profile.</div></div></div>", unsafe_allow_html=True)
    if visible.empty:
        st.info("No homes match the current filters. Try broadening the budget or location.")
        return
    cols = st.columns(3, gap="medium")
    for i, (_, row) in enumerate(visible.head(limit).iterrows()):
        with cols[i % 3]:
            listing_card(row, profile, f"card_{i}_{slug(row.get('Address', ''))}")


def selected_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    selected = st.session_state.get("selected_address") or st.session_state.get("selected_home", {}).get("Address")
    if selected:
        selected_text = str(selected).strip()
        match = df[df["Address"].astype(str).str.strip().eq(selected_text)]
        if not match.empty:
            return match.iloc[0]
    return None


def bullets(items: list[Any], klass: str = "") -> None:
    for item in items:
        st.markdown(f"<div class='review-line {klass}'>{escape(str(item))}</div>", unsafe_allow_html=True)


def listing_preferences(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "family_profile": profile,
        "active_preference_chips": st.session_state.get("active_preference_chips", []),
        "max_price": st.session_state.get("filter_price_range", [None, None])[-1] if st.session_state.get("filter_price_range") else None,
        "min_bedrooms": st.session_state.get("filter_bedrooms", "3+"),
        "school_importance": st.session_state.get("school_importance", 3),
        "yard_importance": st.session_state.get("yard_importance", 4),
        "quiet_importance": st.session_state.get("quiet_importance", 5),
    }


def render_selected_home_review(row: pd.Series | None, visible: pd.DataFrame, profile: dict[str, Any]) -> None:
    st.markdown("<div id='selected-home-review'></div>", unsafe_allow_html=True)
    scroll_to_review_if_needed()
    st.markdown("<div class='section-head'><div><div class='section-title'>Selected Home Review</div><div class='section-sub'>Condensed AI buyer-agent view for deciding whether to tour.</div></div></div>", unsafe_allow_html=True)
    if row is None:
        st.info("Click a map pin or a View Review button to open a selected home review here.")
        return

    preferences = listing_preferences(profile)
    evaluation = get_cached_or_fallback_listing_evaluation(row, preferences, HOME_EVALUATION_CACHE_PATH)
    generate_key = f"generate_ai_review_{slug(row.get('Address', ''))}"
    if st.button("Generate / refresh AI review", key=generate_key):
        with st.spinner("Generating AI review for this home..."):
            evaluation = get_listing_evaluation(row, preferences, HOME_EVALUATION_CACHE_PATH, force=True, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
            st.toast("AI review updated")
    scenarios = evaluation.get("family_fit_scenarios", {}) if isinstance(evaluation.get("family_fit_scenarios"), dict) else {}
    strengths_raw = evaluation.get("strengths", []) if isinstance(evaluation.get("strengths"), list) else []
    risks_raw = evaluation.get("tradeoffs_risks", []) if isinstance(evaluation.get("tradeoffs_risks"), list) else []
    strengths = unique_signals(strengths_raw + why_it_may_work(row, profile, limit=6), limit=4, skip_keys={"price_fit"})
    if not strengths:
        strengths = unique_signals(strengths_raw + why_it_may_work(row, profile, limit=6), limit=4)
    risks = unique_signals(risks_raw + family_concern_items(row, profile, limit=6), limit=4)
    best_for = unique_signals(evaluation.get("best_for", []) if isinstance(evaluation.get("best_for"), list) else [], limit=3)
    less_ideal = unique_signals(scenarios.get("less_ideal_if", []) if isinstance(scenarios.get("less_ideal_if", []), list) else [], limit=3)
    summary = str(evaluation.get("overall_summary", row.get("explanation", "Review this home against the family profile."))).split("\n")[0]

    with st.container(border=True):
        st.markdown(
            f"""
            <div class='review-hero'>
              <div>
                <div class='review-title'>{escape(title(row))}</div>
                <div class='review-meta'>{escape(money(row.get('price_numeric', row.get('Price'))))} | {escape(str(row.get('Bedrooms', '')))} bd | {escape(str(row.get('Bathrooms', '')))} ba | {escape(str(row.get('Size', '')))}</div>
              </div>
              <div style='text-align:right'>{score_badge(row)}<br><span class='fit-badge {tone(row)}'>{escape(fit_label(row))}</span></div>
            </div>
            <div class='review-grid'>
              <div class='insight-card wide'><div class='review-card-title'>Summary</div><p>{escape(summary)}</p></div>
              <div class='insight-card'><div class='review-card-title'>Strengths</div><ul class='insight-list good'>{html_list(strengths)}</ul></div>
              <div class='insight-card'><div class='review-card-title'>Risks</div><ul class='insight-list risk'>{html_list(risks)}</ul></div>
              <div class='insight-card'><div class='review-card-title'>Best for</div><ul class='insight-list'>{html_list(best_for)}</ul></div>
              <div class='insight-card'><div class='review-card-title'>Final recommendation</div><p>{escape(str(evaluation.get('ai_recommendation', fit_label(row))))}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if less_ideal:
            st.markdown("<div class='review-card-title' style='margin-top:12px;'>Less ideal if</div>", unsafe_allow_html=True)
            bullets(less_ideal[:3], "risk")
        links = listing_links(row)
        link_parts = [f"[{name}]({url})" for name, url in links.items() if str(url).strip()]
        if link_parts:
            st.markdown(" | ".join(link_parts))

        with st.expander("Technical details", expanded=False):
            st.caption(f"Narrative source: {evaluation.get('source', 'Unknown')} | Model: {evaluation.get('model', 'none')}")
            facts = facts_dataframe(evaluation)
            evidence = evidence_dataframe(evaluation)
            if not facts.empty:
                st.dataframe(facts, use_container_width=True, hide_index=True)
            if not evidence.empty:
                st.dataframe(evidence, use_container_width=True, hide_index=True)
            st.write("Rule explanation:", row.get("explanation", "Not available"))
            st.write("Family-fit caps:", row.get("family_evaluation_note", "None"))

def render_floating_chat(visible: pd.DataFrame, selected: pd.Series | None) -> None:
    label = "Ask AI"
    with st.popover(label, use_container_width=False):
        selected_title = title(selected) if selected is not None else "the current search"
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        st.markdown("<div class='chat-title'>AI Buyer's Agent</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'>Ask about {escape(selected_title)}, compare homes, or adjust what matters.</div>", unsafe_allow_html=True)
        st.markdown("<div class='prompt-row'><span class='prompt-pill'>Which should we tour first?</span><span class='prompt-pill'>Avoid busy roads</span><span class='prompt-pill'>Best for toddler yard</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        for message in st.session_state.get("chat_messages", [])[-5:]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        question = st.text_input("Ask AI", placeholder="Ask a question...", label_visibility="collapsed", key="floating_ai_question")
        if st.button("Send", key="floating_ai_send", use_container_width=True) and question.strip():
            st.session_state.setdefault("chat_messages", []).append({"role": "user", "content": question})
            if selected is not None:
                answer = answer_home_question(question, selected, visible, st.session_state.get("ai_rules", {}))
            else:
                rules, response = answer_chat(question, visible, st.session_state.get("ai_rules", {}))
                st.session_state["ai_rules"] = rules
                answer = response.get("answer", "I saved that preference.")
            st.session_state.setdefault("chat_messages", []).append({"role": "assistant", "content": answer})
            st.rerun()
