from __future__ import annotations

from html import escape
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from ai_layer import answer_chat, answer_home_question, evidence_dataframe, facts_dataframe, get_listing_evaluation
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
        .flow-heading {display:flex;align-items:flex-start;gap:12px;margin:20px 0 10px;}
        .flow-index {width:34px;height:34px;border-radius:50%;background:#07915a;color:white;display:inline-grid;place-items:center;font-weight:900;flex:0 0 auto;}
        .flow-note {font-size:13px;color:#475569;margin-top:2px;}
        .profile-chip-row {display:flex;flex-wrap:wrap;gap:7px;margin-top:7px;}
        .profile-chip {display:inline-flex;border:1px solid #dbe8df;background:#f6fbf8;color:#17633d;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:750;}
        .profile-label {font-size:12px;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:.02em;margin-bottom:4px;}
        .profile-value {font-size:14px;color:#0f172a;font-weight:750;}
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


def reason_lines(row: pd.Series, profile: dict[str, Any], limit: int = 3) -> None:
    for reason in why_it_may_work(row, profile, limit=limit):
        st.markdown(f"<div class='reason'>+ {escape(reason)}</div>", unsafe_allow_html=True)


def recommendation_card(row: pd.Series, profile: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        st.markdown("<div class='rec-photo'>", unsafe_allow_html=True)
        photo(row)
        st.markdown("</div>", unsafe_allow_html=True)
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
        if st.button("View Details", key=f"detail_{key}", use_container_width=True):
            st.session_state["selected_address"] = str(row.get("Address", ""))
            st.rerun()
        if st.button("Save", key=f"save_{key}", use_container_width=True):
            append_listing_event(LISTING_EVENTS_PATH, "Saved", str(row.get("Address", "")))
            st.toast("Saved home")


def listing_card(row: pd.Series, profile: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        photo(row)
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:space-between;gap:8px;'>"
            f"<strong>{escape(title(row))}</strong>"
            f"<span class='score {tone(row)}-bg'>{score(row)}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{money(row.get('price_numeric', row.get('Price')))} | {row.get('Bedrooms', '')} bd | {row.get('Bathrooms', '')} ba | {row.get('Size', '')}")
        st.caption(interior_size_summary(row))
        badge(row)
        if st.button("View", key=f"view_{key}", use_container_width=True):
            st.session_state["selected_address"] = str(row.get("Address", ""))
            st.rerun()
        if st.button("Save", key=f"card_save_{key}", use_container_width=True):
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


def bullets(items: list[Any]) -> None:
    for item in items:
        st.write(f"- {item}")


def render_home_evaluation(row: pd.Series, visible: pd.DataFrame, profile: dict[str, Any]) -> None:
    key = slug(row.get("Address", ""))
    preferences = {
        "family_profile": profile,
        "active_preference_chips": st.session_state.get("filter_preference_chips", []),
        "max_price": st.session_state.get("filter_price_range", [None, None])[-1] if st.session_state.get("filter_price_range") else None,
        "min_bedrooms": st.session_state.get("filter_bedrooms", "3+"),
        "school_quality": st.session_state.get("filter_school_quality", "Any"),
        "yard_filter": st.session_state.get("filter_yard_size", "Any"),
    }
    force = st.button("Regenerate evaluation", key=f"regen_eval_{key}")
    with st.spinner("Preparing home evaluation narrative..."):
        evaluation = get_listing_evaluation(
            row,
            preferences,
            HOME_EVALUATION_CACHE_PATH,
            force=force,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )

    st.markdown("### Home Evaluation Narrative")
    st.caption(f"Known facts come from Realtor.ca/Excel fields and rule scores. Narrative source: {evaluation.get('source', 'Unknown')} | Model: {evaluation.get('model', 'none')}")
    with st.expander("Known facts used", expanded=False):
        facts = facts_dataframe(evaluation)
        if not facts.empty:
            st.dataframe(facts, use_container_width=True, hide_index=True)

    st.markdown(f"#### {evaluation.get('overall_verdict', 'Overall Verdict')}")
    st.metric("Rule-based score", f"{float(evaluation.get('score', score(row))):.0f}/100")
    st.write(evaluation.get("overall_summary", "No summary available."))

    st.markdown("**Best For**")
    bullets(evaluation.get("best_for", []))
    st.markdown("**Why This Home Is Strong**")
    bullets(evaluation.get("strengths", []))
    st.markdown("**Main Trade-Offs and Risks**")
    bullets(evaluation.get("tradeoffs_risks", []))

    scenarios = evaluation.get("family_fit_scenarios", {}) if isinstance(evaluation.get("family_fit_scenarios"), dict) else {}
    fit_cols = st.columns(2)
    with fit_cols[0]:
        st.markdown("**Excellent fit if...**")
        bullets(scenarios.get("excellent_fit_if", []))
    with fit_cols[1]:
        st.markdown("**Less ideal if...**")
        bullets(scenarios.get("less_ideal_if", []))

    st.markdown("**Investment and Resale View**")
    st.write(evaluation.get("investment_resale_view", "Verify comparable sales, BC Assessment, land value, and renovation potential."))
    st.markdown("**What to Verify Before Touring or Offering**")
    bullets(evaluation.get("verify_checklist", []))
    st.success(f"AI Recommendation: {evaluation.get('ai_recommendation', 'Consider but verify')}")

    with st.expander("Why this score? Raw rule-based evidence", expanded=False):
        evidence = evidence_dataframe(evaluation)
        if not evidence.empty:
            st.dataframe(evidence, use_container_width=True, hide_index=True)
        st.write("Existing rule explanation:", row.get("explanation", "Not available"))
        st.write("Family-fit caps:", row.get("family_evaluation_note", "None"))


def detail_panel(row: pd.Series, visible: pd.DataFrame, profile: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>Property Detail</div>", unsafe_allow_html=True)
        photo(row)
        badge(row)
        st.markdown(f"### {title(row)}")
        st.caption(f"{row.get('City', '')} | {money(row.get('price_numeric', row.get('Price')))} | Score {score(row)}/100")
        render_home_evaluation(row, visible, profile)

        links = listing_links(row)
        link_parts = [f"[{name}]({url})" for name, url in links.items() if str(url).strip()]
        if link_parts:
            st.markdown(" | ".join(link_parts))

        similar = visible[visible["Address"].astype(str).ne(str(row.get("Address", "")))].copy()
        area = str(row.get("detected_area", ""))
        if area and "detected_area" in similar.columns:
            same = similar[similar["detected_area"].astype(str).eq(area)]
            if not same.empty:
                similar = same
        similar = similar.sort_values("match_score", ascending=False).head(3)
        if not similar.empty:
            st.markdown("**Similar Homes**")
            for _, item in similar.iterrows():
                st.caption(f"{title(item)} | {money(item.get('price_numeric'))} | {score(item)}/100")

        question = st.text_input("Ask AI about this home", key=f"ask_{slug(row.get('Address', ''))}")
        if question:
            st.info(answer_home_question(question, row, visible, st.session_state.get("ai_rules", {})))


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


def compact_items(values: Any, limit: int = 6) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        items = [values]
    else:
        items = [str(item) for item in values if str(item).strip()]
    clean = [item.strip() for item in items if item.strip() and item.strip().lower() not in {"nan", "none"}]
    return clean[:limit]


def render_flow_header(number: str, title_text: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class='flow-heading'>
          <span class='flow-index'>{escape(str(number))}</span>
          <div>
            <div class='panel-title'>{escape(title_text)}</div>
            <div class='flow-note'>{escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_group(label: str, values: list[str]) -> None:
    chips_html = "".join(f"<span class='profile-chip'>{escape(value)}</span>" for value in values)
    if not chips_html:
        chips_html = "<span class='profile-chip'>No preference set</span>"
    st.markdown(
        f"""
        <div class='profile-label'>{escape(label)}</div>
        <div class='profile-chip-row'>{chips_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_family_profile_step(
    profile: dict[str, Any],
    chips: list[str],
    price_range: tuple[int, int],
    locations: list[str],
    min_beds: int,
    school_choice: str,
    yard: str,
    counts: dict[str, int],
) -> None:
    render_flow_header("1", "Family Profile", "These are the current family priorities used to rank homes.")
    with st.container(border=True):
        cols = st.columns([0.34, 0.33, 0.33], gap="large")
        with cols[0]:
            st.markdown("<div class='profile-label'>Search Setup</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='profile-value'>{money(price_range[0])} to {money(price_range[1])}<br>"
                f"{min_beds}+ bedrooms | School {escape(school_choice)} | Yard: {escape(yard)}</div>",
                unsafe_allow_html=True,
            )
            active_locations = locations or compact_items(profile.get("preferred_cities"), 4)
            render_chip_group("Locations", active_locations)
        with cols[1]:
            render_chip_group("Preference Chips", compact_items(chips, 7))
            render_chip_group("Important", compact_items(profile.get("important_preferences"), 4))
        with cols[2]:
            render_chip_group("Deal Breakers", compact_items(profile.get("deal_breakers"), 5))
            st.markdown(
                f"<div class='profile-label'>Current Results</div><div class='profile-value'>{counts['visible']} homes visible from {counts['all']} active homes</div>",
                unsafe_allow_html=True,
            )


def render_ranked_homes_step(visible: pd.DataFrame, profile: dict[str, Any]) -> None:
    st.markdown("<div class='right-head'><div class='panel-title'>Ranked Homes</div><div class='view-all'>Top picks</div></div>", unsafe_allow_html=True)
    recs = sort_visible(visible, True).head(4)
    if recs.empty:
        st.info("No ranked homes match these filters yet.")
        return
    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        recommendation_card(row, profile, f"ranked_{i}_{slug(row.get('Address', ''))}")


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
        rules, response = answer_chat(question, visible, st.session_state.get("ai_rules", {}))
        st.session_state["ai_rules"] = rules
        st.session_state.setdefault("chat_messages", []).append({"role": "assistant", "content": response.get("answer", "I saved that preference.")})
        st.rerun()
