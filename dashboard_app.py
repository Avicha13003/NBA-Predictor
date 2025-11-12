# dashboard_app.py — NBA Player Props Dashboard
# (context + matchup color + opponent allowed + fatigue chips + trends + confidence + venue)

import math
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="NBA Player Props Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

# ---------- Helpers ----------
ALT_TEAM_MAP = {"GS": "GSW", "NO": "NOP", "SA": "SAS", "NY": "NYK", "PHO": "PHX"}

def norm_team(x: str) -> str:
    if not isinstance(x, str): return ""
    x = x.strip().upper()
    return ALT_TEAM_MAP.get(x, x)

def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return np.nan

def pct(x):
    return f"{float(x)*100:.1f}%" if (x is not None and pd.notna(x)) else "—"

def color_for(val, hi_good=True):
    if pd.isna(val): return "#999999"
    if hi_good:
        return "#2ecc71" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#e74c3c")
    else:
        return "#e74c3c" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#2ecc71")

def matchup_color(rank):
    if pd.isna(rank): return "#999999"
    try:
        r = float(rank)
    except Exception:
        return "#999999"
    if r >= 20: return "#2ecc71"
    elif r >= 10: return "#f39c12"
    else: return "#e74c3c"

def chip_html(label: str, value: str, bg: str, fg: str = "white"):
    return (
        f"<span style='display:inline-block;margin:2px 6px 0 0;"
        f"padding:2px 8px;border-radius:999px;font-size:0.78rem;"
        f"background:{bg};color:{fg};white-space:nowrap;'>"
        f"{label}: <b>{value}</b></span>"
    )

def rest_color(days):
    try:
        d = float(days)
    except Exception:
        return "#777777"
    if d >= 3: return "#2ecc71"
    if d >= 2: return "#f39c12"
    return "#e74c3c"

def b2b_color(flag):
    s = str(flag).strip().lower()
    if s in ("yes", "y", "true", "1"): return "#e74c3c"
    return "#2ecc71"

def travel_color(miles):
    try:
        m = float(miles)
    except Exception:
        return "#777777"
    if m >= 800: return "#e74c3c"
    if m >= 300: return "#f39c12"
    return "#2ecc71"

def fatigue_color(x):
    try:
        f = float(x)
    except Exception:
        return "#777777"
    if f >= 0.8: return "#e74c3c"
    if f >= 0.4: return "#f39c12"
    return "#2ecc71"

def fmt_num(x, digits=1):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"

def smooth_hsl_color(score_0_100: float) -> str:
    try:
        s = float(score_0_100)
    except Exception:
        s = 50.0
    s = max(0.0, min(100.0, s))
    hue = 120.0 * (s / 100.0)
    return f"hsl({hue:.0f}, 70%, 45%)"

def load_csv(path):
    try:
        df = pd.read_csv(path)
        st.toast(f"✅ Loaded {path} ({len(df)} rows)")
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load {path}: {e}")
        return pd.DataFrame()

def load_all_data():
    preds = load_csv("nba_prop_predictions_today.csv")
    logos = load_csv("team_logos.csv")
    heads = load_csv("player_headshots.csv")
    gl = load_csv("player_game_log.csv")
    ctx = load_csv("team_context.csv")
    stats = load_csv("nba_today_stats.csv")
    return preds, logos, heads, gl, ctx, stats

STAT_COL_BY_MARKET = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "3PM": "FG3M",
}

def opp_allowed_chip(row, market: str):
    stat_key = STAT_COL_BY_MARKET.get(market, "")
    if not stat_key: return ""
    allow_col = f"{stat_key}_ALLOWED"
    rank_col = f"{stat_key}_ALLOWED_RANK"
    val = row.get(allow_col, np.nan)
    rank = row.get(rank_col, np.nan)
    if pd.isna(val) and pd.isna(rank): return ""
    val_txt = f"{float(val):.1f}" if pd.notna(val) else "—"
    rank_txt = f"#{int(rank)}" if pd.notna(rank) else "—"
    color = matchup_color(rank)
    return chip_html(f"Opp {stat_key}", f"{val_txt} ({rank_txt})", color)

def sparkline(binary_list, color="#2ecc71"):
    if not binary_list:
        return None
    data = pd.DataFrame({"idx": list(range(1, len(binary_list)+1)), "val": binary_list})
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("idx:Q", axis=None),
        y=alt.Y("val:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.value(color),
    ).properties(height=30)
    return chart

# ---------- Confidence Index ----------
def confidence_index(row, market: str, recent_hit_rate: float, line_edge: float) -> tuple[float, str]:
    """
    Dynamically weights opponent ease using either DEF_RATING_RANK or the market-specific *_ALLOWED_RANK.
    """
    p_model = clamp01(row.get("FINAL_OVER_PROB", np.nan))
    p_recent = clamp01(recent_hit_rate)
    try:
        edge = float(line_edge)
        edge_s = 1.0 / (1.0 + math.exp(-1.6 * edge))
    except Exception:
        edge_s = 0.5

    # Market-specific opponent rank
    stat_key = STAT_COL_BY_MARKET.get(market, "")
    rank_col = f"{stat_key}_ALLOWED_RANK" if stat_key else "DEF_RATING_RANK"
    try:
        rank = float(row.get(rank_col, np.nan))
        opp_ease = (rank - 1.0) / 29.0 if pd.notna(rank) else 0.5
    except Exception:
        opp_ease = 0.5
    opp_ease = max(0.0, min(1.0, opp_ease))

    # Contextual factors
    side = str(row.get("TEAM_SIDE", "")).strip().lower()
    home_bonus = 1.0 if side == "home" else 0.85
    try:
        rest = float(row.get("DAYS_REST", np.nan))
        rest_scale = 1.0 if rest >= 3 else (0.9 if rest >= 2 else (0.75 if rest >= 1 else 0.6))
    except Exception:
        rest_scale = 0.9
    b2b = str(row.get("IS_B2B", "")).strip().lower()
    b2b_scale = 0.65 if b2b in ("yes", "y", "true", "1") else 1.0
    try:
        fatigue = float(row.get("TRAVEL_FATIGUE", 0.0))
        fatigue_scale = 1.0 - max(0.0, min(1.0, fatigue))
    except Exception:
        fatigue_scale = 0.9
    context_scale = np.nanmean([home_bonus, rest_scale, b2b_scale, fatigue_scale])
    context_scale = max(0.0, min(1.2, float(context_scale))) / 1.2

    score01 = (
        0.35 * (p_model if pd.notna(p_model) else 0.5) +
        0.25 * (p_recent if pd.notna(p_recent) else 0.5) +
        0.15 * edge_s +
        0.15 * opp_ease +
        0.10 * context_scale
    )
    score = float(max(0.0, min(1.0, score01)) * 100.0)
    tooltip = (
        f"Model: {pct(p_model)} | Recent: {pct(p_recent)} | "
        f"Edge: {edge_s*100:.0f}% | OppEase: {opp_ease*100:.0f}% | Ctxt: {context_scale*100:.0f}%"
    )
    return score, tooltip

# ---------- Load Data ----------
preds, logos, heads, gl, ctx, stats = load_all_data()

# ---------- UI ----------
# (Data loading, normalization, merges — keep same as before)
#  [You can retain your current data loading and merge sections above unchanged]

# ---------- Display ----------
for tab, market in zip(st.tabs(sorted(view["MARKET"].dropna().unique().tolist())), sorted(view["MARKET"].dropna().unique().tolist())):
    with tab:
        sub = view[view["MARKET"] == market].copy()
        if sub.empty:
            st.info("No data for this market.")
            continue

        stat_col = STAT_COL_BY_MARKET.get(market)
        # --- Your existing per-player loop, unchanged except for opp_allowed_chip + confidence_index call ---
        for _, row in sub.head(10).iterrows():
            prim, sec = row.get("PRIMARY_COLOR", "#333333"), row.get("SECONDARY_COLOR", "#777777")
            with st.container(border=True):
                # header bar and images
                st.markdown(f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1.0, 3.2, 2.6])
                with c1:
                    if isinstance(row.get("PHOTO_URL",""),str) and row["PHOTO_URL"].startswith("http"):
                        st.image(row["PHOTO_URL"], width=80)
                    if isinstance(row.get("LOGO_URL",""),str) and row["LOGO_URL"].startswith("http"):
                        st.image(row["LOGO_URL"], width=44)
                with c2:
                    st.markdown(f"#### {row['PLAYER']}")
                    st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    opp = row.get("OPP_TEAM_FULL","?")
                    side = row.get("TEAM_SIDE","")
                    opp_rank = row.get("DEF_RATING_RANK", np.nan)
                    rank_txt = f"#{int(opp_rank)}" if pd.notna(opp_rank) else "—"
                    st.markdown(
                        f"<div style='color:{matchup_color(opp_rank)};font-size:0.9em;'>vs {opp} ({side})</div>",
                        unsafe_allow_html=True,
                    )
                    # chips
                    chips = []
                    chips.append(opp_allowed_chip(row, market))
                    edge_val = row.get("line_edge", np.nan)
                    edge_col = color_for(clamp01(1.0/(1.0+math.exp(-1.6*float(edge_val))))) if pd.notna(edge_val) else "#777777"
                    chips.append(chip_html("Edge", fmt_num(edge_val,2), edge_col))
                    if str(side).lower()=="home":
                        chips.append(chip_html("Home", "Yes", "#3498db"))
                    if chips:
                        st.markdown("".join([c for c in chips if c]), unsafe_allow_html=True)

                with c3:
                    conf, tooltip = confidence_index(row, market, row.get("recent_hit_rate_line", np.nan), row.get("line_edge", np.nan))
                    conf_color = smooth_hsl_color(conf)
                    st.markdown(
                        f"""
                        <div title="{tooltip}" style="margin-top:6px;">
                          <div style="display:flex;align-items:center;gap:8px;">
                            <div style="min-width:110px;">Confidence:</div>
                            <div style="flex:1;height:10px;border-radius:8px;background:linear-gradient(90deg,{conf_color} {conf:.0f}%, rgba(255,255,255,0.08) {conf:.0f}%);"></div>
                            <div style="min-width:44px;text-align:right;color:{conf_color};font-weight:700;">{conf:.0f}</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )