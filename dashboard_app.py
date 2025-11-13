# dashboard_app.py — NBA Player Props Dashboard
# (context + safe merge + matchup color + fatigue chips + trends + confidence + venue)

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
    """Green = easier matchup (higher rank number = worse defense)."""
    if pd.isna(rank): return "#999999"
    try:
        r = float(rank)
    except Exception:
        return "#999999"
    if r >= 20: return "#2ecc71"   # easier opponent
    elif r >= 10: return "#f39c12" # medium
    else: return "#e74c3c"         # tough defense

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
    """
    Smooth gradient from red (0) -> yellow (~50) -> green (100) using HSL hue.
    """
    try:
        s = float(score_0_100)
    except Exception:
        s = 50.0
    s = max(0.0, min(100.0, s))
    hue = 120.0 * (s / 100.0)   # 0=red, 120=green
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

def load_results_history():
    df = load_csv("results_history.csv")

    # 🔥 FIX: Normalize date format to yyyy-mm-dd for matching
    if not df.empty and "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")

    return df

STAT_COL_BY_MARKET = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "3PM": "FG3M",
}

# --- Opponent Allowed Chip helper ---
def opp_allowed_chip(row, market: str):
    """
    Show opponent allowed averages for this market (e.g., 'Opp PTS 112.4 (#25)').
    Uses columns provided by team_context.csv merge (e.g., PTS_ALLOWED, PTS_ALLOWED_RANK).
    """
    stat_key = STAT_COL_BY_MARKET.get(market, "")
    if not stat_key:
        return ""
    allow_col = f"{stat_key}_ALLOWED"
    rank_col  = f"{stat_key}_ALLOWED_RANK"

    val  = row.get(allow_col, np.nan)
    rank = row.get(rank_col, np.nan)

    if pd.isna(val) and pd.isna(rank):
        return ""

    val_txt  = f"{float(val):.1f}" if pd.notna(val) else "—"
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

def recent_hits_vs_line(gl_df: pd.DataFrame, player: str, stat_col: str, line_val: float, lookback: int = 5):
    if gl_df.empty or stat_col not in gl_df.columns:
        return (np.nan, 0, [])
    g = gl_df.copy()
    g["PLAYER"] = g["PLAYER"].astype(str).str.strip()
    g = g[g["PLAYER"] == player].copy()
    if g.empty: return (np.nan, 0, [])
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    g = g.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")
    g = g.tail(lookback).copy()
    vals = pd.to_numeric(g[stat_col], errors="coerce")
    hits = (vals > float(line_val)).astype(int)
    hit_rate = hits.mean() if len(hits) else np.nan
    return (float(hit_rate) if pd.notna(hit_rate) else np.nan, int(len(hits)), hits.tolist())

def recent_window_stats(gl_df: pd.DataFrame, player: str, stat_col: str, lookback: int = 5):
    """
    Return last-N average and std for player's stat_col.
    """
    if gl_df.empty or stat_col not in gl_df.columns:
        return (np.nan, np.nan)
    g = gl_df.copy()
    g["PLAYER"] = g["PLAYER"].astype(str).str.strip()
    g = g[g["PLAYER"] == player].copy()
    if g.empty: return (np.nan, np.nan)
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    g = g.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").tail(lookback)
    vals = pd.to_numeric(g[stat_col], errors="coerce").dropna()
    if vals.empty: return (np.nan, np.nan)
    return (float(vals.mean()), float(vals.std(ddof=0)))

# ---------- Load Data ----------
preds, logos, heads, gl, ctx, stats = load_all_data()
results = load_results_history()

# ---------- Manual Reload ----------
if st.sidebar.button("🔄 Reload Data"):
    preds, logos, heads, gl, ctx, stats = load_all_data()
    st.toast("✅ Data reloaded successfully!")

if preds.empty:
    st.error("No player props found.")
    st.stop()

# Normalize
preds["PLAYER"] = preds["PLAYER"].astype(str).str.strip()
preds["TEAM"] = preds["TEAM"].astype(str).map(norm_team)
preds["PLAYER_NORM"] = preds["PLAYER"].str.lower().str.strip()

# Merge headshots
if not heads.empty:
    col_player = "PLAYER" if "PLAYER" in heads.columns else "player"
    col_url = "PHOTO_URL" if "PHOTO_URL" in heads.columns else "image_url"
    heads = heads.rename(columns={col_player: "PLAYER", col_url: "PHOTO_URL"})
    heads["PLAYER_NORM"] = heads["PLAYER"].astype(str).str.lower().str.strip()
    preds = preds.merge(heads[["PLAYER_NORM", "PHOTO_URL"]], on="PLAYER_NORM", how="left")
else:
    preds["PHOTO_URL"] = ""

# Merge logos
if not logos.empty:
    logos["TEAM"] = logos["TEAM"].astype(str).str.strip().str.upper().map(norm_team)
    preds["TEAM"] = preds["TEAM"].astype(str).str.strip().str.upper().map(norm_team)
    preds = preds.merge(
        logos[["TEAM", "LOGO_URL", "PRIMARY_COLOR", "SECONDARY_COLOR"]],
        on="TEAM", how="left"
    )
else:
    preds[["TEAM_FULL", "LOGO_URL", "PRIMARY_COLOR", "SECONDARY_COLOR"]] = ["", "", "", ""]

preds["PHOTO_URL"] = preds["PHOTO_URL"].fillna("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png")
preds["LOGO_URL"] = preds["LOGO_URL"].fillna("")
preds["PRIMARY_COLOR"] = preds["PRIMARY_COLOR"].fillna("#333333")
preds["SECONDARY_COLOR"] = preds["SECONDARY_COLOR"].fillna("#777777")

# ---------- Merge Contextual Data (matchup + fatigue, safe) ----------
if not ctx.empty:
    ctx = ctx.copy()
    ctx["TEAM_ABBREVIATION"] = ctx["TEAM_ABBREVIATION"].astype(str).map(norm_team)

    if "DEF_RATING_RANK" not in ctx.columns and "DEF_RATING" in ctx.columns:
        ctx["DEF_RATING_RANK"] = ctx["DEF_RATING"].rank(ascending=True)

    # baseline context cols
    maybe_cols = [
        "TEAM_ABBREVIATION", "OPP_TEAM_FULL",
        "DEF_RATING", "DEF_RATING_RANK",
        "DAYS_REST", "IS_B2B", "TRAVEL_MILES", "TRAVEL_FATIGUE"
    ]

    # add opponent-allowed averages if present in team_context.csv
    # these come from team_allowed_averages.csv being merged upstream
    allowed_bases = ["PTS", "REB", "AST", "STL", "FG3M"]
    for base in allowed_bases:
        for suffix in ["ALLOWED", "ALLOWED_RANK"]:
            col = f"{base}_{suffix}"
            if col in ctx.columns:
                maybe_cols.append(col)

    use_cols = [c for c in maybe_cols if c in ctx.columns]

    preds = preds.merge(
        ctx[use_cols],
        left_on="TEAM",
        right_on="TEAM_ABBREVIATION",
        how="left"
    )
else:
    preds["OPP_TEAM_FULL"] = ""
    preds["DEF_RATING"] = np.nan
    preds["DEF_RATING_RANK"] = np.nan
    preds["DAYS_REST"] = np.nan
    preds["IS_B2B"] = ""
    preds["TRAVEL_MILES"] = np.nan
    preds["TRAVEL_FATIGUE"] = np.nan
    # optional: initialize allowed columns so UI doesn't error if missing
    for c in ["PTS_ALLOWED","PTS_ALLOWED_RANK","REB_ALLOWED","REB_ALLOWED_RANK",
              "AST_ALLOWED","AST_ALLOWED_RANK","STL_ALLOWED","STL_ALLOWED_RANK",
              "FG3M_ALLOWED","FG3M_ALLOWED_RANK"]:
        preds[c] = np.nan

# Safely merge nba_today_stats.csv (team side + season avgs + venue if present)
if not stats.empty:
    stats = stats.copy()
    stats["PLAYER"] = stats["PLAYER"].astype(str).str.strip()
    stats["TEAM"] = stats["TEAM"].astype(str).map(norm_team)

    cols_base = ["PLAYER", "TEAM", "TEAM_SIDE", "PTS", "REB", "AST", "STL", "FG3M"]
    venue_cols = [c for c in ["ARENA", "CITY", "STATE"] if c in stats.columns]
    have = [c for c in cols_base if c in stats.columns] + venue_cols

    preds = preds.merge(stats[have], on=["PLAYER", "TEAM"], how="left")
else:
    preds["TEAM_SIDE"] = ""
    for col in ["PTS", "REB", "AST", "STL", "FG3M"]:
        if col not in preds.columns:
            preds[col] = np.nan
    for col in ["ARENA", "CITY", "STATE"]:
        preds[col] = ""

# ---------- Mode Toggle ----------
view_mode = st.sidebar.radio("View Mode", ["📊 Predictions", "🕓 Yesterday's Results"], index=0)

# ---------- Sidebar (Predictions) ----------
if view_mode == "📊 Predictions":
    st.sidebar.title("🔎 Filters")

    teams = ["All Teams"] + sorted(preds["TEAM"].dropna().unique().tolist())
    team_pick = st.sidebar.selectbox("Select Team", teams)

    if team_pick != "All Teams":
        player_opts = ["All Players"] + sorted(preds.loc[preds["TEAM"] == team_pick, "PLAYER"].unique().tolist())
    else:
        player_opts = ["All Players"] + sorted(preds["PLAYER"].unique().tolist())

    player_pick = st.sidebar.selectbox("Select Player", player_opts)
    sort_by = st.sidebar.selectbox("Sort by", ["Prob Over (desc)", "Recent Hit Rate (desc)", "Line Edge (SEASON_VAL - LINE)"])
    lookback = st.sidebar.slider("Lookback (games)", 3, 10, 5)

# ---------- Filter ----------
    view = preds.copy()
    if team_pick != "All Teams":
        view = view[view["TEAM"] == team_pick]
    if player_pick != "All Players":
        view = view[view["PLAYER"] == player_pick]

    st.markdown("### 🏀 NBA Player Props Dashboard")
    st.caption("Daily NBA Trends & Predictions — updated at least 2 hours before first tip.")

    if view.empty:
        st.info("No matching player props.")
        st.stop()

    markets = view["MARKET"].dropna().unique().tolist()
    tabs = st.tabs([m for m in markets])

# ---------- Confidence Index ----------
    def confidence_index(row, stat_col: str, recent_hit_rate: float, lookback_avg: float, line_edge: float) -> tuple    [float, str]:
        """
        Returns (score_0_100, tooltip_text).
        Weights:
          - Model prob (35%)
          - Recent hit vs line (25%)
          - Line edge (15%, logistic)
          - Opponent ease (prefers MARKET_ALLOWED_RANK; falls back to DEF_RATING_RANK) (15%)
          - Context (home/rest/travel/b2b/fatigue) (10%)
        """
        # model & recent
        p_model  = clamp01(row.get("FINAL_OVER_PROB", np.nan))
        p_recent = clamp01(recent_hit_rate)

        # line edge -> logistic
        try:
            edge = float(line_edge)
            edge_s = 1.0 / (1.0 + math.exp(-1.6 * edge))
        except Exception:
            edge_s = 0.5

        # opponent ease: prefer market-specific *_ALLOWED_RANK if present
        # infer current market from stat_col via STAT_COL_BY_MARKET reverse map
        rev = {v: k for k, v in STAT_COL_BY_MARKET.items()}
        market = rev.get(stat_col, "")
        rank_col = f"{stat_col}_ALLOWED_RANK" if market else None
        if not rank_col or rank_col not in row.index:
            # fall back gracefully
            rank_col = "DEF_RATING_RANK"

        try:
            rank = float(row.get(rank_col, np.nan))
            opp_ease = (rank - 1.0) / 29.0 if pd.notna(rank) else 0.5
            opp_ease = max(0.0, min(1.0, opp_ease))
        except Exception:
            opp_ease = 0.5

        # context
        side = str(row.get("TEAM_SIDE", "")).strip().lower()
        home_bonus = 1.0 if side == "home" else (0.85 if side in ("away", "road") else 0.9)
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

    # ---------- Display ----------
    for tab, market in zip(tabs, markets):
        with tab:
            sub = view[view["MARKET"] == market].copy()
            if sub.empty:
                st.info("No data for this market.")
                continue

            stat_col = STAT_COL_BY_MARKET.get(market, None)
            if stat_col and not gl.empty:
                gl_local = gl.copy()
                gl_local["PLAYER"] = gl_local["PLAYER"].astype(str).str.strip()
                if stat_col in gl_local.columns:
                    gl_local[stat_col] = pd.to_numeric(gl_local[stat_col], errors="coerce")
                gl_local["GAME_DATE"] = pd.to_datetime(gl_local["GAME_DATE"], errors="coerce")
                gl_local = gl_local.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")
            else:
                gl_local = pd.DataFrame()

            # Per-row calculations
            hit_rates, hit_ns, hit_series = [], [], []
            last_avgs, last_stds = [], []
            for _, r in sub.iterrows():
                if not stat_col or gl_local.empty or pd.isna(r.get("LINE")):
                    hit_rates.append(np.nan); hit_ns.append(0); hit_series.append([]); last_avgs.append(np.nan); last_stds.append(np.nan); continue
                hr, n_used, series = recent_hits_vs_line(gl_local, r["PLAYER"], stat_col, float(r["LINE"]), lookback=lookback)
                avg_n, std_n = recent_window_stats(gl_local, r["PLAYER"], stat_col, lookback=lookback)
                hit_rates.append(hr); hit_ns.append(n_used); hit_series.append(series); last_avgs.append(avg_n); last_stds.append(std_n)

            sub["recent_hit_rate_line"], sub["recent_hit_n"], sub["recent_hit_series"] = hit_rates, hit_ns, hit_series
            sub["recent_avg"], sub["recent_std"] = last_avgs, last_stds
            sub["line_edge"] = pd.to_numeric(sub.get("SEASON_VAL", 0), errors="coerce") - pd.to_numeric(sub.get("LINE", 0), errors="coerce")

            # Sorting
            if sort_by == "Prob Over (desc)":
                sub = sub.sort_values("FINAL_OVER_PROB", ascending=False)
            elif sort_by == "Recent Hit Rate (desc)":
                sub = sub.sort_values(sub["recent_hit_rate_line"].fillna(-1), ascending=False)
            elif sort_by == "Line Edge (SEASON_VAL - LINE)":
                sub = sub.sort_values("line_edge", ascending=False)

            st.subheader(f"{market} · Top Overs")
            st.divider()

            for _, row in sub.head(10).iterrows():
                prim, sec = row.get("PRIMARY_COLOR", "#333333"), row.get("SECONDARY_COLOR", "#777777")
                with st.container(border=True):
                    st.markdown(
                        f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>",
                        unsafe_allow_html=True,
                    )
                    c1, c2, c3 = st.columns([1.0, 3.2, 2.6])

                    with c1:
                        if isinstance(row.get("PHOTO_URL",""),str) and str(row["PHOTO_URL"]).startswith("http"):
                            st.image(row["PHOTO_URL"], width=80)
                        if isinstance(row.get("LOGO_URL",""),str) and str(row["LOGO_URL"]).startswith("http"):
                            st.image(row["LOGO_URL"], width=44)

                    with c2:
                        st.markdown(f"#### {row['PLAYER']}")
                        st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")

                        # Matchup / defense rank line
                        opp = row.get("OPP_TEAM_FULL","?")
                        side = row.get("TEAM_SIDE","")
                        avg_val = row.get(stat_col, np.nan)  # season avg for stat_col (from nba_today_stats.csv)
                        opp_def = row.get("DEF_RATING", np.nan)
                        opp_rank = row.get("DEF_RATING_RANK", np.nan)
                        m_color = matchup_color(opp_rank)
                        avg_txt = f"{avg_val:.1f}" if pd.notna(avg_val) else "—"
                        def_txt = f"{opp_def:.1f}" if pd.notna(opp_def) else "—"
                        rank_txt = f"#{int(opp_rank)}" if pd.notna(opp_rank) else "—"
                        st.markdown(
                            f"<div style='color:{m_color};font-size:0.9em;'>vs {opp} ({side}) | "
                            f"Avg: {avg_txt} {market} | Opp D-Rtg: {def_txt} ({rank_txt})</div>",
                            unsafe_allow_html=True,
                         )

                         # Venue (if available)
                        venue_bits = [str(row.get("ARENA","")).strip(), str(row.get("CITY","")).strip(), str(row.get("STATE","")).strip()]
                        venue_bits = [b for b in venue_bits if b]
                        if venue_bits:
                            st.caption("Venue: " + " · ".join(venue_bits))

                        # Fatigue / travel chips
                        days_rest = row.get("DAYS_REST", np.nan)
                        is_b2b = row.get("IS_B2B", "")
                        miles = row.get("TRAVEL_MILES", np.nan)
                        fatigue = row.get("TRAVEL_FATIGUE", np.nan)

                        chips = []
                        if pd.notna(days_rest):
                            chips.append(chip_html("Rest", f"{fmt_num(days_rest,0)}d", rest_color(days_rest)))
                        if str(is_b2b) != "":
                            chips.append(chip_html("B2B", str(is_b2b), b2b_color(is_b2b)))
                        if pd.notna(miles):
                            chips.append(chip_html("Travel", f"{fmt_num(miles,0)} mi", travel_color(miles)))
                        if pd.notna(fatigue):
                            chips.append(chip_html("Fatigue", fmt_num(fatigue,3), fatigue_color(fatigue)))

                        # Contextual boost chips
                        opp_chip = chip_html("Opp Ease", rank_txt, matchup_color(opp_rank))
                        edge_val = row.get("line_edge", np.nan)
                        edge_col = color_for( clamp01( 1.0/(1.0+math.exp(-1.6*float(edge_val))) ) ) if pd.notna(edge_val) else "#777777"
                        edge_chip = chip_html("Edge", fmt_num(edge_val,2), edge_col)
                        home_chip = chip_html("Home", "Yes" if str(side).lower()=="home" else "No", "#3498db", "white")

                        chips.extend([opp_chip, edge_chip, home_chip])
                    
                        allowed_chip = opp_allowed_chip(row, market)
                        if allowed_chip:
                            chips.append(allowed_chip)

                        if chips:
                            st.markdown("".join(chips), unsafe_allow_html=True)

                        st.markdown(f"Team: `{row['TEAM']}` | Injury: {row.get('INJ_Status','Active')}")

                    with c3:
                        # Model prob + Last N vs line
                        st.metric("Prob. Over (Model)", row.get("FINAL_OVER_PROB_PCT","—"))
                        hr = row.get("recent_hit_rate_line", np.nan)
                        n = int(row.get("recent_hit_n", 0) or 0)
                        st.markdown(
                            f"<div style='color:{color_for(hr)}'>Last {lookback} vs line: {pct(hr)} ({n}g)</div>",
                            unsafe_allow_html=True,
                        )
                        chart = sparkline(row.get("recent_hit_series", []), color=color_for(hr))
                        if chart is not None:
                            st.altair_chart(chart, use_container_width=True)

                        # Performance Trend (last N vs season) & volatility
                        recent_avg = row.get("recent_avg", np.nan)
                        season_avg = row.get(stat_col, np.nan)
                        delta = (recent_avg - season_avg) if (pd.notna(recent_avg) and pd.notna(season_avg)) else np.nan
                        vol = row.get("recent_std", np.nan)
                        delta_color = "#2ecc71" if (pd.notna(delta) and delta > 0) else ("#e74c3c" if pd.notna(delta) else "#999999")
                        st.markdown(
                            f"<div style='margin-top:6px;font-size:0.9em;'>"
                            f"Trend (last {lookback}): <b>{fmt_num(recent_avg,1)}</b> vs season <b>{fmt_num(season_avg,1)}</b> "
                            f"<span style='color:{delta_color};'>(Δ {fmt_num(delta,1)})</span> • "
                            f"Vol: σ={fmt_num(vol,2)}</div>",
                            unsafe_allow_html=True
                        )

                        # Confidence Index
                        conf, tooltip = confidence_index(row, stat_col, hr, recent_avg, row.get("line_edge", np.nan))
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

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            st.divider()

else:
    # ---------- Yesterday's Results ----------
    st.markdown("### 🕓 Yesterday’s Results — Top 10 Overs Recap")

    if results.empty:
        st.info("No results_history.csv found or it is empty.")
    else:
        # Yesterday = most recent date in results_history.csv
        latest_date = results["DATE"].max()   # yyyy-mm-dd string
        df_yday = results[results["DATE"] == latest_date]

        if df_yday.empty:
            st.warning(f"No results found for {latest_date}.")
        else:
            pretty_date = pd.to_datetime(latest_date).strftime("%B %d, %Y")
            st.markdown(f"#### Results for {pretty_date}")

    # ---- Market Selector ----
    markets = sorted(df_yday["MARKET"].unique())
    market_pick = st.selectbox("Select Market", markets, index=0)

    df_market = df_yday[df_yday["MARKET"] == market_pick].copy()

    st.markdown(f"### 🔍 {market_pick} — Player Results")

    # Color-coded hit/miss column
    def hit_style(val):
        color = "#2ecc71" if val == 1 else "#e74c3c"
        return f"background-color:{color}; color:white; font-weight:bold;"

    # Prepare table
    df_show = df_market[["PLAYER", "TEAM", "LINE", "ACTUAL", "didHitOver"]].copy()
    df_show = df_show.sort_values("didHitOver", ascending=False)

    df_show["RESULT"] = df_show["didHitOver"].map({1: "HIT", 0: "MISS"})

    # Display table with highlight
    st.dataframe(
        df_show.style.applymap(
            lambda v: hit_style(v) if v in (1,0,"HIT","MISS") else ""
        ),
        hide_index=True,
    )

    # Optional: Show only top 10 players for that market
    st.markdown("### ⭐ Top 10 for This Market")
    st.dataframe(
        df_show.head(10).style.applymap(
            lambda v: hit_style(v) if v in (1,0,"HIT","MISS") else ""
        ),
        hide_index=True
    )

            # Per-category summary (Top 10 lists combined)
    summary = (
                df_yday.groupby("MARKET")["didHitOver"]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={"sum": "Hits", "count": "Total"})
            )
    summary["HitRate"] = (summary["Hits"] / summary["Total"]).fillna(0.0)
    summary["HitRatePct"] = summary["HitRate"].apply(lambda x: f"{x*100:.1f}%")

    st.dataframe(summary[["MARKET", "Hits", "Total", "HitRatePct"]])

    # Overall yesterday (all categories)
    total_hits = int(summary["Hits"].sum())
    total_total = int(summary["Total"].sum())
    overall_rate = (total_hits / total_total) if total_total > 0 else 0.0
    st.markdown(
        f"**Overall Yesterday Hit Rate (All Categories):** {total_hits}/{total_total} → {overall_rate*100:.1f}%"
    )

    # Running totals across all history
    summary_all = (
        results.groupby("MARKET")["didHitOver"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "AllHits", "count": "AllTotal"})
    )
    summary_all["AllHitRate"] = (summary_all["AllHits"] / summary_all["AllTotal"]).fillna(0.0)
    summary_all["AllHitRatePct"] = summary_all["AllHitRate"].apply(lambda x: f"{x*100:.1f}%")

    st.markdown("#### 📈 Running Totals by Category (All-Time)")
    st.dataframe(summary_all[["MARKET", "AllHits", "AllTotal", "AllHitRatePct"]])

    # Overall running totals (all categories combined)
    overall_hits_all = int(summary_all["AllHits"].sum())
    overall_total_all = int(summary_all["AllTotal"].sum())
    overall_rate_all = (overall_hits_all / overall_total_all) if overall_total_all > 0 else 0.0
    st.markdown(
        f"**Overall Running Hit Rate (All Categories):** {overall_hits_all}/{overall_total_all} → {overall_rate_all*100:.1f}%"
    )

    # Optional chart: yesterday per-category hit rate
    try:
        import altair as alt
        chart = (
            alt.Chart(summary)
            .mark_bar()
            .encode(
                x=alt.X("MARKET:N", title="Market"),
                y=alt.Y("HitRate:Q", title="Hit Rate", scale=alt.Scale(domain=[0, 1])),
                tooltip=["MARKET", "Hits", "Total", "HitRatePct"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass  # keep UI resilient

st.caption("Daily NBA Trends & Predictions — powered by your pipeline • Context, trends & confidence • Free on Streamlit Cloud")