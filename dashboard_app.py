# dashboard_app.py — NBA Player Props Dashboard (context + color-coded matchup difficulty)
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

def pct(x):
    return f"{float(x)*100:.1f}%" if (x is not None and pd.notna(x)) else "—"

def color_for(val, hi_good=True):
    if pd.isna(val): return "#999999"
    if hi_good:
        return "#2ecc71" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#e74c3c")
    else:
        return "#e74c3c" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#2ecc71")

def matchup_color(rank):
    """Green = easy matchup (high rank number = worse defense)."""
    if pd.isna(rank): return "#999999"
    if rank >= 20: return "#2ecc71"   # easier opponent
    elif rank >= 10: return "#f39c12" # medium
    else: return "#e74c3c"            # tough defense

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

# Map dashboard markets → game log stat columns
STAT_COL_BY_MARKET = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "3PM": "FG3M",
}

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

# ---------- Load Data ----------
preds, logos, heads, gl, ctx, stats = load_all_data()

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

# ---------- Merge Contextual Data ----------
if not ctx.empty:
    ctx["TEAM_ABBREVIATION"] = ctx["TEAM_ABBREVIATION"].astype(str).map(norm_team)
    ctx["OPP_TEAM_FULL"] = ctx["OPP_TEAM_FULL"].astype(str)
    ctx["DEF_RATING_RANK"] = ctx["DEF_RATING"].rank(ascending=True)
    preds = preds.merge(
        ctx[["TEAM_ABBREVIATION", "OPP_TEAM_FULL", "DEF_RATING", "DEF_RATING_RANK"]],
        left_on="TEAM", right_on="TEAM_ABBREVIATION", how="left"
    )
else:
    preds["OPP_TEAM_FULL"] = ""
    preds["DEF_RATING"] = np.nan
    preds["DEF_RATING_RANK"] = np.nan

# Safely merge nba_today_stats.csv
if not stats.empty:
    stats["PLAYER"] = stats["PLAYER"].astype(str).str.strip()
    stats["TEAM"] = stats["TEAM"].astype(str).map(norm_team)

    # Only include existing columns
    cols = ["PLAYER", "TEAM_SIDE", "PTS", "REB", "AST", "FG3M"]
    existing_cols = [c for c in cols if c in stats.columns]

    preds = preds.merge(stats[existing_cols + ["TEAM"]], on=["PLAYER", "TEAM"], how="left")
else:
    preds["TEAM_SIDE"] = ""
    for col in ["PTS", "REB", "AST", "STL", "FG3M"]:
        preds[col] = np.nan

# ---------- Sidebar ----------
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

        hit_rates, hit_ns, hit_series = [], [], []
        for _, r in sub.iterrows():
            if not stat_col or gl_local.empty or pd.isna(r.get("LINE")):
                hit_rates.append(np.nan); hit_ns.append(0); hit_series.append([]); continue
            hr, n_used, series = recent_hits_vs_line(gl_local, r["PLAYER"], stat_col, float(r["LINE"]), lookback=lookback)
            hit_rates.append(hr); hit_ns.append(n_used); hit_series.append(series)
        sub["recent_hit_rate_line"], sub["recent_hit_n"], sub["recent_hit_series"] = hit_rates, hit_ns, hit_series
        sub["line_edge"] = pd.to_numeric(sub.get("SEASON_VAL", 0), errors="coerce") - pd.to_numeric(sub.get("LINE", 0), errors="coerce")

        if sort_by == "Prob Over (desc)": sub = sub.sort_values("FINAL_OVER_PROB", ascending=False)
        elif sort_by == "Recent Hit Rate (desc)": sub = sub.sort_values(sub["recent_hit_rate_line"].fillna(-1), ascending=False)
        elif sort_by == "Line Edge (SEASON_VAL - LINE)": sub = sub.sort_values("line_edge", ascending=False)

        st.subheader(f"{market} · Top Overs")
        st.divider()

        for _, row in sub.head(10).iterrows():
            prim, sec = row.get("PRIMARY_COLOR", "#333333"), row.get("SECONDARY_COLOR", "#777777")
            with st.container(border=True):
                st.markdown(f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1.0, 3.0, 2.2])

                with c1:
                    if isinstance(row.get("PHOTO_URL",""),str) and row["PHOTO_URL"].startswith("http"):
                        st.image(row["PHOTO_URL"], width=80)
                    if isinstance(row.get("LOGO_URL",""),str) and row["LOGO_URL"].startswith("http"):
                        st.image(row["LOGO_URL"], width=40)

                with c2:
                    st.markdown(f"#### {row['PLAYER']}")
                    st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    opp = row.get("OPP_TEAM_FULL","?")
                    side = row.get("TEAM_SIDE","")
                    avg_val = row.get(stat_col, np.nan)
                    opp_def = row.get("DEF_RATING", np.nan)
                    opp_rank = row.get("DEF_RATING_RANK", np.nan)
                    color = matchup_color(opp_rank)
                    if pd.notna(avg_val):
                        avg_txt = f"{avg_val:.1f}"
                    else:
                        avg_txt = "—"
                    st.markdown(
                        f"<div style='color:{color};font-size:0.9em;'>vs {opp} ({side}) | "
                        f"Avg: {avg_txt} {market} | Opp D-Rtg: {opp_def:.1f if pd.notna(opp_def) else '—'} (#{int(opp_rank) if pd.notna(opp_rank) else '—'})</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"Team: `{row['TEAM']}` | Injury: {row.get('INJ_Status','Active')}")

                with c3:
                    st.metric("Prob. Over (Model)", row.get("FINAL_OVER_PROB_PCT","—"))
                    hr = row.get("recent_hit_rate_line", np.nan)
                    n = int(row.get("recent_hit_n", 0) or 0)
                    st.markdown(f"<div style='color:{color_for(hr)}'>Last {lookback} vs line: {pct(hr)} ({n}g)</div>", unsafe_allow_html=True)
                    chart = sparkline(row.get("recent_hit_series", []), color=color_for(hr))
                    if chart is not None: st.altair_chart(chart, use_container_width=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.divider()

st.caption("Daily NBA Trends & Predictions — powered by your pipeline • Context-aware matchup insights • Free on Streamlit Cloud")