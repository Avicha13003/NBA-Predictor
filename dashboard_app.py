# dashboard_app.py — NBA Player Props Dashboard (line-based recent hit rates, logos fixed, GitHub hidden)
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
    ctx = load_csv("team_context.csv")   # optional, not used directly below
    return preds, logos, heads, gl, ctx

# Map dashboard markets → game log stat columns
STAT_COL_BY_MARKET = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "3PM": "FG3M",
}

def sparkline(binary_list, color="#2ecc71"):
    """binary_list: list of 0/1 by most-recent order; renders tiny line with points."""
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
    """
    Compute last-N games (by date) for `player` and whether each exceeded today's `line_val`.
    Returns (hit_rate, n_games_used, binary_series_for_sparkline)
    """
    if gl_df.empty or stat_col not in gl_df.columns:
        return (np.nan, 0, [])

    g = gl_df.copy()
    # Normalize
    g["PLAYER"] = g["PLAYER"].astype(str).str.strip()
    g = g[g["PLAYER"] == player].copy()
    if g.empty:
        return (np.nan, 0, [])

    # Parse and sort by date
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    g = g.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")

    # Take last N rows
    g = g.tail(lookback).copy()
    if g.empty:
        return (np.nan, 0, [])

    # Ensure numeric
    vals = pd.to_numeric(g[stat_col], errors="coerce")
    # Strictly greater than the line counts as OVER (push is not over)
    hits = (vals > float(line_val)).astype(int)
    hit_rate = hits.mean() if len(hits) else np.nan

    return (float(hit_rate) if pd.notna(hit_rate) else np.nan, int(len(hits)), hits.tolist())

# ---------- Initial Load ----------
preds, logos, heads, gl, ctx = load_all_data()

# ---------- Manual Reload Button ----------
if st.sidebar.button("🔄 Reload Data"):
    preds, logos, heads, gl, ctx = load_all_data()
    st.toast("✅ Data reloaded successfully!")

# ---------- Normalize + Merge ----------
if preds.empty:
    st.error("No player props found.")
    st.stop()

preds["PLAYER"] = preds["PLAYER"].astype(str).str.strip()
preds["TEAM"] = preds["TEAM"].astype(str).map(norm_team)
preds["PLAYER_NORM"] = preds["PLAYER"].str.lower().str.strip()

# --- Merge Headshots ---
if not heads.empty:
    col_player = "PLAYER" if "PLAYER" in heads.columns else "player"
    col_url = "PHOTO_URL" if "PHOTO_URL" in heads.columns else "image_url"
    heads = heads.rename(columns={col_player: "PLAYER", col_url: "PHOTO_URL"})
    heads["PLAYER_NORM"] = heads["PLAYER"].astype(str).str.lower().str.strip()
    preds = preds.merge(heads[["PLAYER_NORM", "PHOTO_URL"]], on="PLAYER_NORM", how="left")
else:
    preds["PHOTO_URL"] = ""

# --- Merge Logos (Fixed) ---
if not logos.empty:
    logos["TEAM"] = logos["TEAM"].astype(str).str.strip().str.upper().map(norm_team)
    preds["TEAM"] = preds["TEAM"].astype(str).str.strip().str.upper().map(norm_team)

    preds = preds.merge(
        logos[["TEAM", "LOGO_URL", "PRIMARY_COLOR", "SECONDARY_COLOR"]],
        on="TEAM", how="left"
    )

    # Fallback fill
    logo_map = dict(zip(logos["TEAM"], logos["LOGO_URL"]))
    preds["LOGO_URL"] = preds["LOGO_URL"].fillna(preds["TEAM"].map(logo_map))
else:
    preds[["TEAM_FULL", "LOGO_URL", "PRIMARY_COLOR", "SECONDARY_COLOR"]] = ["", "", "", ""]

preds["PHOTO_URL"] = preds["PHOTO_URL"].fillna("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png")
preds["LOGO_URL"] = preds["LOGO_URL"].fillna("")
preds["PRIMARY_COLOR"] = preds["PRIMARY_COLOR"].fillna("#333333")
preds["SECONDARY_COLOR"] = preds["SECONDARY_COLOR"].fillna("#777777")

# ---------- Sidebar Filters ----------
st.sidebar.title("🔎 Filters")
teams = ["All Teams"] + sorted(preds["TEAM"].dropna().unique().tolist())
team_pick = st.sidebar.selectbox("Select Team", teams)

if team_pick != "All Teams":
    player_opts = ["All Players"] + sorted(preds.loc[preds["TEAM"] == team_pick, "PLAYER"].unique().tolist())
else:
    player_opts = ["All Players"] + sorted(preds["PLAYER"].unique().tolist())

player_pick = st.sidebar.selectbox("Select Player", player_opts)

sort_by = st.sidebar.selectbox(
    "Sort by",
    ["Prob Over (desc)", "Recent Hit Rate (desc)", "Line Edge (SEASON_VAL - LINE)"]
)

lookback = st.sidebar.slider("Lookback (games)", 3, 10, 5)

# ---------- Filter ----------
view = preds.copy()
if team_pick != "All Teams":
    view = view[view["TEAM"] == team_pick]
if player_pick != "All Players":
    view = view[view["PLAYER"] == player_pick]

# ---------- Header ----------
st.markdown("### 🏀 NBA Player Props Dashboard")
st.caption("Daily NBA Trends & Predictions — updated at least 2 hours before first tip.")

if view.empty:
    st.info("No matching player props.")
    st.stop()

markets = view["MARKET"].dropna().unique().tolist()
tabs = st.tabs([m for m in markets])

# ---------- Main Display ----------
for tab, market in zip(tabs, markets):
    with tab:
        sub = view[view["MARKET"] == market].copy()
        if sub.empty:
            st.info("No data for this market.")
            continue

        # Compute today's line-based recent hit rate per row using player_game_log.csv
        stat_col = STAT_COL_BY_MARKET.get(market, None)
        if stat_col and not gl.empty:
            # Normalize game log once
            gl_local = gl.copy()
            gl_local["PLAYER"] = gl_local["PLAYER"].astype(str).str.strip()
            # Ensure numeric stat column
            if stat_col in gl_local.columns:
                gl_local[stat_col] = pd.to_numeric(gl_local[stat_col], errors="coerce")
            # Parse dates once
            gl_local["GAME_DATE"] = pd.to_datetime(gl_local["GAME_DATE"], errors="coerce")
            gl_local = gl_local.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")
        else:
            gl_local = pd.DataFrame()

        hit_rates, hit_ns, hit_series = [], [], []
        for _, r in sub.iterrows():
            if not stat_col or gl_local.empty or pd.isna(r.get("LINE")):
                hit_rates.append(np.nan)
                hit_ns.append(0)
                hit_series.append([])
                continue
            hr, n_used, series = recent_hits_vs_line(gl_local, r["PLAYER"], stat_col, float(r["LINE"]), lookback=lookback)
            hit_rates.append(hr)
            hit_ns.append(n_used)
            hit_series.append(series)

        sub["recent_hit_rate_line"] = hit_rates
        sub["recent_hit_n"] = hit_ns
        sub["recent_hit_series"] = hit_series

        # Extra: line edge for sort
        sub["line_edge"] = (pd.to_numeric(sub.get("SEASON_VAL", 0), errors="coerce") -
                            pd.to_numeric(sub.get("LINE", 0), errors="coerce"))

        # Sorting choice
        if sort_by == "Prob Over (desc)":
            sub = sub.sort_values("FINAL_OVER_PROB", ascending=False)
        elif sort_by == "Recent Hit Rate (desc)":
            sub = sub.sort_values(sub["recent_hit_rate_line"].fillna(-1), ascending=False)
        elif sort_by == "Line Edge (SEASON_VAL - LINE)":
            sub = sub.sort_values("line_edge", ascending=False)

        st.subheader(f"{market} · Top Overs")
        st.divider()

        for _, row in sub.head(10).iterrows():
            prim = row.get("PRIMARY_COLOR", "#333333")
            sec = row.get("SECONDARY_COLOR", "#777777")

            with st.container(border=True):
                st.markdown(
                    f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns([1.0, 3.0, 2.2])

                with c1:
                    if isinstance(row.get("PHOTO_URL", ""), str) and row["PHOTO_URL"].startswith("http"):
                        st.image(row["PHOTO_URL"], width=80)
                    if isinstance(row.get("LOGO_URL", ""), str) and row["LOGO_URL"].startswith("http"):
                        st.image(row["LOGO_URL"], width=40)

                with c2:
                    st.markdown(f"#### {row['PLAYER']}")
                    st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    st.markdown(f"Team: `{row['TEAM']}` | Injury: {row.get('INJ_Status', 'Active')}")

                with c3:
                    # Show model prob
                    st.metric("Prob. Over (Model)", row.get("FINAL_OVER_PROB_PCT", "—"))
                    # Show line-based last-N hit rate
                    hr = row.get("recent_hit_rate_line", np.nan)
                    n = int(row.get("recent_hit_n", 0) or 0)
                    st.markdown(
                        f"<div style='color:{color_for(hr)}'>Last {lookback} vs line: {pct(hr)} ({n}g)</div>",
                        unsafe_allow_html=True,
                    )
                    # Sparkline (0/1 vs line)
                    chart = sparkline(row.get("recent_hit_series", []), color=color_for(hr))
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.divider()

st.caption("Daily NBA Trends & Predictions — powered by your pipeline • Mobile-friendly • Free on Streamlit Cloud")