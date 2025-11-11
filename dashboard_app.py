# dashboard_app.py — NBA Player Props Dashboard (GitHub hidden + Recent Hit Rate fixed + logos)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------- Page & UI tweaks ----------
st.set_page_config(
    page_title="NBA Player Props Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

# Hide GitHub “view source” and other viewer badges (robust selectors for Streamlit Cloud)
st.markdown(
    """
    <style>
      /* hide any GitHub / viewer badge / header buttons */
      a[href*="github.com"]{display:none !important;}
      [data-testid="stAppViewBlockContainer"] a[aria-label*="GitHub"]{display:none !important;}
      button[title*="GitHub"]{display:none !important;}
      svg[data-testid="github-icon"]{display:none !important;}
      .viewerBadge_link__, .styles_viewerBadge__ {display:none !important;}
      /* optional: hide the "Manage app" footer link if desired */
      /* a[href*="share.streamlit.io"]{display:none !important;} */
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
ALT_TEAM_MAP = {"GS": "GSW", "NO": "NOP", "SA": "SAS", "NY": "NYK", "PHO": "PHX"}

def norm_team(x: str) -> str:
    if not isinstance(x, str): 
        return ""
    x = x.strip().upper()
    return ALT_TEAM_MAP.get(x, x)

def pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "—"

def color_for(val, hi_good=True):
    if pd.isna(val): 
        return "#999999"
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
    gl    = load_csv("player_game_log.csv")
    ctx   = load_csv("team_context.csv")  # (not used for hit rate but kept for future)
    return preds, logos, heads, gl, ctx

# ---------- Initial Load ----------
preds, logos, heads, gl, ctx = load_all_data()

# ---------- Manual Reload Button ----------
if st.sidebar.button("🔄 Reload Data"):
    preds, logos, heads, gl, ctx = load_all_data()
    st.toast("✅ Data reloaded successfully!")

# ---------- Normalize + Merge ----------
if preds.empty:
    st.error("No player props found today.")
    st.stop()

preds["PLAYER"] = preds["PLAYER"].astype(str).str.strip()
preds["TEAM"]   = preds["TEAM"].astype(str).map(norm_team)
preds["PLAYER_NORM"] = preds["PLAYER"].str.lower().str.strip()

# Headshots
if not heads.empty:
    col_player = "PLAYER" if "PLAYER" in heads.columns else "player"
    col_url    = "PHOTO_URL" if "PHOTO_URL" in heads.columns else "image_url"
    heads = heads.rename(columns={col_player: "PLAYER", col_url: "PHOTO_URL"})
    heads["PLAYER_NORM"] = heads["PLAYER"].astype(str).str.lower().str.strip()
    preds = preds.merge(heads[["PLAYER_NORM", "PHOTO_URL"]], on="PLAYER_NORM", how="left")
else:
    preds["PHOTO_URL"] = ""

# Logos (use ESPN CDN you confirmed works)
if not logos.empty:
    logos["TEAM"] = logos["TEAM"].astype(str).str.strip().str.upper().map(norm_team)
    preds["TEAM"] = preds["TEAM"].astype(str).str.strip().str.upper().map(norm_team)

    preds = preds.merge(
        logos[["TEAM","LOGO_URL","PRIMARY_COLOR","SECONDARY_COLOR"]],
        on="TEAM", how="left"
    )
    # Fallback fill
    logo_map = dict(zip(logos["TEAM"], logos["LOGO_URL"]))
    preds["LOGO_URL"] = preds["LOGO_URL"].fillna(preds["TEAM"].map(logo_map))
else:
    preds[["LOGO_URL","PRIMARY_COLOR","SECONDARY_COLOR"]] = ["","",""]

# Fill defaults
preds["PHOTO_URL"]      = preds["PHOTO_URL"].fillna("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png")
preds["LOGO_URL"]       = preds["LOGO_URL"].fillna("")
preds["PRIMARY_COLOR"]  = preds["PRIMARY_COLOR"].fillna("#333333")
preds["SECONDARY_COLOR"]= preds["SECONDARY_COLOR"].fillna("#777777")

# ---------- Sidebar Filters ----------
st.sidebar.title("🔎 Filters")
teams = ["All Teams"] + sorted(preds["TEAM"].dropna().unique().tolist())
team_pick = st.sidebar.selectbox("Select Team", teams)

if team_pick != "All Teams":
    player_opts = ["All Players"] + sorted(preds.loc[preds["TEAM"]==team_pick,"PLAYER"].unique().tolist())
else:
    player_opts = ["All Players"] + sorted(preds["PLAYER"].unique().tolist())
player_pick = st.sidebar.selectbox("Select Player", player_opts)

sort_by = st.sidebar.selectbox(
    "Sort by",
    ["Prob Over (desc)", "Recent Hit Rate", "Line Edge (SEASON_VAL - LINE)", "Volatility (std, asc)"]
)

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
    st.info("No player props match your filters.")
    st.stop()

# ---------- Recent Hit Rate logic ----------
HIT_COL = {
    "PTS":  "didHitOver_PTS",
    "3PM":  "didHitOver_FG3M",
    "REB":  "didHitOver_REB",
    "AST":  "didHitOver_AST",
    "STL":  "didHitOver_STL",
}
STAT_COL = {"PTS":"PTS","3PM":"FG3M","REB":"REB","AST":"AST","STL":"STL"}

def recent_hit_stats(gl_all: pd.DataFrame, player: str, market: str, line: float, N: int = 5):
    """
    Returns: rate (0-1 or nan), N_used, series_list (0/1 last N), last_game dict, last5_mean, last5_std
    - Prefers precomputed didHitOver_* columns.
    - Falls back to stat >= line if hit col is missing.
    """
    out = dict(rate=np.nan, n=0, series=[], last_game=None, last5_mean=np.nan, last5_std=np.nan)
    if gl_all.empty: 
        return out

    g = gl_all.copy()
    g["PLAYER"] = g["PLAYER"].astype(str).str.strip()
    g = g[g["PLAYER"] == player].copy()
    if g.empty:
        return out

    # sort by date
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    g = g.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")

    # last game info based on market stat
    stat_col = STAT_COL.get(market)
    if stat_col in g.columns:
        g[stat_col] = pd.to_numeric(g[stat_col], errors="coerce")
        last5 = g.tail(5)
        out["last5_mean"] = float(last5[stat_col].mean()) if len(last5) else np.nan
        out["last5_std"]  = float(last5[stat_col].std(ddof=0)) if len(last5) else np.nan
        lg = g.tail(1)
        if not lg.empty:
            out["last_game"] = {
                "date": lg["GAME_DATE"].iloc[0].date().isoformat(),
                "team": str(lg["TEAM"].iloc[0]),
                "val": float(lg[stat_col].iloc[0]),
            }

    hit_col = HIT_COL.get(market)
    s = None
    if hit_col and hit_col in g.columns:
        s = pd.to_numeric(g[hit_col], errors="coerce").fillna(0).astype(int)
    elif stat_col in g.columns:  # fallback compute
        s = (pd.to_numeric(g[stat_col], errors="coerce").fillna(0) >= float(line)).astype(int)

    if s is not None:
        sN = s.tail(N)
        out["series"] = sN.tolist()
        out["n"] = int(len(sN))
        out["rate"] = float(sN.mean()) if len(sN) else np.nan

    return out

def sparkline(series_list, color="#2ecc71"):
    if not series_list:
        return None
    data = pd.DataFrame({"x": range(1, len(series_list)+1), "y": series_list})
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0,1])),
        color=alt.value(color)
    ).properties(height=30)

# ---------- Markets / Tabs ----------
markets = view["MARKET"].dropna().unique().tolist()
tabs = st.tabs([m for m in markets])

for tab, market in zip(tabs, markets):
    with tab:
        sub = view[view["MARKET"] == market].copy()
        if sub.empty:
            st.info("No data for this market.")
            continue

        # compute context (recent hit, volatility, last game) per row
        ctx_rows = []
        for _, r in sub.iterrows():
            line_val = float(pd.to_numeric(r.get("LINE", 0), errors="coerce"))
            ctxp = recent_hit_stats(gl, r["PLAYER"], market, line_val, N=5)
            ctx_rows.append({
                "PLAYER": r["PLAYER"],
                "hit_rate_recent": ctxp["rate"],
                "n_recent": ctxp["n"],
                "series_list": ctxp["series"],
                "last_game": ctxp["last_game"],
                "last5_mean": ctxp["last5_mean"],
                "last5_std": ctxp["last5_std"],
            })
        ctx_df = pd.DataFrame(ctx_rows)
        sub = sub.merge(ctx_df, on="PLAYER", how="left")

        # sorting
        if sort_by == "Prob Over (desc)":
            sub = sub.sort_values("FINAL_OVER_PROB", ascending=False)
        elif sort_by == "Recent Hit Rate":
            sub = sub.sort_values("hit_rate_recent", ascending=False, na_position="last")
        elif sort_by == "Line Edge (SEASON_VAL - LINE)":
            sub["line_edge"] = (
                pd.to_numeric(sub.get("SEASON_VAL", 0), errors="coerce") - 
                pd.to_numeric(sub.get("LINE", 0), errors="coerce")
            )
            sub = sub.sort_values("line_edge", ascending=False, na_position="last")
        elif sort_by == "Volatility (std, asc)":
            sub = sub.sort_values(sub["last5_std"].fillna(1e9), ascending=True)

        st.subheader(f"{market} · Top Overs")
        st.divider()

        for _, row in sub.head(10).iterrows():
            prim = row.get("PRIMARY_COLOR", "#333333")
            sec  = row.get("SECONDARY_COLOR", "#777777")

            with st.container(border=True):
                st.markdown(
                    f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>",
                    unsafe_allow_html=True
                )
                c1, c2, c3, c4 = st.columns([1.0, 3.0, 2.0, 1.6])

                with c1:
                    # headshot
                    if isinstance(row.get("PHOTO_URL",""), str) and row["PHOTO_URL"].startswith("http"):
                        st.image(row["PHOTO_URL"], width=80)
                    else:
                        st.image("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png", width=80)
                    # team logo
                    if isinstance(row.get("LOGO_URL",""), str) and row["LOGO_URL"].startswith("http"):
                        st.image(row["LOGO_URL"], width=40)

                with c2:
                    st.markdown(f"#### {row['PLAYER']}")
                    st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    inj = row.get("INJ_Status", "Active")
                    team_badge = f"<span style='background:{prim};color:white;padding:2px 6px;border-radius:6px;font-size:0.8rem'>{row.get('TEAM','')}</span>"
                    st.markdown(f"Team: {team_badge} &nbsp;|&nbsp; Injury: **{inj}**", unsafe_allow_html=True)
                    if isinstance(row.get("last_game"), dict) and row["last_game"]:
                        lg = row["last_game"]
                        st.caption(f"Last game ({lg['date']}): {lg['val']} {market} — Team: {lg['team']}")

                with c3:
                    st.metric("Prob. Over", row.get("FINAL_OVER_PROB_PCT", "—"))
                    hr = row.get("hit_rate_recent", np.nan)
                    n  = int(row.get("n_recent", 0) or 0)
                    st.markdown(
                        f"<div style='color:{color_for(hr)}'>Recent Hit Rate: {pct(hr)} ({n}g)</div>",
                        unsafe_allow_html=True
                    )
                    if "line_edge" in row:
                        try:
                            st.caption(f"Line edge: {float(row['line_edge']):+.2f}")
                        except Exception:
                            pass

                with c4:
                    chart = sparkline(row.get("series_list"), color=color_for(row.get("hit_rate_recent"), hi_good=True))
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.caption("No recent series")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

st.caption("Daily NBA Trends & Predictions — powered by your pipeline • Mobile-friendly • Free on Streamlit Cloud")