# dashboard_app.py — NBA Player Props Dashboard (no cache, GitHub hidden, logos fixed)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="NBA Player Props Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# ---------- Helpers ----------
ALT_TEAM_MAP = {"GS": "GSW", "NO": "NOP", "SA": "SAS", "NY": "NYK", "PHO": "PHX"}

def norm_team(x: str) -> str:
    if not isinstance(x, str): return ""
    x = x.strip().upper()
    return ALT_TEAM_MAP.get(x, x)

def pct(x): 
    return f"{x*100:.1f}%" if pd.notna(x) else "—"

def color_for(val, hi_good=True):
    if pd.isna(val): return "#999999"
    if hi_good:
        return "#2ecc71" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#e74c3c")
    else:
        return "#e74c3c" if val >= 0.6 else ("#f39c12" if val >= 0.4 else "#2ecc71")

# ---------- Data Loaders (no caching) ----------
def load_csv(path):
    try:
        df = pd.read_csv(path)
        st.toast(f"Loaded {path} ({len(df)} rows)", icon="✅")
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
    return preds, logos, heads, gl, ctx

# Initial load
preds, logos, heads, gl, ctx = load_all_data()

# Reload button
if st.sidebar.button("🔄 Reload Data"):
    preds, logos, heads, gl, ctx = load_all_data()
    st.toast("✅ Data reloaded successfully!")

# ---------- Normalize + Merge ----------
if not preds.empty:
    preds["PLAYER"] = preds["PLAYER"].astype(str).str.strip()
    preds["TEAM"] = preds["TEAM"].astype(str).map(norm_team)
    preds["PLAYER_NORM"] = preds["PLAYER"].str.lower().str.strip()

if not heads.empty:
    col_player = "PLAYER" if "PLAYER" in heads.columns else "player"
    col_url = "PHOTO_URL" if "PHOTO_URL" in heads.columns else "image_url"
    heads = heads.rename(columns={col_player: "PLAYER", col_url: "PHOTO_URL"})
    heads["PLAYER_NORM"] = heads["PLAYER"].astype(str).str.lower().str.strip()
    preds = preds.merge(heads[["PLAYER_NORM","PHOTO_URL"]], on="PLAYER_NORM", how="left")
else:
    preds["PHOTO_URL"] = ""

if not logos.empty:
    logos["TEAM"] = logos["TEAM"].astype(str).map(norm_team)
    preds["TEAM"] = preds["TEAM"].astype(str).map(norm_team)
    preds = preds.merge(logos, on="TEAM", how="left")
else:
    preds[["TEAM_FULL","LOGO_URL","PRIMARY_COLOR","SECONDARY_COLOR"]] = ["","","",""]

preds["PHOTO_URL"] = preds["PHOTO_URL"].fillna("https://cdn.nba.com/manage/2021/10/NBA_Silhouette.png")
preds["LOGO_URL"] = preds["LOGO_URL"].fillna("")
preds["PRIMARY_COLOR"] = preds["PRIMARY_COLOR"].fillna("#333333")
preds["SECONDARY_COLOR"] = preds["SECONDARY_COLOR"].fillna("#777777")

# ---------- Sidebar Filters ----------
st.sidebar.title("🔎 Filters")
teams = ["All Teams"] + sorted(preds["TEAM"].dropna().unique().tolist())
team_pick = st.sidebar.selectbox("Select Team", teams)
if team_pick != "All Teams":
    player_opts = ["All Players"] + sorted(preds.loc[preds["TEAM"]==team_pick,"PLAYER"].unique().tolist())
else:
    player_opts = ["All Players"] + sorted(preds["PLAYER"].unique().tolist())
player_pick = st.sidebar.selectbox("Select Player", player_opts)
sort_by = st.sidebar.selectbox("Sort by", ["Prob Over (desc)","Recent Hit Rate","Line Edge"])

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
    st.info("No player props found.")
    st.stop()

markets = view["MARKET"].dropna().unique().tolist()
tabs = st.tabs([m for m in markets])

# ---------- Main Display ----------
def sparkline(series, color="#2ecc71"):
    if not series or len(series)==0: return None
    data = pd.DataFrame({"x": range(1, len(series)+1), "y": series})
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0,1])),
        color=alt.value(color)
    ).properties(height=30)

for tab, market in zip(tabs, markets):
    with tab:
        sub = view[view["MARKET"]==market].copy()
        if sub.empty:
            st.info("No data for this market.")
            continue

        st.subheader(f"{market} · Top Overs")
        st.divider()

        for _, row in sub.head(10).iterrows():
            prim = row.get("PRIMARY_COLOR","#333333")
            sec  = row.get("SECONDARY_COLOR","#777777")

            with st.container(border=True):
                st.markdown(f"<div style='height:4px;background:linear-gradient(90deg,{prim},{sec});border-radius:4px;'></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1.0,3.0,2.0])

                with c1:
                    if isinstance(row.get("PHOTO_URL",""), str) and row["PHOTO_URL"].startswith("http"):
                        st.image(row["PHOTO_URL"], width=80)
                    if isinstance(row.get("LOGO_URL",""), str) and row["LOGO_URL"].startswith("http"):
                        st.image(row["LOGO_URL"], width=40)

                with c2:
                    st.markdown(f"#### {row['PLAYER']}")
                    st.markdown(f"**{row['PROP_NAME']} o{row['LINE']}**")
                    st.markdown(f"Team: `{row['TEAM']}` | Injury: {row.get('INJ_Status','Active')}")

                with c3:
                    st.metric("Prob. Over", row.get("FINAL_OVER_PROB_PCT","—"))

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

st.caption("Daily NBA Trends & Predictions — powered by your pipeline • Mobile-friendly • Free on Streamlit Cloud")