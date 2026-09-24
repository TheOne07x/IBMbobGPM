"""
app.py — Global Precipitation Measurement (GPM) Rainfall Engine
================================================================
Interactive Streamlit dashboard.

Prerequisites:
    1. Run `python train_model.py` once to generate model_artifacts/
    2. pip install streamlit plotly pandas numpy scikit-learn joblib

Launch:
    streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPM Rainfall Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME — dark telemetry, no unclosed divs, no external font imports
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
.main .block-container {
    background-color: #06090E !important;
    color: #E2E8F0 !important;
}

header[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stSidebar"] {
    background-color: #0B0F17 !important;
}

/* ── KPI cards ─────────────────────────────────────────────────────────────── */
.kpi-card {
    background: #0D1520;
    border: 1px solid #1E2D42;
    border-radius: 10px;
    padding: 18px 20px 14px;
}
.kpi-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5A7A9A;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #5A7A9A;
    margin-top: 4px;
}
.blue  { color: #38BDF8; }
.green { color: #10B981; }
.red   { color: #EF4444; }
.gold  { color: #F59E0B; }

/* ── section cards ──────────────────────────────────────────────────────────── */
.panel {
    background: #0D1520;
    border: 1px solid #1E2D42;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 6px;
}
.section-title {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 14px;
}

/* ── status badge ───────────────────────────────────────────────────────────── */
.badge-low    { background:rgba(239,68,68,0.15);   color:#EF4444; border:1px solid rgba(239,68,68,0.4);   border-radius:20px; padding:4px 14px; font-weight:600; font-size:0.85rem; }
.badge-decent { background:rgba(16,185,129,0.15);  color:#10B981; border:1px solid rgba(16,185,129,0.4);  border-radius:20px; padding:4px 14px; font-weight:600; font-size:0.85rem; }
.badge-heavy  { background:rgba(56,189,248,0.15);  color:#38BDF8; border:1px solid rgba(56,189,248,0.4);  border-radius:20px; padding:4px 14px; font-weight:600; font-size:0.85rem; }

/* ── impact briefing list ───────────────────────────────────────────────────── */
.impact-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1A2535;
    font-size: 0.84rem;
    color: #CBD5E1;
    line-height: 1.5;
}
.impact-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

/* ── sliders & selects ──────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div {
    background-color: #0D1520 !important;
    border-color: #1E2D42 !important;
    color: #F8FAFC !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "model_artifacts"
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "rainfall_model.pkl")
META_PATH     = os.path.join(ARTIFACTS_DIR, "feature_meta.pkl")

STATUS_COLORS = {
    "Low (Water Difficulty)": "#EF4444",
    "Decent (Normal)"       : "#10B981",
    "Heavy (Surplus)"       : "#38BDF8",
}

ANCHOR_COUNTRIES = [
    "India", "USA", "China", "Brazil", "Australia",
    "Germany", "South Africa", "UK", "Canada",
    "Russia", "France", "Japan", "Indonesia", "Mexico", "Argentina",
]

# Rough continent mapping (covers dataset countries)
CONTINENT_MAP = {
    "India": "Asia", "China": "Asia", "Japan": "Asia", "Indonesia": "Asia",
    "USA": "Americas", "Canada": "Americas", "Brazil": "Americas",
    "Argentina": "Americas", "Mexico": "Americas",
    "UK": "Europe", "Germany": "Europe", "France": "Europe", "Russia": "Europe",
    "Australia": "Oceania",
    "South Africa": "Africa",
}

BG        = "rgba(0,0,0,0)"   # transparent for plotly panels
GRID_COL  = "#1A2535"
FONT_COL  = "#94A3B8"


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising ML core …")
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(META_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict_for_country(
    model, meta, country: str,
    target_year: int,
    custom_temp_delta: float = 0.0,
    custom_extreme: int | None = None,
) -> tuple[float, str]:
    """
    Forecast rainfall for a given country / scenario year.
    For future years beyond the dataset, temperature and CO₂ are
    extrapolated at modest linear rates.
    """
    df = meta["processed_df"]
    c_df = df[df["Country"] == country].sort_values("Year")
    if c_df.empty:
        return 1200.0, "Decent (Normal)"

    latest       = c_df.iloc[-1]
    last_year    = int(latest["Year"])
    years_ahead  = max(0, target_year - last_year)

    base_temp    = float(latest["Avg Temperature (°C)"])
    base_co2     = float(latest["CO2 Emissions (Tons/Capita)"])
    base_ext     = float(latest["Extreme Weather Events"]) if custom_extreme is None else float(custom_extreme)
    lag1         = float(latest.get("Rainfall_lag1",  latest["Rainfall (mm)"]))
    lag2         = float(latest.get("Rainfall_lag2",  latest["Rainfall (mm)"]))
    roll3        = float(latest.get("Rainfall_roll3", latest["Rainfall (mm)"]))

    sim_temp = base_temp + (years_ahead * 0.04) + custom_temp_delta
    sim_co2  = base_co2  + (years_ahead * 0.08)

    features  = meta["feature_cols"]
    input_row = pd.DataFrame([{
        "Avg Temperature (°C)"        : sim_temp,
        "CO2 Emissions (Tons/Capita)" : sim_co2,
        "Extreme Weather Events"      : base_ext,
        "Rainfall_lag1"               : lag1,
        "Rainfall_lag2"               : lag2,
        "Rainfall_roll3"              : roll3,
    }])[features]

    pred_rain = float(model.predict(input_row)[0])

    stats = meta["country_stats"]
    row_s = stats[stats["Country"] == country]
    c_mean = float(row_s["country_mean"].values[0]) if not row_s.empty else df["Rainfall (mm)"].mean()
    c_std  = float(row_s["country_std"].values[0])  if not row_s.empty else df["Rainfall (mm)"].std()
    c_std  = max(c_std, 1.0)

    z = (pred_rain - c_mean) / c_std
    if z < meta["zscore_low"]:
        status = "Low (Water Difficulty)"
    elif z > meta["zscore_high"]:
        status = "Heavy (Surplus)"
    else:
        status = "Decent (Normal)"

    return pred_rain, status


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK LAYOUT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def dark_layout(fig, height: int = 320, title: str = "", margin=None):
    m = margin or dict(l=10, r=10, t=30 if title else 14, b=10)
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FONT_COL, size=11),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=height,
        margin=m,
        title_text=title,
        title_font=dict(color="#38BDF8", size=13),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STATUS BADGE HTML
# ─────────────────────────────────────────────────────────────────────────────
def status_badge(status: str) -> str:
    cls = {
        "Low (Water Difficulty)": "badge-low",
        "Decent (Normal)"       : "badge-decent",
        "Heavy (Surplus)"       : "badge-heavy",
    }.get(status, "badge-decent")
    return f"<span class='{cls}'>{status}</span>"


# ─────────────────────────────────────────────────────────────────────────────
# DERIVE COUNTRY TELEMETRY STATS
# ─────────────────────────────────────────────────────────────────────────────
def get_country_stats(df: pd.DataFrame, country: str) -> dict:
    c_df = df[df["Country"] == country].sort_values("Year").copy()
    if c_df.empty:
        return {}

    mean_rain = c_df["Rainfall (mm)"].mean()
    mean_temp = c_df["Avg Temperature (°C)"].mean()
    mean_ext  = c_df["Extreme Weather Events"].mean()

    # 20-year warming trend (°C / decade) via linear regression on available data
    if len(c_df) >= 2:
        years_arr = c_df["Year"].values.astype(float)
        temps_arr = c_df["Avg Temperature (°C)"].values.astype(float)
        coeffs    = np.polyfit(years_arr, temps_arr, 1)
        trend_per_decade = round(coeffs[0] * 10, 2)
    else:
        trend_per_decade = 0.0

    # Deficit frequency → vulnerability index
    if "Water_Status" in c_df.columns:
        n_low    = (c_df["Water_Status"] == "Low (Water Difficulty)").sum()
        frac_low = n_low / max(len(c_df), 1)
    else:
        frac_low = 0.0

    if frac_low > 0.5:
        vuln_label = "High Risk"
        vuln_color = "#EF4444"
    elif frac_low > 0.25:
        vuln_label = "Moderate Risk"
        vuln_color = "#F59E0B"
    else:
        vuln_label = "Low Risk"
        vuln_color = "#10B981"

    return {
        "mean_rain"         : mean_rain,
        "mean_temp"         : mean_temp,
        "trend_per_decade"  : trend_per_decade,
        "mean_ext"          : mean_ext,
        "vuln_label"        : vuln_label,
        "vuln_color"        : vuln_color,
        "frac_low"          : frac_low,
        "c_df"              : c_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMPACT BRIEFING LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def build_impact_briefing(
    country: str,
    sim_rain: float,
    hist_mean: float,
    status: str,
    temp_delta: float,
    extreme_events: int,
) -> list[dict]:
    delta_pct  = ((sim_rain - hist_mean) / max(hist_mean, 1)) * 100
    is_deficit = status == "Low (Water Difficulty)"
    is_surplus = status == "Heavy (Surplus)"

    briefing = []

    # Agricultural risk
    if is_deficit:
        briefing.append({
            "icon": "🌾",
            "text": (
                f"Agricultural drought risk is <b>elevated</b>. A {abs(delta_pct):.0f}% "
                f"rainfall deficit below {country}'s historical norm signals potential crop failure "
                f"in rain-fed farming zones."
            ),
        })
    elif is_surplus:
        briefing.append({
            "icon": "🌾",
            "text": (
                f"Agricultural sector faces <b>flood-induced crop damage</b> risk. "
                f"A {delta_pct:.0f}% surplus can inundate low-lying farmland and disrupt harvest cycles."
            ),
        })
    else:
        briefing.append({
            "icon": "🌾",
            "text": (
                f"Agricultural conditions remain <b>within normal operating range</b>. "
                f"Rainfall is {delta_pct:+.0f}% relative to the historical baseline — "
                f"seasonal irrigation planning can proceed as standard."
            ),
        })

    # Monsoon / seasonal variability
    if temp_delta >= 2.0:
        briefing.append({
            "icon": "🌀",
            "text": (
                f"A +{temp_delta:.1f} °C thermal anomaly is likely to <b>disrupt monsoon onset timing</b> "
                f"by 1–3 weeks, creating unpredictable intra-seasonal dry spells and intense bursts."
            ),
        })
    elif temp_delta >= 0.8:
        briefing.append({
            "icon": "🌀",
            "text": (
                f"Moderate thermal anomaly (+{temp_delta:.1f} °C) may cause <b>minor monsoon variability</b>. "
                f"Expect marginal shifts in peak wet-season intensity."
            ),
        })
    else:
        briefing.append({
            "icon": "🌀",
            "text": (
                "Seasonal monsoon patterns show <b>minimal thermal disruption</b> at the simulated "
                "temperature anomaly. Precipitation timing is expected to remain largely stable."
            ),
        })

    # Flood defence
    if is_surplus and extreme_events >= 8:
        briefing.append({
            "icon": "🏗️",
            "text": (
                f"<b>Flood defence infrastructure is under high stress</b>. "
                f"{extreme_events} extreme events combined with a heavy-surplus rainfall regime "
                f"demands immediate review of drainage capacity and embankment readiness."
            ),
        })
    elif is_deficit and extreme_events >= 10:
        briefing.append({
            "icon": "🏗️",
            "text": (
                f"Despite a rainfall deficit, <b>high extreme-event frequency ({extreme_events}/yr)</b> "
                f"suggests flash-flood and intense localised downpour risk — "
                f"retention infrastructure remains critical."
            ),
        })
    else:
        briefing.append({
            "icon": "🏗️",
            "text": (
                f"Flood defence systems face <b>standard operational load</b> under this scenario. "
                f"Routine maintenance protocols are sufficient at {extreme_events} extreme events per year."
            ),
        })

    # Groundwater / reservoir outlook
    if is_deficit:
        briefing.append({
            "icon": "💧",
            "text": (
                "Groundwater recharge rates are projected to fall <b>below replenishment thresholds</b>. "
                "Reservoir draw-down will accelerate; water rationing advisories are warranted."
            ),
        })
    else:
        briefing.append({
            "icon": "💧",
            "text": (
                "Groundwater and reservoir levels are projected to <b>remain adequate</b> under this scenario. "
                "Surplus conditions may even offer managed aquifer recharge opportunities."
            ),
        })

    return briefing


# ─────────────────────────────────────────────────────────────────────────────
# ══ SECTION: HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(
            "<h1 style='"
            "margin:0 0 8px;"
            "font-weight:800;"
            "font-size:2.6rem;"
            "color:#FFFFFF;"
            "letter-spacing:-0.02em;"
            "line-height:1.1;"
            "text-transform:uppercase;"
            "'>"
            "🛰️ Global Precipitation Measurement Rainfall Engine"
            "</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='"
            "margin:0;"
            "color:#94A3B8;"
            "font-size:1.0rem;"
            "letter-spacing:0.01em;"
            "line-height:1.5;"
            "'>"
            "Planetary Hydro-Climatic Intelligence &nbsp;·&nbsp; "
            "Multi-Decade Satellite Observation &amp; Gradient Boosting Predictive Model"
            "</p>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            "<div style='text-align:right; padding-top:18px;'>"
            "<span style='"
            "background:rgba(16,185,129,0.12);"
            "color:#10B981;"
            "padding:7px 16px;"
            "border-radius:20px;"
            "font-size:0.78rem;"
            "font-weight:600;"
            "border:1px solid rgba(16,185,129,0.4);"
            "letter-spacing:0.07em;"
            "white-space:nowrap;"
            "'>"
            "● AI FORECAST ONLINE"
            "</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ SECTION: YEAR + MODE CONTROL BAR
# ─────────────────────────────────────────────────────────────────────────────
def render_control_bar(df: pd.DataFrame) -> tuple[int, bool]:
    with st.container():
        c1, c2, c3 = st.columns([3, 1, 2])
        with c1:
            selected_year = st.slider(
                "📅 TARGET YEAR",
                min_value=2000, max_value=2030,
                value=int(df["Year"].max()), step=1,
            )
        is_future = selected_year > int(df["Year"].max())
        badge_col = "#38BDF8" if is_future else "#10B981"
        mode_text = "AI FORECAST HORIZON" if is_future else "HISTORICAL ARCHIVE"
        ctx_text  = (
            f"Synthesising projected climate drivers for {selected_year}"
            if is_future
            else f"Displaying recorded climate indices for {selected_year}"
        )
        with c2:
            st.markdown(
                f"<div style='padding-top:26px'>"
                f"<div class='kpi-label'>MODE</div>"
                f"<div style='color:{badge_col}; font-weight:600; font-size:0.82rem; letter-spacing:0.05em;'>{mode_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div style='padding-top:26px'>"
                f"<div class='kpi-label'>SIMULATION CONTEXT</div>"
                f"<div style='color:#CBD5E1; font-size:0.82rem; margin-top:4px;'>{ctx_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    return selected_year, is_future


# ─────────────────────────────────────────────────────────────────────────────
# ══ SECTION: GLOBAL KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(map_df: pd.DataFrame):
    low_cnt    = int((map_df["Water_Status"] == "Low (Water Difficulty)").sum())
    decent_cnt = int((map_df["Water_Status"] == "Decent (Normal)").sum())
    heavy_cnt  = int((map_df["Water_Status"] == "Heavy (Surplus)").sum())
    mean_val   = map_df["Rainfall_mm"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Global Mean Rainfall</div>"
            f"<div class='kpi-value blue'>{mean_val:,.0f} <span style='font-size:1rem;color:#5A7A9A'>mm</span></div>"
            f"<div class='kpi-sub'>Across all tracked nations</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Water Difficulty Nations</div>"
            f"<div class='kpi-value red'>{low_cnt} <span style='font-size:1rem;color:#5A7A9A'>nations</span></div>"
            f"<div class='kpi-sub'>Active deficit / drought signal</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Stable Water Nations</div>"
            f"<div class='kpi-value green'>{decent_cnt} <span style='font-size:1rem;color:#5A7A9A'>nations</span></div>"
            f"<div class='kpi-sub'>Normal precipitation range</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Surplus / Flood Watch</div>"
            f"<div class='kpi-value' style='color:#38BDF8'>{heavy_cnt} <span style='font-size:1rem;color:#5A7A9A'>nations</span></div>"
            f"<div class='kpi-sub'>Heavy rainfall surplus</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ SECTION: WORLD MAP
# ─────────────────────────────────────────────────────────────────────────────
def render_world_map(map_df: pd.DataFrame, selected_year: int):
    st.markdown(
        "<div class='section-title'>🌐 GLOBAL PRECIPITATION & WATER ANOMALY MATRIX</div>",
        unsafe_allow_html=True,
    )
    fig = px.choropleth(
        map_df,
        locations="Country",
        locationmode="country names",
        color="Water_Status",
        color_discrete_map=STATUS_COLORS,
        hover_name="Country",
        hover_data={"Rainfall_mm": ":,.0f", "Water_Status": True},
        category_orders={"Water_Status": list(STATUS_COLORS.keys())},
        title=f"Water Availability Status — {selected_year}",
    )
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#1E2D42",
            showland=True,
            landcolor="#0D1520",
            showocean=True,
            oceancolor="#06090E",
            projection_type="natural earth",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(13,21,32,0.9)",
            bordercolor="#1E2D42",
            borderwidth=1,
        ),
        title_font=dict(color="#38BDF8", size=14),
        font=dict(color=FONT_COL),
        height=460,
        margin=dict(l=0, r=0, t=42, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ SECTION: SOVEREIGN CLIMATE DOSSIER
# ─────────────────────────────────────────────────────────────────────────────
def render_sovereign_dossier(df: pd.DataFrame, model, meta, selected_year: int, is_future: bool):
    st.markdown(
        "<h2 style='font-size:1.1rem; font-weight:700; color:#F8FAFC; margin:0 0 4px;'>"
        "🏛️ Sovereign Climate Intelligence &amp; Water Security Dossier"
        "</h2>"
        "<p style='color:#5A7A9A; font-size:0.82rem; margin:0 0 18px;'>"
        "Deep-dive analytics, precipitation trajectory, and climate scenario simulation for any nation."
        "</p>",
        unsafe_allow_html=True,
    )

    all_countries = sorted(df["Country"].unique().tolist())
    anchor_first  = [c for c in ANCHOR_COUNTRIES if c in all_countries]
    rest          = [c for c in all_countries if c not in anchor_first]
    ordered       = anchor_first + rest

    country = st.selectbox(
        "🌐 Select Sovereign Nation",
        options=ordered,
        index=ordered.index("India") if "India" in ordered else 0,
        key="dossier_country",
    )

    stats = get_country_stats(df, country)
    if not stats:
        st.warning(f"No data available for {country}.")
        return

    c_df       = stats["c_df"]
    mean_rain  = stats["mean_rain"]

    # ── A. Telemetry Cards ───────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    trend_sign = "+" if stats["trend_per_decade"] >= 0 else ""
    with t1:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Historical Baseline Rainfall</div>"
            f"<div class='kpi-value blue'>{mean_rain:,.0f} <span style='font-size:0.9rem;color:#5A7A9A'>mm</span></div>"
            f"<div class='kpi-sub'>All-time country mean</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Mean Temp &amp; Warming Trend</div>"
            f"<div class='kpi-value' style='color:#F59E0B'>{stats['mean_temp']:.1f}°C "
            f"<span style='font-size:0.9rem;color:#5A7A9A'>avg</span></div>"
            f"<div class='kpi-sub'>Warming: {trend_sign}{stats['trend_per_decade']:.2f} °C / decade</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Avg Extreme Weather / Year</div>"
            f"<div class='kpi-value red'>{stats['mean_ext']:.1f} <span style='font-size:0.9rem;color:#5A7A9A'>events</span></div>"
            f"<div class='kpi-sub'>Annual mean frequency</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with t4:
        vuln_col = stats['vuln_color']
        vuln_lbl = stats['vuln_label']
        frac_pct = stats['frac_low'] * 100
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Water Stress / Vulnerability</div>"
            f"<div class='kpi-value' style='color:{vuln_col}'>{vuln_lbl}</div>"
            f"<div class='kpi-sub'>Deficit frequency: {frac_pct:.0f}% of years</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── B. Expanded Analytics: 2-column layout ───────────────────────────────
    col_left, col_right = st.columns([1.05, 0.95], gap="medium")

    # ╔══ LEFT COLUMN — Analytical Charts ═════════════════════════════════════
    with col_left:

        # ── B1: Precipitation Trajectory ─────────────────────────────────────
        st.markdown("<div class='section-title'>PRECIPITATION TRAJECTORY</div>", unsafe_allow_html=True)

        yearly = (
            c_df.groupby("Year")
                .agg(Rainfall=("Rainfall (mm)", "mean"), Temp=("Avg Temperature (°C)", "mean"))
                .reset_index()
                .sort_values("Year")
        )
        yearly["Roll3"] = yearly["Rainfall"].rolling(3, min_periods=1).mean()
        baseline_val    = float(yearly["Rainfall"].mean())

        fig_traj = go.Figure()

        # historical bars (subtle area)
        fig_traj.add_trace(go.Scatter(
            x=yearly["Year"], y=yearly["Rainfall"],
            mode="lines+markers",
            name="Recorded Rainfall",
            line=dict(color="#0EA5E9", width=1.6),
            marker=dict(size=5, color="#38BDF8"),
            fill="tozeroy",
            fillcolor="rgba(14,165,233,0.07)",
        ))
        # 3-yr rolling average
        fig_traj.add_trace(go.Scatter(
            x=yearly["Year"], y=yearly["Roll3"],
            mode="lines",
            name="3-yr Rolling Avg",
            line=dict(color="#F59E0B", width=2, dash="dot"),
        ))
        # baseline reference
        fig_traj.add_hline(
            y=baseline_val,
            line_dash="dash", line_color="#5A7A9A", line_width=1.2,
            annotation_text=f"Baseline {baseline_val:,.0f} mm",
            annotation_font_color="#5A7A9A",
            annotation_font_size=10,
        )
        # AI forecast diamond
        max_hist_year = int(yearly["Year"].max())
        fc_year       = max(selected_year, max_hist_year)
        fc_rain, fc_status = predict_for_country(model, meta, country, fc_year)
        fc_color      = "#EF4444" if fc_rain < baseline_val * 0.9 else "#10B981"

        fig_traj.add_trace(go.Scatter(
            x=[fc_year], y=[fc_rain],
            mode="markers",
            name=f"AI Forecast {fc_year}",
            marker=dict(size=13, color=fc_color, symbol="diamond",
                        line=dict(color="#F8FAFC", width=1.5)),
        ))

        dark_layout(fig_traj, height=300,
                    title=f"Rainfall Trajectory — {country} (2000 → {fc_year})")
        fig_traj.update_xaxes(title_text="Year")
        fig_traj.update_yaxes(title_text="Rainfall (mm)")
        st.plotly_chart(fig_traj, use_container_width=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── B2: Temperature vs Extreme Weather Overlay ────────────────────────
        st.markdown(
            "<div class='section-title'>TEMPERATURE vs EXTREME WEATHER FREQUENCY</div>",
            unsafe_allow_html=True,
        )

        ext_yearly = (
            c_df.groupby("Year")
                .agg(Temp=("Avg Temperature (°C)", "mean"),
                     Ext=("Extreme Weather Events", "mean"))
                .reset_index()
                .sort_values("Year")
        )

        fig_ov = go.Figure()
        fig_ov.add_trace(go.Bar(
            x=ext_yearly["Year"], y=ext_yearly["Ext"],
            name="Extreme Events",
            marker_color="rgba(239,68,68,0.5)",
            yaxis="y",
        ))
        fig_ov.add_trace(go.Scatter(
            x=ext_yearly["Year"], y=ext_yearly["Temp"],
            name="Avg Temperature (°C)",
            mode="lines+markers",
            line=dict(color="#F59E0B", width=2),
            marker=dict(size=5),
            yaxis="y2",
        ))
        fig_ov.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=FONT_COL, size=11),
            xaxis=dict(gridcolor=GRID_COL, title="Year"),
            yaxis=dict(
                gridcolor=GRID_COL,
                title=dict(text="Extreme Events", font=dict(color="#EF4444")),
            ),
            yaxis2=dict(
                title=dict(text="Avg Temp (°C)", font=dict(color="#F59E0B")),
                overlaying="y", side="right",
                gridcolor="rgba(0,0,0,0)",
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            barmode="overlay",
            height=270,
            margin=dict(l=10, r=40, t=14, b=10),
        )
        st.plotly_chart(fig_ov, use_container_width=True)

    # ╔══ RIGHT COLUMN — What-If & Impact Engine ═══════════════════════════════
    with col_right:
        st.markdown(
            "<div class='section-title'>INTERACTIVE CLIMATE SCENARIO SIMULATOR</div>",
            unsafe_allow_html=True,
        )

        temp_delta     = st.slider(
            "🌡️ Simulated Temp Anomaly (+°C)",
            min_value=0.0, max_value=5.0, value=1.5, step=0.1,
            key="sim_temp",
        )
        extreme_events = st.slider(
            "⛈️ Simulated Extreme Weather Events / Year",
            min_value=0, max_value=20, value=6, step=1,
            key="sim_ext",
        )

        sim_rain, sim_status = predict_for_country(
            model, meta, country, selected_year,
            custom_temp_delta=temp_delta,
            custom_extreme=extreme_events,
        )

        delta_pct  = ((sim_rain - mean_rain) / max(mean_rain, 1)) * 100
        stat_col   = STATUS_COLORS.get(sim_status, "#38BDF8")
        delta_sign = "+" if delta_pct >= 0 else ""
        delta_col  = "#10B981" if delta_pct >= 0 else "#EF4444"

        # Simulation result card
        st.markdown(
            f"<div style='background:#0D1520; border:1px solid {stat_col}44; "
            f"border-radius:12px; padding:20px 22px; margin:14px 0 16px;'>"
            f"<div style='color:#5A7A9A; font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;'>Simulated Precipitation Outcome</div>"
            f"<div style='font-size:2.6rem; font-weight:800; color:#F8FAFC; line-height:1; margin-bottom:8px;'>"
            f"{sim_rain:,.0f} <span style='font-size:1rem; color:#5A7A9A; font-weight:400;'>mm</span>"
            f"</div>"
            f"<div style='margin-bottom:10px;'>{status_badge(sim_status)}</div>"
            f"<div style='font-size:0.82rem; color:{delta_col}; font-weight:600;'>"
            f"{delta_sign}{delta_pct:.1f}% vs historical baseline ({mean_rain:,.0f} mm)"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Input summary strip
        st.markdown(
            f"<div style='background:#0B1119; border:1px solid #1E2D42; border-radius:8px; "
            f"padding:10px 14px; font-size:0.78rem; color:#5A7A9A; margin-bottom:16px; "
            f"display:flex; gap:20px;'>"
            f"<span>🌡️ Base temp: <b style='color:#CBD5E1'>{stats['mean_temp']:.1f} °C</b></span>"
            f"&nbsp;·&nbsp;"
            f"<span>➕ Anomaly: <b style='color:#F59E0B'>+{temp_delta:.1f} °C</b></span>"
            f"&nbsp;·&nbsp;"
            f"<span>⛈️ Events: <b style='color:#EF4444'>{extreme_events}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Impact Briefing ───────────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>IMPACT BRIEFING</div>",
            unsafe_allow_html=True,
        )

        briefing = build_impact_briefing(
            country, sim_rain, mean_rain, sim_status, temp_delta, extreme_events
        )
        items_html = ""
        for item in briefing:
            items_html += (
                f"<div class='impact-item'>"
                f"<span class='impact-icon'>{item['icon']}</span>"
                f"<span>{item['text']}</span>"
                f"</div>"
            )

        st.markdown(
            f"<div style='background:#0D1520; border:1px solid #1E2D42; border-radius:10px; "
            f"padding:14px 18px;'>{items_html}</div>",
            unsafe_allow_html=True,
        )

        # ── Mini forecast gauge ───────────────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>RAINFALL GAUGE</div>",
            unsafe_allow_html=True,
        )
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sim_rain,
            delta={
                "reference": mean_rain,
                "suffix": " mm",
                "increasing": {"color": "#10B981"},
                "decreasing": {"color": "#EF4444"},
            },
            number={"suffix": " mm", "font": {"color": "#F8FAFC", "size": 30}},
            gauge={
                "axis"      : {"range": [0, max(4000, sim_rain * 1.3)],
                               "tickcolor": FONT_COL, "tickfont": {"size": 9}},
                "bar"       : {"color": stat_col},
                "bgcolor"   : "#0D1520",
                "bordercolor": "#1E2D42",
                "steps"     : [
                    {"range": [0, mean_rain * 0.7],              "color": "#0B1119"},
                    {"range": [mean_rain * 0.7, mean_rain * 1.3], "color": "#0D1520"},
                    {"range": [mean_rain * 1.3, 4500],           "color": "#0B1119"},
                ],
                "threshold" : {
                    "line" : {"color": "#F8FAFC", "width": 2},
                    "thickness": 0.75,
                    "value": mean_rain,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor=BG,
            font=dict(color=FONT_COL),
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ SECTION: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def render_data_explorer(df: pd.DataFrame):
    with st.expander("📋 Raw Data Explorer", expanded=False):
        display_cols = [
            "Year", "Country", "Rainfall (mm)", "Water_Status",
            "Avg Temperature (°C)", "CO2 Emissions (Tons/Capita)",
            "Extreme Weather Events", "Sea Level Rise (mm)",
        ]
        cols = [c for c in display_cols if c in df.columns]
        st.markdown(
            f"<span style='color:#5A7A9A; font-size:0.8rem;'>"
            f"{len(df):,} rows · {df['Country'].nunique()} countries · {df['Year'].nunique()} years"
            f"</span>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            df[cols].sort_values(["Country", "Year"]),
            use_container_width=True,
            height=320,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══ MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model, meta = load_artifacts()
    if model is None or meta is None:
        st.error(
            "⚠️ Model artifacts not found.  \n"
            "Run **`python train_model.py`** first, then refresh."
        )
        st.code("python train_model.py", language="bash")
        st.stop()

    df = meta["processed_df"].copy()
    df["Continent"] = df["Country"].map(CONTINENT_MAP).fillna("Other")

    # ── HEADER ────────────────────────────────────────────────────────────────
    render_header()

    # ── CONTROL BAR ──────────────────────────────────────────────────────────
    selected_year, is_future = render_control_bar(df)

    # ── BUILD MAP DATA ────────────────────────────────────────────────────────
    all_countries = sorted(df["Country"].unique().tolist())
    records       = []
    for c in all_countries:
        c_sub = df[(df["Country"] == c) & (df["Year"] == selected_year)]
        if not c_sub.empty and not is_future:
            val    = float(c_sub["Rainfall (mm)"].mean())
            status = str(c_sub["Water_Status"].mode()[0])
        else:
            val, status = predict_for_country(model, meta, c, selected_year)
        records.append({"Country": c, "Rainfall_mm": val, "Water_Status": status})
    map_df = pd.DataFrame(records)

    # ── GLOBAL KPIs ───────────────────────────────────────────────────────────
    render_kpis(map_df)

    # ── WORLD MAP ─────────────────────────────────────────────────────────────
    render_world_map(map_df, selected_year)

    # ── DIVIDER ───────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:none; border-top:1px solid #1E2D42; margin:4px 0 24px;'>",
        unsafe_allow_html=True,
    )

    # ── SOVEREIGN DOSSIER ─────────────────────────────────────────────────────
    render_sovereign_dossier(df, model, meta, selected_year, is_future)

    # ── DATA EXPLORER ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    render_data_explorer(df)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-top:48px; padding-top:16px; border-top:1px solid #1E2D42; "
        "text-align:center; color:#2A3B50; font-size:0.72rem; letter-spacing:0.04em;'>"
        "GPM RAINFALL ENGINE &nbsp;·&nbsp; Gradient Boosting ML &nbsp;·&nbsp; "
        "Streamlit + Plotly &nbsp;·&nbsp; Climate Intelligence Platform"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
