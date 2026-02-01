"""
ADAS Risk Heatmap Generator (Streamlit)
Author: Suresh Konar
Purpose: OEM-style ADAS risk heatmap with AEB/LDW overlay, road-type weighting & analysis
"""

# =========================
# STREAMLIT CONFIG
# =========================
import streamlit as st
st.set_page_config(
    page_title="RoadSentinel – ADAS Heatmaps & Analysis",
    page_icon="🚗",
    layout="wide"
)

# =========================
# IMPORTS
# =========================
import os
import pandas as pd
import numpy as np
import kagglehub
import pydeck as pdk
import requests
import plotly.express as px

# =========================
# TOOL DESCRIPTION & DISCLAIMER
# =========================
st.title("🚨 RoadSentinel – ADAS Heatmaps & Analysis")
st.markdown("""
**Description:**  
This tool visualizes India-focused ADAS risk intelligence using road accident data, weather effects, and road-type weighting.  
It includes **ADAS Risk Heatmap**, **AEB (Autonomous Emergency Braking) Risk Zones**, and **LDW (Lane Departure Warning) Risk Zones**.  

**How to interpret:**  
- **Red dots / high heatmap intensity:** High risk  
- **Yellow / orange:** Medium risk  
- **Green / low intensity:** Low risk  

**Disclaimer:**  
Data sources are publicly available datasets. Weather and road-type weighting are illustrative and may differ from OEM-calibrated values.
""")

# =========================
# SIDEBAR - Expanded State / City Selection
# =========================
st.sidebar.header("⚙️ Controls")

state_city_map = {
    "All India": ["All Cities"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Karnataka": ["Bengaluru", "Mysore", "Mangalore", "Hubli", "Belgaum"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Delhi": ["Delhi"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Agra"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode"],
    "Andhra Pradesh": ["Vijayawada", "Visakhapatnam", "Guntur"],
    "Telangana": ["Hyderabad", "Warangal"],
    "Punjab": ["Amritsar", "Ludhiana", "Jalandhar"]
}

selected_state = st.sidebar.selectbox("Select State", list(state_city_map.keys()))

if selected_state == "All India":
    selected_city = "All Cities"
else:
    selected_city = st.sidebar.selectbox("Select City", state_city_map[selected_state])

grid_size = st.sidebar.slider(
    "Grid Size (degrees)",
    0.005, 0.05, 0.01, step=0.005
)

enable_weather = st.sidebar.checkbox(
    "Enable Weather Risk (Rain / Fog)", True
)

enable_road_weight = st.sidebar.checkbox(
    "Enable Road-Type Weighting (NH/SH/Urban)", True
)

run = st.sidebar.button("▶ Run Pipeline")

# =========================
# UTILS
# =========================
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

# =========================
# DATA LOAD
# =========================
@st.cache_data
def load_data():
    path = kagglehub.dataset_download(
        "data125661/india-road-accident-dataset"
    )
    csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(path, csv))

    df.columns = [c.lower() for c in df.columns]
    # lat = [c for c in df.columns if "lat" in c][0]
    # lon = [c for c in df.columns if "lon" in c][0]

    print("Columns in CSV:", df.columns.tolist())  # For debug

    # Try to find latitude/longitude columns
    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower()]

    if not lat_candidates or not lon_candidates:
        raise ValueError("Latitude or longitude column not found in dataset")

    lat = lat_candidates[0]
    lon = lon_candidates[0]


    # Optional road type (demo)
    if "road_type" not in df.columns:
        np.random.seed(0)
        df["road_type"] = np.random.choice(["NH","SH","Urban"], size=len(df))

    df = df[[lat, lon, "road_type"]].dropna()
    df = df[df[lat].between(6, 36) & df[lon].between(68, 98)]

    df.rename(columns={lat: "lat", lon: "lon"}, inplace=True)
    return df

# =========================
# WEATHER RISK
# =========================
def weather_risk(lat, lon):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=precipitation_probability,relativehumidity_2m,temperature_2m"
            "&forecast_days=1"
        )
        d = requests.get(url, timeout=10).json()
        rain = max(d["hourly"]["precipitation_probability"]) / 100
        hum = max(d["hourly"]["relativehumidity_2m"])
        temp = min(d["hourly"]["temperature_2m"])
        fog = 0.7 if hum > 85 and temp < 20 else 0.2
        return 1 + 0.15 * (0.6 * rain + 0.4 * fog)
    except:
        return 1.0

# =========================
# PIPELINE
# =========================
if run:
    with st.spinner("Loading accident data..."):
        df = load_data()

        # Filter city or All India
        if selected_state != "All India":
            city_center = {
                "Mumbai": (19.076, 72.877), "Pune": (18.520, 73.856), "Nagpur": (21.145, 79.088),
                "Nashik": (19.997, 73.789), "Aurangabad": (19.876, 75.343),
                "Bengaluru": (12.971, 77.594), "Mysore": (12.295, 76.639), "Mangalore": (12.915, 74.856),
                "Hubli": (15.364, 75.123), "Belgaum": (15.849, 74.497),
                "Chennai": (13.082, 80.270), "Coimbatore": (11.016, 76.955), "Madurai": (9.925, 78.119),
                "Tiruchirappalli": (10.790, 78.704), "Salem": (11.664, 78.146),
                "Delhi": (28.613, 77.209),
                "Ahmedabad": (23.022, 72.571), "Surat": (21.170, 72.831), "Vadodara": (22.307, 73.181), "Rajkot": (22.303, 70.802),
                "Kolkata": (22.572, 88.363), "Howrah": (22.595, 88.263), "Durgapur": (23.520, 87.311),
                "Jaipur": (26.912, 75.787), "Udaipur": (24.585, 73.712), "Jodhpur": (26.238, 73.024),
                "Lucknow": (26.846, 80.946), "Kanpur": (26.449, 80.331), "Noida": (28.535, 77.391), "Agra": (27.176, 78.008),
                "Bhopal": (23.259, 77.412), "Indore": (22.719, 75.857), "Gwalior": (26.218, 78.182),
                "Thiruvananthapuram": (8.524, 76.936), "Kochi": (9.931, 76.267), "Kozhikode": (11.258, 75.780),
                "Vijayawada": (16.506, 80.648), "Visakhapatnam": (17.686, 83.218), "Guntur": (16.306, 80.436),
                "Hyderabad": (17.385, 78.486), "Warangal": (17.978, 79.594),
                "Amritsar": (31.634, 74.872), "Ludhiana": (30.901, 75.857), "Jalandhar": (31.326, 75.576)
            }
            lat0, lon0 = city_center[selected_city]
            df = df[(df["lat"].between(lat0-0.2, lat0+0.2)) &
                    (df["lon"].between(lon0-0.2, lon0+0.2))]
        else:
            lat0, lon0 = df["lat"].mean(), df["lon"].mean()

    st.success(f"Loaded {len(df):,} accident points for {selected_city}")

    # GRID AGGREGATION
    df["lat_bin"] = (df["lat"] / grid_size).astype(int) * grid_size
    df["lon_bin"] = (df["lon"] / grid_size).astype(int) * grid_size

    grid = (
        df.groupby(["lat_bin", "lon_bin"])
        .agg({
            "road_type": lambda x: x.mode()[0],
            "lat": "mean",
            "lon": "mean"
        })
        .reset_index()
    )
    grid["accidents"] = df.groupby(["lat_bin", "lon_bin"]).size().values
    grid["risk"] = normalize(grid["accidents"]) * 100

    # WEATHER
    if enable_weather:
        c_lat = grid["lat_bin"].mean()
        c_lon = grid["lon_bin"].mean()
        w = weather_risk(c_lat, c_lon)
        grid["risk"] *= w
        st.info(f"🌧 Weather multiplier applied: {round(w,2)}")

    # ROAD-TYPE WEIGHTING
    if enable_road_weight:
        road_weight = {"NH": 1.2, "SH": 1.0, "Urban": 0.8}
        grid["risk"] *= grid["road_type"].map(lambda x: road_weight.get(x,1))
        st.info("🛣 Road-type weighting applied")

    # AEB / LDW zones
    aeb_zones = grid[grid['risk'] > 70].copy()
    ldw_zones = grid[(grid['risk'] > 50) & (grid['risk'] <= 70)].copy()
    for dfz in [aeb_zones, ldw_zones]:
        dfz.rename(columns={'lat_bin':'lat', 'lon_bin':'lon'}, inplace=True)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Grid Cells", len(grid))
    c2.metric("High Risk Cells", (grid["risk"] > 60).sum())
    c3.metric("Max Risk", round(grid["risk"].max(), 2))

    # =========================
    # TABS: ADAS / AEB / LDW
    # =========================
    tabs = st.tabs(["ADAS Risk Heatmap", "AEB Risk Zones", "LDW Risk Zones"])

    # ---- ADAS Risk Heatmap ----
    with tabs[0]:
        st.subheader(f"🗺️ ADAS Risk Heatmap for {selected_city}")
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=grid,
            get_position="[lon, lat]",
            get_weight="risk",
            radiusPixels=70,
            intensity=1.2,
            threshold=0.05
        )
        view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6 if selected_city=="All Cities" else 11)
        st.pydeck_chart(pdk.Deck(
            layers=[heatmap_layer],
            initial_view_state=view,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"text": "Risk Score: {risk}"}
        ))

    # ---- AEB Risk Zones ----
    with tabs[1]:
        st.subheader(f"🛑 AEB (Autonomous Emergency Braking) Risk Zones")
        aeb_layer = pdk.Layer(
            "ScatterplotLayer",
            data=aeb_zones,
            get_position="[lon, lat]",
            get_fill_color="[255, 0, 0, 120]",
            get_radius=100,
            pickable=True,
            auto_highlight=True
        )
        st.pydeck_chart(pdk.Deck(layers=[aeb_layer], initial_view_state=view, map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"))

    # ---- LDW Risk Zones ----
    with tabs[2]:
        st.subheader(f"⚠ LDW (Lane Departure Warning) Risk Zones")
        ldw_layer = pdk.Layer(
            "ScatterplotLayer",
            data=ldw_zones,
            get_position="[lon, lat]",
            get_fill_color="[255, 255, 0, 120]",
            get_radius=100,
            pickable=True,
            auto_highlight=True
        )
        st.pydeck_chart(pdk.Deck(layers=[ldw_layer], initial_view_state=view, map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"))

    # =========================
    # ANALYSIS GRAPH
    # =========================
    st.subheader("📊 AEB / LDW Zone Analysis")
    aeb_summary = aeb_zones.groupby("risk").size().reset_index(name="count")
    ldw_summary = ldw_zones.groupby("risk").size().reset_index(name="count")

    fig = px.bar(
        pd.concat([
            aeb_summary.assign(zone="AEB"),
            ldw_summary.assign(zone="LDW")
        ]),
        x="risk",
        y="count",
        color="zone",
        barmode="group",
        title=f"AEB vs LDW Risk Distribution in {selected_city}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ ADAS Risk Analysis Complete")
