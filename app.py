# """
# ADAS Risk Heatmap Generator (Streamlit)
# Author: Suresh Konar
# Purpose: OEM-style prototype for ADAS risk prioritization
# """

# # =========================
# # STREAMLIT CONFIG
# # =========================
# import streamlit as st
# st.set_page_config(
#     page_title="ADAS Risk Heatmap Generator",
#     page_icon="🚗",
#     layout="wide"
# )

# # =========================
# # IMPORTS
# # =========================
# import os
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point

# import kagglehub
# import osmnx as ox
# import folium
# from streamlit_folium import st_folium
# import requests

# # =========================
# # UI HEADER
# # =========================
# st.title("🚨 ADAS Risk Heatmap Generator")
# st.caption("India-focused ADAS risk intelligence using accidents, roads & weather")

# # =========================
# # SIDEBAR CONTROLS
# # =========================
# st.sidebar.header("⚙️ ADAS Controls")

# buffer_km = st.sidebar.slider(
#     "Road Network Radius (km)",
#     min_value=2,
#     max_value=20,
#     value=5
# )

# enable_weather = st.sidebar.checkbox(
#     "Enable Weather Risk (Rain / Fog)",
#     value=True
# )

# run_pipeline = st.sidebar.button("▶ Run ADAS Risk Pipeline")

# # =========================
# # UTILITY FUNCTIONS
# # =========================
# def normalize(series):
#     return (series - series.min()) / (series.max() - series.min() + 1e-6)

# def get_risk_color(score):
#     if score < 30:
#         return "green"
#     elif score < 60:
#         return "orange"
#     else:
#         return "red"

# # =========================
# # STEP 1: DOWNLOAD DATA
# # =========================
# @st.cache_data(show_spinner=False)
# def download_dataset():
#     path = kagglehub.dataset_download(
#         "data125661/india-road-accident-dataset"
#     )
#     return path

# # =========================
# # STEP 2: LOAD & CLEAN
# # =========================
# @st.cache_data(show_spinner=False)
# def load_and_clean_data(path):
#     csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]
#     df = pd.read_csv(os.path.join(path, csv))

#     df.columns = [c.strip().lower() for c in df.columns]
#     lat = [c for c in df.columns if "lat" in c][0]
#     lon = [c for c in df.columns if "lon" in c][0]

#     df = df.dropna(subset=[lat, lon])
#     df = df[df[lat].between(-90, 90) & df[lon].between(-180, 180)]

#     gdf = gpd.GeoDataFrame(
#         df,
#         geometry=[Point(xy) for xy in zip(df[lon], df[lat])],
#         crs="EPSG:4326"
#     )

#     return gdf, lat, lon

# # =========================
# # STEP 3: ROAD NETWORK
# # =========================
# @st.cache_data(show_spinner=False)
# def load_road_network(_gdf, buffer_km):
#     center = _gdf.geometry.unary_union.centroid
#     G = ox.graph_from_point(
#         (center.y, center.x),
#         dist=buffer_km * 1000,
#         network_type="drive"
#     )
#     edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
#     return edges

# # =========================
# # STEP 4: ACCIDENT DENSITY
# # =========================
# def compute_accident_density(accidents, roads):
#     roads = roads.to_crs(4326)
#     accidents = accidents.to_crs(4326)

#     joined = gpd.sjoin(
#         roads,
#         accidents,
#         how="left",
#         predicate="intersects"
#     )

#     density = joined.groupby(joined.index).size()
#     roads["accident_count"] = density
#     roads["accident_count"] = roads["accident_count"].fillna(0)

#     roads["accidents_per_km"] = (
#         roads["accident_count"] /
#         (roads["length"] / 1000)
#     )

#     return roads

# # =========================
# # STEP 5: ADAS RISK ENGINE
# # =========================
# ROAD_TYPE_RISK = {
#     "motorway": 0.9,
#     "trunk": 0.8,
#     "primary": 0.7,
#     "secondary": 0.6,
#     "tertiary": 0.5,
#     "residential": 0.4,
#     "service": 0.3
# }

# def compute_adas_risk(roads):
#     roads["complexity"] = normalize(1 / (roads["length"] + 1))
#     roads["type_risk"] = roads["highway"].apply(
#         lambda x: ROAD_TYPE_RISK.get(x[0], 0.5) if isinstance(x, list)
#         else ROAD_TYPE_RISK.get(x, 0.5)
#     )
#     roads["density_score"] = normalize(roads["accidents_per_km"])

#     roads["adas_risk"] = (
#         0.5 * roads["density_score"] +
#         0.3 * roads["complexity"] +
#         0.2 * roads["type_risk"]
#     ) * 100

#     return roads

# # =========================
# # STEP 6: WEATHER RISK
# # =========================
# def fetch_weather_risk(lat, lon):
#     url = (
#         "https://api.open-meteo.com/v1/forecast"
#         f"?latitude={lat}&longitude={lon}"
#         "&hourly=precipitation_probability,relativehumidity_2m,temperature_2m"
#         "&forecast_days=1"
#     )

#     try:
#         data = requests.get(url, timeout=10).json()
#         rain = max(data["hourly"]["precipitation_probability"]) / 100
#         humidity = max(data["hourly"]["relativehumidity_2m"])
#         temp = min(data["hourly"]["temperature_2m"])

#         fog = 0.7 if humidity > 85 and temp < 20 else 0.2
#         return round(0.6 * rain + 0.4 * fog, 2)
#     except:
#         return 0.3

# # =========================
# # STEP 7: MAP
# # =========================
# # def create_map(_roads_gdf):
# #     center = _roads_gdf.geometry.unary_union.centroid
# #     m = folium.Map(
# #         location=[center.y, center.x],
# #         zoom_start=13,
# #         tiles="cartodbpositron"
# #     )

# #     for _, r in _roads_gdf.iterrows():
# #         if r.geometry is None:
# #             continue

# #         coords = [(lat, lon) for lon, lat in r.geometry.coords]

# #         folium.PolyLine(
# #             locations=coords,
# #             color=get_risk_color(r["adas_risk"]),
# #             weight=5,
# #             opacity=0.8,
# #             popup=f"""
# #             <b>ADAS Risk:</b> {round(r['adas_risk'],2)}<br>
# #             <b>Acc/km:</b> {round(r['accidents_per_km'],2)}
# #             """
# #         ).add_to(m)

# #     return m
# def create_map(_roads_gdf):
#     center = _roads_gdf.geometry.union_all().centroid

#     m = folium.Map(
#         location=[center.y, center.x],
#         zoom_start=13,
#         tiles="cartodbpositron"
#     )

#     for _, r in _roads_gdf.iterrows():
#         if r.geometry is None:
#             continue

#         coords = [(lat, lon) for lon, lat in r.geometry.coords]

#         folium.PolyLine(
#             locations=coords,
#             color=get_risk_color(r["adas_risk"]),
#             weight=5,
#             opacity=0.8,
#             popup=f"""
#             <b>ADAS Risk:</b> {round(r['adas_risk'],2)}<br>
#             <b>Acc/km:</b> {round(r['accidents_per_km'],2)}
#             """
#         ).add_to(m)

#     return m


# # =========================
# # MAIN EXECUTION
# # =========================
# if run_pipeline:

#     with st.spinner("Downloading and preparing accident data..."):
#         dataset_path = download_dataset()
#         accidents, lat, lon = load_and_clean_data(dataset_path)

#     st.success(f"Accident records loaded: {len(accidents):,}")

#     with st.spinner("Downloading road network..."):
#         roads = load_road_network(accidents, buffer_km)

#     with st.spinner("Computing accident density & ADAS risk..."):
#         roads = compute_accident_density(accidents, roads)
#         roads = compute_adas_risk(roads)

#     if enable_weather:
#         with st.spinner("Applying weather-based risk..."):
#             c = roads.geometry.unary_union.centroid
#             weather_risk = fetch_weather_risk(c.y, c.x)
#             roads["adas_risk"] = roads["adas_risk"] * (1 + 0.15 * weather_risk)

#         st.info(f"🌧 Weather risk applied: {weather_risk}")

#     # KPIs
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Road Segments", len(roads))
#     col2.metric("High Risk Segments", (roads["adas_risk"] > 60).sum())
#     col3.metric("Max ADAS Risk", round(roads["adas_risk"].max(), 2))

#     st.subheader("🗺️ ADAS Risk Heatmap")
#     adas_map = create_map(roads)
#     st_folium(adas_map, width=1200, height=600)

#     st.success("✅ ADAS Risk Heatmap Generated")


# """
# ADAS Risk Heatmap Generator (Streamlit)
# Author: Suresh Konar
# Purpose: OEM-style ADAS risk heatmap (stable architecture)
# """

# # =========================
# # STREAMLIT CONFIG
# # =========================
# import streamlit as st
# st.set_page_config(
#     page_title="ADAS Risk Heatmap Generator",
#     page_icon="🚗",
#     layout="wide"
# )

# # =========================
# # IMPORTS
# # =========================
# import os
# import pandas as pd
# import numpy as np
# import kagglehub
# import pydeck as pdk
# import requests

# # =========================
# # UI HEADER
# # =========================
# st.title("🚨 ADAS Risk Heatmap Generator")
# st.caption("India-focused ADAS risk intelligence using accidents & weather")

# # =========================
# # SIDEBAR
# # =========================
# st.sidebar.header("⚙️ Controls")

# grid_size = st.sidebar.slider(
#     "Grid Size (degrees)",
#     0.005, 0.05, 0.01, step=0.005
# )

# enable_weather = st.sidebar.checkbox(
#     "Enable Weather Risk (Rain / Fog)", True
# )

# run = st.sidebar.button("▶ Run Pipeline")

# # =========================
# # UTILS
# # =========================
# def normalize(x):
#     return (x - x.min()) / (x.max() - x.min() + 1e-6)

# # =========================
# # DATA LOAD
# # =========================
# @st.cache_data
# def load_data():
#     path = kagglehub.dataset_download(
#         "data125661/india-road-accident-dataset"
#     )
#     csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]
#     df = pd.read_csv(os.path.join(path, csv))

#     df.columns = [c.lower() for c in df.columns]
#     lat = [c for c in df.columns if "lat" in c][0]
#     lon = [c for c in df.columns if "lon" in c][0]

#     df = df[[lat, lon]].dropna()
#     df = df[df[lat].between(6, 36) & df[lon].between(68, 98)]

#     df.rename(columns={lat: "lat", lon: "lon"}, inplace=True)
#     return df

# # =========================
# # WEATHER RISK
# # =========================
# def weather_risk(lat, lon):
#     try:
#         url = (
#             "https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}"
#             "&hourly=precipitation_probability,relativehumidity_2m,temperature_2m"
#             "&forecast_days=1"
#         )
#         d = requests.get(url, timeout=10).json()
#         rain = max(d["hourly"]["precipitation_probability"]) / 100
#         hum = max(d["hourly"]["relativehumidity_2m"])
#         temp = min(d["hourly"]["temperature_2m"])
#         fog = 0.7 if hum > 85 and temp < 20 else 0.2
#         return 1 + 0.15 * (0.6 * rain + 0.4 * fog)
#     except:
#         return 1.0

# # =========================
# # ADAS Feature Overlay
# # =========================
# def adas_feature_multiplier(grid):
#     # Example logic: certain lat/lon zones are NH (higher speeds), increase risk
#     grid['adas_multiplier'] = 1.0
    
#     # Example: add 20% risk for high-speed corridors (simulate AEB trigger zones)
#     # You could replace this with a real dataset of NH / SH / Urban zones
#     grid.loc[(grid['lat_bin'] > 25) & (grid['lon_bin'] > 75), 'adas_multiplier'] = 1.2
#     grid['risk'] = grid['risk'] * grid['adas_multiplier']
#     return grid

# # Apply in pipeline
# grid = adas_feature_multiplier(grid)

# # =========================
# # ROAD TYPE WEIGHTING
# # =========================
# def road_type_weighting(grid):
#     # Example logic (latitude/longitude based simulation)
#     # NH = higher risk multiplier
#     # SH = medium
#     # Urban = lower
#     def get_multiplier(lat, lon):
#         if lat > 25 and lon > 75:
#             return 1.2   # NH
#         elif lat > 15 and lon > 70:
#             return 1.1   # SH
#         else:
#             return 0.9   # Urban

#     grid['road_multiplier'] = grid.apply(lambda r: get_multiplier(r['lat_bin'], r['lon_bin']), axis=1)
#     grid['risk'] = grid['risk'] * grid['road_multiplier']
#     return grid

# # Apply in pipeline
# grid = road_type_weighting(grid)



# # =========================
# # PIPELINE
# # =========================
# if run:
#     with st.spinner("Loading accident data..."):
#         df = load_data()

#     st.success(f"Loaded {len(df):,} accident points")

#     # GRID AGGREGATION
#     df["lat_bin"] = (df["lat"] / grid_size).astype(int) * grid_size
#     df["lon_bin"] = (df["lon"] / grid_size).astype(int) * grid_size

#     grid = (
#         df.groupby(["lat_bin", "lon_bin"])
#         .size()
#         .reset_index(name="accidents")
#     )

#     grid["risk"] = normalize(grid["accidents"]) * 100

#     # WEATHER
#     if enable_weather:
#         c_lat = grid["lat_bin"].mean()
#         c_lon = grid["lon_bin"].mean()
#         w = weather_risk(c_lat, c_lon)
#         grid["risk"] *= w
#         st.info(f"🌧 Weather multiplier applied: {round(w,2)}")

#     # KPIs
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Grid Cells", len(grid))
#     c2.metric("High Risk Cells", (grid["risk"] > 60).sum())
#     c3.metric("Max Risk", round(grid["risk"].max(), 2))

#     # =========================
#     # MAP (pydeck)
#     # =========================
#     st.subheader("🗺️ ADAS Risk Heatmap")

#     # layer = pdk.Layer(
#     #     "HeatmapLayer",
#     #     data=grid,
#     #     get_position="[lon_bin, lat_bin]",
#     #     get_weight="risk",
#     #     radiusPixels=50
#     # )

#     layer = pdk.Layer(
#         "HeatmapLayer",
#         data=grid,
#         get_position="[lon_bin, lat_bin]",
#         get_weight="risk",
#         radiusPixels=70,
#         intensity=1.2,
#         threshold=0.05
#     )


#     view = pdk.ViewState(
#         latitude=grid["lat_bin"].mean(),
#         longitude=grid["lon_bin"].mean(),
#         zoom=6
#     )

#     # st.pydeck_chart(pdk.Deck(
#     #     layers=[layer],
#     #     initial_view_state=view,
#     #     map_style="mapbox://styles/mapbox/light-v10"
#     # ))

#     st.pydeck_chart(pdk.Deck(
#         layers=[layer],
#         initial_view_state=view,
#         map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
#         tooltip={"text": "Risk Score: {risk}"}
#     ))


#     st.success("✅ ADAS Risk Heatmap Generated")



# """
# ADAS Risk Heatmap Generator (Streamlit)
# Author: Suresh Konar
# Purpose: OEM-style ADAS risk heatmap with AEB/LDW overlay and analysis
# """

# # =========================
# # STREAMLIT CONFIG
# # =========================
# import streamlit as st
# st.set_page_config(
#     page_title="ADAS Risk Heatmap Generator",
#     page_icon="🚗",
#     layout="wide"
# )

# # =========================
# # IMPORTS
# # =========================
# import os
# import pandas as pd
# import numpy as np
# import kagglehub
# import pydeck as pdk
# import requests
# import plotly.express as px

# # =========================
# # UI HEADER
# # =========================
# st.title("🚨 ADAS Risk Heatmap Generator")
# st.caption("India-focused ADAS risk intelligence using accidents & weather")

# # =========================
# # SIDEBAR
# # =========================
# st.sidebar.header("⚙️ Controls")

# # Select State / City
# state_city_map = {
#     "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
#     "Karnataka": ["Bengaluru", "Mysore", "Mangalore"],
#     "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
#     "Delhi": ["Delhi"]
# }

# selected_state = st.sidebar.selectbox("Select State", list(state_city_map.keys()))
# selected_city = st.sidebar.selectbox("Select City", state_city_map[selected_state])

# grid_size = st.sidebar.slider(
#     "Grid Size (degrees)",
#     0.005, 0.05, 0.01, step=0.005
# )

# enable_weather = st.sidebar.checkbox(
#     "Enable Weather Risk (Rain / Fog)", True
# )

# run = st.sidebar.button("▶ Run Pipeline")

# # =========================
# # UTILS
# # =========================
# def normalize(x):
#     return (x - x.min()) / (x.max() - x.min() + 1e-6)

# # =========================
# # DATA LOAD
# # =========================
# @st.cache_data
# def load_data():
#     path = kagglehub.dataset_download(
#         "data125661/india-road-accident-dataset"
#     )
#     csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]
#     df = pd.read_csv(os.path.join(path, csv))

#     df.columns = [c.lower() for c in df.columns]
#     lat = [c for c in df.columns if "lat" in c][0]
#     lon = [c for c in df.columns if "lon" in c][0]

#     df = df[[lat, lon]].dropna()
#     df = df[df[lat].between(6, 36) & df[lon].between(68, 98)]

#     df.rename(columns={lat: "lat", lon: "lon"}, inplace=True)
#     return df

# # =========================
# # WEATHER RISK
# # =========================
# def weather_risk(lat, lon):
#     try:
#         url = (
#             "https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}"
#             "&hourly=precipitation_probability,relativehumidity_2m,temperature_2m"
#             "&forecast_days=1"
#         )
#         d = requests.get(url, timeout=10).json()
#         rain = max(d["hourly"]["precipitation_probability"]) / 100
#         hum = max(d["hourly"]["relativehumidity_2m"])
#         temp = min(d["hourly"]["temperature_2m"])
#         fog = 0.7 if hum > 85 and temp < 20 else 0.2
#         return 1 + 0.15 * (0.6 * rain + 0.4 * fog)
#     except:
#         return 1.0

# # =========================
# # PIPELINE
# # =========================
# if run:
#     with st.spinner("Loading accident data..."):
#         df = load_data()
#         # Filter for selected city bounding box (simple demo)
#         # Ideally use shapefiles per city for precise filtering
#         city_center = {
#             "Mumbai": (19.076, 72.877),
#             "Pune": (18.520, 73.856),
#             "Nagpur": (21.145, 79.088),
#             "Bengaluru": (12.971, 77.594),
#             "Mysore": (12.295, 76.639),
#             "Mangalore": (12.915, 74.856),
#             "Chennai": (13.082, 80.270),
#             "Coimbatore": (11.016, 76.955),
#             "Madurai": (9.925, 78.119),
#             "Delhi": (28.613, 77.209)
#         }
#         lat0, lon0 = city_center[selected_city]
#         df = df[(df["lat"].between(lat0-0.2, lat0+0.2)) &
#                 (df["lon"].between(lon0-0.2, lon0+0.2))]

#     st.success(f"Loaded {len(df):,} accident points for {selected_city}")

#     # GRID AGGREGATION
#     df["lat_bin"] = (df["lat"] / grid_size).astype(int) * grid_size
#     df["lon_bin"] = (df["lon"] / grid_size).astype(int) * grid_size

#     grid = (
#         df.groupby(["lat_bin", "lon_bin"])
#         .size()
#         .reset_index(name="accidents")
#     )

#     grid["risk"] = normalize(grid["accidents"]) * 100

#     # WEATHER
#     if enable_weather:
#         c_lat = grid["lat_bin"].mean()
#         c_lon = grid["lon_bin"].mean()
#         w = weather_risk(c_lat, c_lon)
#         grid["risk"] *= w
#         st.info(f"🌧 Weather multiplier applied: {round(w,2)}")

#     # Define AEB / LDW zones
#     aeb_zones = grid[grid['risk'] > 70].copy()
#     ldw_zones = grid[(grid['risk'] > 50) & (grid['risk'] <= 70)].copy()
#     for dfz in [aeb_zones, ldw_zones]:
#         dfz.rename(columns={'lat_bin':'lat', 'lon_bin':'lon'}, inplace=True)

#     # KPIs
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Grid Cells", len(grid))
#     c2.metric("High Risk Cells", (grid["risk"] > 60).sum())
#     c3.metric("Max Risk", round(grid["risk"].max(), 2))

#     # =========================
#     # MAP (Heatmap + Scatter overlays)
#     # =========================
#     st.subheader(f"🗺️ ADAS Risk Heatmap for {selected_city}")

#     heatmap_layer = pdk.Layer(
#         "HeatmapLayer",
#         data=grid,
#         get_position="[lon_bin, lat_bin]",
#         get_weight="risk",
#         radiusPixels=70,
#         intensity=1.2,
#         threshold=0.05
#     )

#     aeb_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=aeb_zones,
#         get_position="[lon, lat]",
#         get_fill_color="[255, 0, 0, 120]",
#         get_radius=100,
#         pickable=True,
#         auto_highlight=True
#     )

#     ldw_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=ldw_zones,
#         get_position="[lon, lat]",
#         get_fill_color="[255, 255, 0, 120]",
#         get_radius=100,
#         pickable=True,
#         auto_highlight=True
#     )

#     view = pdk.ViewState(
#         latitude=lat0,
#         longitude=lon0,
#         zoom=11
#     )

#     st.pydeck_chart(pdk.Deck(
#         layers=[heatmap_layer, aeb_layer, ldw_layer],
#         initial_view_state=view,
#         map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
#         tooltip={"text": "Risk Score: {risk}"}
#     ))

#     # =========================
#     # ANALYSIS GRAPH
#     # =========================
#     st.subheader("📊 AEB / LDW Zone Analysis")

#     aeb_summary = aeb_zones.groupby("risk").size().reset_index(name="count")
#     ldw_summary = ldw_zones.groupby("risk").size().reset_index(name="count")

#     fig = px.bar(
#         pd.concat([
#             aeb_summary.assign(zone="AEB"),
#             ldw_summary.assign(zone="LDW")
#         ]),
#         x="risk",
#         y="count",
#         color="zone",
#         barmode="group",
#         title=f"AEB vs LDW Risk Distribution in {selected_city}"
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     st.success("✅ ADAS Risk Analysis Complete")


# """
# ADAS Risk Heatmap Generator (Streamlit)
# Author: Suresh Konar
# Purpose: OEM-style ADAS risk heatmap with AEB/LDW overlay, road-type weighting & analysis
# """

# # =========================
# # STREAMLIT CONFIG
# # =========================
# import streamlit as st
# st.set_page_config(
#     page_title="ADAS Risk Heatmap Generator",
#     page_icon="🚗",
#     layout="wide"
# )

# # =========================
# # IMPORTS
# # =========================
# import os
# import pandas as pd
# import numpy as np
# import kagglehub
# import pydeck as pdk
# import requests
# import plotly.express as px
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# # =========================
# # UI HEADER
# # =========================
# st.title("🚨 ADAS Risk Heatmap Generator")
# st.caption("India-focused ADAS risk intelligence using accidents, weather & road-type weighting")

# # =========================
# # SIDEBAR
# # =========================
# st.sidebar.header("⚙️ Controls")

# # Select State / City
# state_city_map = {
#     "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
#     "Karnataka": ["Bengaluru", "Mysore", "Mangalore"],
#     "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
#     "Delhi": ["Delhi"]
# }

# selected_state = st.sidebar.selectbox("Select State", list(state_city_map.keys()))
# selected_city = st.sidebar.selectbox("Select City", state_city_map[selected_state])

# grid_size = st.sidebar.slider(
#     "Grid Size (degrees)",
#     0.005, 0.05, 0.01, step=0.005
# )

# enable_weather = st.sidebar.checkbox(
#     "Enable Weather Risk (Rain / Fog)", True
# )

# enable_road_weight = st.sidebar.checkbox(
#     "Enable Road-Type Weighting (NH/SH/Urban)", True
# )

# run = st.sidebar.button("▶ Run Pipeline")

# # =========================
# # UTILS
# # =========================
# def normalize(x):
#     return (x - x.min()) / (x.max() - x.min() + 1e-6)

# # =========================
# # DATA LOAD
# # =========================
# @st.cache_data
# def load_data():
#     path = kagglehub.dataset_download(
#         "data125661/india-road-accident-dataset"
#     )
#     csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]
#     df = pd.read_csv(os.path.join(path, csv))

#     df.columns = [c.lower() for c in df.columns]
#     lat = [c for c in df.columns if "lat" in c][0]
#     lon = [c for c in df.columns if "lon" in c][0]

#     # Optional road type (demo: NH/SH/Urban)
#     if "road_type" not in df.columns:
#         np.random.seed(0)
#         df["road_type"] = np.random.choice(["NH","SH","Urban"], size=len(df))

#     df = df[[lat, lon, "road_type"]].dropna()
#     df = df[df[lat].between(6, 36) & df[lon].between(68, 98)]

#     df.rename(columns={lat: "lat", lon: "lon"}, inplace=True)
#     return df

# # =========================
# # WEATHER RISK
# # =========================
# def weather_risk(lat, lon):
#     try:
#         url = (
#             "https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}"
#             "&hourly=precipitation_probability,relativehumidity_2m,temperature_2m"
#             "&forecast_days=1"
#         )
#         d = requests.get(url, timeout=10).json()
#         rain = max(d["hourly"]["precipitation_probability"]) / 100
#         hum = max(d["hourly"]["relativehumidity_2m"])
#         temp = min(d["hourly"]["temperature_2m"])
#         fog = 0.7 if hum > 85 and temp < 20 else 0.2
#         return 1 + 0.15 * (0.6 * rain + 0.4 * fog)
#     except:
#         return 1.0

# # =========================
# # PIPELINE
# # =========================
# if run:
#     with st.spinner("Loading accident data..."):
#         df = load_data()
#         # Filter for selected city bounding box
#         city_center = {
#             "Mumbai": (19.076, 72.877),
#             "Pune": (18.520, 73.856),
#             "Nagpur": (21.145, 79.088),
#             "Bengaluru": (12.971, 77.594),
#             "Mysore": (12.295, 76.639),
#             "Mangalore": (12.915, 74.856),
#             "Chennai": (13.082, 80.270),
#             "Coimbatore": (11.016, 76.955),
#             "Madurai": (9.925, 78.119),
#             "Delhi": (28.613, 77.209)
#         }
#         lat0, lon0 = city_center[selected_city]
#         df = df[(df["lat"].between(lat0-0.2, lat0+0.2)) &
#                 (df["lon"].between(lon0-0.2, lon0+0.2))]

#     st.success(f"Loaded {len(df):,} accident points for {selected_city}")

#     # GRID AGGREGATION
#     df["lat_bin"] = (df["lat"] / grid_size).astype(int) * grid_size
#     df["lon_bin"] = (df["lon"] / grid_size).astype(int) * grid_size

#     grid = (
#         df.groupby(["lat_bin", "lon_bin"])
#         .agg({
#             "road_type": lambda x: x.mode()[0],
#             "lat": "mean",
#             "lon": "mean"
#         })
#         .reset_index()
#     )
#     grid["accidents"] = df.groupby(["lat_bin", "lon_bin"]).size().values
#     grid["risk"] = normalize(grid["accidents"]) * 100

#     # WEATHER
#     if enable_weather:
#         c_lat = grid["lat_bin"].mean()
#         c_lon = grid["lon_bin"].mean()
#         w = weather_risk(c_lat, c_lon)
#         grid["risk"] *= w
#         st.info(f"🌧 Weather multiplier applied: {round(w,2)}\n- Accounts for rain & fog conditions")

#     # ROAD-TYPE WEIGHTING
#     if enable_road_weight:
#         road_weight = {"NH": 1.2, "SH": 1.0, "Urban": 0.8}
#         grid["risk"] *= grid["road_type"].map(lambda x: road_weight.get(x,1))
#         st.info("🛣 Road-type weighting applied:\n- NH: +20% risk\n- SH: no change\n- Urban: -20% risk")

#     # Define AEB / LDW zones
#     aeb_zones = grid[grid['risk'] > 70].copy()
#     ldw_zones = grid[(grid['risk'] > 50) & (grid['risk'] <= 70)].copy()
#     for dfz in [aeb_zones, ldw_zones]:
#         dfz.rename(columns={'lat_bin':'lat', 'lon_bin':'lon'}, inplace=True)

#     # KPIs
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Grid Cells", len(grid))
#     c2.metric("High Risk Cells", (grid["risk"] > 60).sum())
#     c3.metric("Max Risk", round(grid["risk"].max(), 2))

#     # =========================
#     # HEATMAP COLOR LEGEND
#     # =========================
#     st.subheader("🌈 Heatmap Color Legend")
#     st.markdown("""
#     - 🟢 Green: Low Risk (0–30)  
#     - 🟠 Orange: Medium Risk (30–60)  
#     - 🔴 Red: High Risk (60–100)
#     """)
#     # Optional: matplotlib gradient legend
#     fig, ax = plt.subplots(figsize=(6, 0.5))
#     cmap = mcolors.LinearSegmentedColormap.from_list("", ["green","orange","red"])
#     norm = mcolors.Normalize(vmin=0, vmax=100)
#     fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal', label='ADAS Risk Score')
#     st.pyplot(fig)

#     # =========================
#     # MAP (Heatmap + Scatter overlays)
#     # =========================
#     st.subheader(f"🗺️ ADAS Risk Heatmap for {selected_city}")

#     heatmap_layer = pdk.Layer(
#         "HeatmapLayer",
#         data=grid,
#         get_position="[lon, lat]",
#         get_weight="risk",
#         radiusPixels=70,
#         intensity=1.2,
#         threshold=0.05
#     )

#     aeb_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=aeb_zones,
#         get_position="[lon, lat]",
#         get_fill_color="[255, 0, 0, 120]",
#         get_radius=100,
#         pickable=True,
#         auto_highlight=True
#     )

#     ldw_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=ldw_zones,
#         get_position="[lon, lat]",
#         get_fill_color="[255, 255, 0, 120]",
#         get_radius=100,
#         pickable=True,
#         auto_highlight=True
#     )

#     view = pdk.ViewState(
#         latitude=lat0,
#         longitude=lon0,
#         zoom=11
#     )

#     st.pydeck_chart(pdk.Deck(
#         layers=[heatmap_layer, aeb_layer, ldw_layer],
#         initial_view_state=view,
#         map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
#         tooltip={"text": "Risk Score: {risk}"}
#     ))

#     # =========================
#     # ANALYSIS GRAPH
#     # =========================
#     st.subheader("📊 AEB / LDW Zone Analysis")

#     aeb_summary = aeb_zones.groupby("risk").size().reset_index(name="count")
#     ldw_summary = ldw_zones.groupby("risk").size().reset_index(name="count")

#     fig = px.bar(
#         pd.concat([
#             aeb_summary.assign(zone="AEB"),
#             ldw_summary.assign(zone="LDW")
#         ]),
#         x="risk",
#         y="count",
#         color="zone",
#         barmode="group",
#         title=f"AEB vs LDW Risk Distribution in {selected_city}"
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     st.success("✅ ADAS Risk Analysis Complete")


# """
# ADAS Risk Heatmap Generator (Streamlit)
# Author: Suresh Konar
# Purpose: OEM-style ADAS risk heatmap with AEB/LDW overlays, road-type weighting & analysis
# """

# # =========================
# # STREAMLIT CONFIG
# # =========================
# import streamlit as st
# st.set_page_config(
#     page_title="ADAS Risk Heatmap Generator",
#     page_icon="🚗",
#     layout="wide"
# )

# # =========================
# # IMPORTS
# # =========================
# import os
# import pandas as pd
# import numpy as np
# import kagglehub
# import pydeck as pdk
# import requests
# import plotly.express as px

# # =========================
# # TOOL DESCRIPTION
# # =========================
# st.title("🚨 ADAS Risk Heatmap Generator")
# st.markdown("""
# This tool visualizes **ADAS risk zones** for a selected city in India, including:
# - **ADAS Risk Heatmap** (general accident risk)
# - **AEB Risk Zones** (Autonomous Emergency Braking risk)
# - **LDW Risk Zones** (Lane Departure Warning risk)

# **Features:**
# - Weather-based risk multiplier (rain/fog)
# - Road-type weighting (NH/SH/Urban)
# - Interactive PyDeck maps with tooltips
# - Analysis charts for AEB/LDW zones

# **Disclaimer:**  
# Data sources are transparent and publicly available, primarily from Kaggle and Open Meteo APIs.
# """)

# # =========================
# # SIDEBAR
# # =========================
# st.sidebar.header("⚙️ Controls")

# state_city_map = {
#     "All India": ["All Cities"],
#     "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
#     "Karnataka": ["Bengaluru", "Mysore", "Mangalore", "Hubli", "Belgaum"],
#     "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
#     "Delhi": ["Delhi"],
#     "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
#     "West Bengal": ["Kolkata", "Howrah", "Durgapur"],
#     "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur"],
#     "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Agra"],
#     "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior"],
#     "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode"],
#     "Andhra Pradesh": ["Vijayawada", "Visakhapatnam", "Guntur"],
#     "Telangana": ["Hyderabad", "Warangal"],
#     "Punjab": ["Amritsar", "Ludhiana", "Jalandhar"]
# }

# selected_state = st.sidebar.selectbox("Select State", list(state_city_map.keys()))

# # If "All India" selected, disable city dropdown
# if selected_state == "All India":
#     selected_city = "All Cities"
# else:
#     selected_state = st.sidebar.selectbox("Select State", list(state_city_map.keys()))
#     selected_city = st.sidebar.selectbox("Select City", state_city_map[selected_state])



# grid_size = st.sidebar.slider(
#     "Grid Size (degrees)",
#     0.005, 0.05, 0.01, step=0.005
# )

# enable_weather = st.sidebar.checkbox(
#     "Enable Weather Risk (Rain / Fog)", True
# )

# enable_road_weight = st.sidebar.checkbox(
#     "Enable Road-Type Weighting (NH/SH/Urban)", True
# )

# run = st.sidebar.button("▶ Run Pipeline")

# # =========================
# # UTILS
# # =========================
# def normalize(x):
#     return (x - x.min()) / (x.max() - x.min() + 1e-6)

# # =========================
# # DATA LOAD
# # =========================
# @st.cache_data
# def load_data():
#     path = kagglehub.dataset_download(
#         "data125661/india-road-accident-dataset"
#     )
#     csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]
#     df = pd.read_csv(os.path.join(path, csv))

#     df.columns = [c.lower() for c in df.columns]
#     lat = [c for c in df.columns if "lat" in c][0]
#     lon = [c for c in df.columns if "lon" in c][0]

#     # Optional road type (demo: NH/SH/Urban)
#     if "road_type" not in df.columns:
#         np.random.seed(0)
#         df["road_type"] = np.random.choice(["NH","SH","Urban"], size=len(df))

#     df = df[[lat, lon, "road_type"]].dropna()
#     df = df[df[lat].between(6, 36) & df[lon].between(68, 98)]

#     df.rename(columns={lat: "lat", lon: "lon"}, inplace=True)
#     return df

# # =========================
# # WEATHER RISK
# # =========================
# def weather_risk(lat, lon):
#     try:
#         url = (
#             "https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}"
#             "&hourly=precipitation_probability,relativehumidity_2m,temperature_2m"
#             "&forecast_days=1"
#         )
#         d = requests.get(url, timeout=10).json()
#         rain = max(d["hourly"]["precipitation_probability"]) / 100
#         hum = max(d["hourly"]["relativehumidity_2m"])
#         temp = min(d["hourly"]["temperature_2m"])
#         fog = 0.7 if hum > 85 and temp < 20 else 0.2
#         return 1 + 0.15 * (0.6 * rain + 0.4 * fog)
#     except:
#         return 1.0

# # =========================
# # PIPELINE
# # =========================
# if run:
#     with st.spinner("Loading accident data..."):
#         df = load_data()

#         # Filter for selected city / All India
#     if selected_state != "All India":
#         city_center = {
#             "Mumbai": (19.076, 72.877),
#             "Pune": (18.520, 73.856),
#             "Nagpur": (21.145, 79.088),
#             "Nashik": (19.997, 73.789),
#             "Aurangabad": (19.876, 75.343),
#             "Bengaluru": (12.971, 77.594),
#             "Mysore": (12.295, 76.639),
#             "Mangalore": (12.915, 74.856),
#             "Hubli": (15.364, 75.123),
#             "Belgaum": (15.849, 74.497),
#             "Chennai": (13.082, 80.270),
#             "Coimbatore": (11.016, 76.955),
#             "Madurai": (9.925, 78.119),
#             "Tiruchirappalli": (10.790, 78.704),
#             "Salem": (11.664, 78.146),
#             "Delhi": (28.613, 77.209),
#             "Ahmedabad": (23.022, 72.571),
#             "Surat": (21.170, 72.831),
#             "Vadodara": (22.307, 73.181),
#             "Rajkot": (22.303, 70.802),
#             "Kolkata": (22.572, 88.363),
#             "Howrah": (22.595, 88.263),
#             "Durgapur": (23.520, 87.311),
#             "Jaipur": (26.912, 75.787),
#             "Udaipur": (24.585, 73.712),
#             "Jodhpur": (26.238, 73.024),
#             "Lucknow": (26.846, 80.946),
#             "Kanpur": (26.449, 80.331),
#             "Noida": (28.535, 77.391),
#             "Agra": (27.176, 78.008),
#             "Bhopal": (23.259, 77.412),
#             "Indore": (22.719, 75.857),
#             "Gwalior": (26.218, 78.182),
#             "Thiruvananthapuram": (8.524, 76.936),
#             "Kochi": (9.931, 76.267),
#             "Kozhikode": (11.258, 75.780),
#             "Vijayawada": (16.506, 80.648),
#             "Visakhapatnam": (17.686, 83.218),
#             "Guntur": (16.306, 80.436),
#             "Hyderabad": (17.385, 78.486),
#             "Warangal": (17.978, 79.594),
#             "Amritsar": (31.634, 74.872),
#             "Ludhiana": (30.901, 75.857),
#             "Jalandhar": (31.326, 75.576)
#         }

#         lat0, lon0 = city_center[selected_city]
#         df = df[(df["lat"].between(lat0-0.2, lat0+0.2)) &
#                 (df["lon"].between(lon0-0.2, lon0+0.2))]
#     else:
#         # For All India, keep entire df
#         lat0, lon0 = df["lat"].mean(), df["lon"].mean()

#     st.success(f"Loaded {len(df):,} accident points for {selected_city}")

#     # -------------------------
#     # GRID AGGREGATION
#     # -------------------------
#     df["lat_bin"] = (df["lat"] / grid_size).astype(int) * grid_size
#     df["lon_bin"] = (df["lon"] / grid_size).astype(int) * grid_size

#     grid = (
#         df.groupby(["lat_bin", "lon_bin"])
#         .agg({
#             "road_type": lambda x: x.mode()[0],
#             "lat": "mean",
#             "lon": "mean"
#         })
#         .reset_index()
#     )
#     grid["accidents"] = df.groupby(["lat_bin", "lon_bin"]).size().values
#     grid["risk"] = normalize(grid["accidents"]) * 100

#     # -------------------------
#     # WEATHER MULTIPLIER
#     # -------------------------
#     if enable_weather:
#         c_lat = grid["lat_bin"].mean()
#         c_lon = grid["lon_bin"].mean()
#         w = weather_risk(c_lat, c_lon)
#         grid["risk"] *= w
#         st.info(f"🌧 Weather multiplier applied: {round(w,2)}")

#     # -------------------------
#     # ROAD-TYPE WEIGHTING
#     # -------------------------
#     if enable_road_weight:
#         road_weight = {"NH": 1.2, "SH": 1.0, "Urban": 0.8}
#         grid["risk"] *= grid["road_type"].map(lambda x: road_weight.get(x,1))
#         st.info("🛣 Road-type weighting applied")

#     # -------------------------
#     # DEFINE AEB / LDW ZONES
#     # -------------------------
#     aeb_zones = grid[grid['risk'] > 70].copy()
#     ldw_zones = grid[(grid['risk'] > 50) & (grid['risk'] <= 70)].copy()
#     for dfz in [aeb_zones, ldw_zones]:
#         dfz.rename(columns={'lat_bin':'lat', 'lon_bin':'lon'}, inplace=True)

#     # -------------------------
#     # KPIs
#     # -------------------------
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Grid Cells", len(grid))
#     c2.metric("High Risk Cells", (grid["risk"] > 60).sum())
#     c3.metric("Max Risk", round(grid["risk"].max(), 2))

#     # =========================
#     # MAP VISUALIZATION (Separate Tabs)
#     # =========================
#     st.subheader(f"🗺️ Map Visualization for {selected_city}")

#     tab1, tab2, tab3 = st.tabs(["ADAS Risk Heatmap", "AEB Risk Zones", "LDW Risk Zones"])

#     # -------------------------
#     # ADAS Heatmap Tab
#     # -------------------------
#     with tab1:
#         heatmap_layer = pdk.Layer(
#             "HeatmapLayer",
#             data=grid,
#             get_position="[lon, lat]",
#             get_weight="risk",
#             radiusPixels=70,
#             intensity=1.2,
#             threshold=0.05
#         )

#         view = pdk.ViewState(
#             latitude=lat0,
#             longitude=lon0,
#             zoom=11
#         )

#         st.pydeck_chart(pdk.Deck(
#             layers=[heatmap_layer],
#             initial_view_state=view,
#             map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
#             tooltip={"text": "Risk Score: {risk}\nRoad Type: {road_type}"}
#         ))
#         st.markdown("**Heatmap Legend:** Green=Low Risk, Orange=Medium Risk, Red=High Risk")

#     # -------------------------
#     # AEB Zones Tab
#     # -------------------------
#     with tab2:
#         aeb_layer = pdk.Layer(
#             "ScatterplotLayer",
#             data=aeb_zones,
#             get_position="[lon, lat]",
#             get_fill_color="[255, 0, 0, 180]",
#             get_radius=100,
#             pickable=True,
#             auto_highlight=True
#         )

#         st.pydeck_chart(pdk.Deck(
#             layers=[aeb_layer],
#             initial_view_state=view,
#             map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
#             tooltip={"text": "AEB Zone\nRisk Score: {risk}\nRoad Type: {road_type}"}
#         ))
#         st.markdown("**AEB = Autonomous Emergency Braking**")

#     # -------------------------
#     # LDW Zones Tab
#     # -------------------------
#     with tab3:
#         ldw_layer = pdk.Layer(
#             "ScatterplotLayer",
#             data=ldw_zones,
#             get_position="[lon, lat]",
#             get_fill_color="[255, 255, 0, 180]",
#             get_radius=100,
#             pickable=True,
#             auto_highlight=True
#         )

#         st.pydeck_chart(pdk.Deck(
#             layers=[ldw_layer],
#             initial_view_state=view,
#             map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
#             tooltip={"text": "LDW Zone\nRisk Score: {risk}\nRoad Type: {road_type}"}
#         ))
#         st.markdown("**LDW = Lane Departure Warning**")

#     # =========================
#     # ANALYSIS GRAPH
#     # =========================
#     st.subheader("📊 AEB / LDW Zone Analysis")

#     aeb_summary = aeb_zones.groupby("risk").size().reset_index(name="count")
#     ldw_summary = ldw_zones.groupby("risk").size().reset_index(name="count")

#     fig = px.bar(
#         pd.concat([
#             aeb_summary.assign(zone="AEB"),
#             ldw_summary.assign(zone="LDW")
#         ]),
#         x="risk",
#         y="count",
#         color="zone",
#         barmode="group",
#         title=f"AEB vs LDW Risk Distribution in {selected_city}"
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     st.success("✅ ADAS Risk Analysis Complete")


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
    page_title="ADAS Risk Heatmap Generator",
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
st.title("🚨 ADAS Risk Heatmap Generator")
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
    lat = [c for c in df.columns if "lat" in c][0]
    lon = [c for c in df.columns if "lon" in c][0]

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
