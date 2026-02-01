"""
ADAS Risk Heatmap Generator
Step 1: Indian Road Accident Data Ingestion & Cleaning

Author: Suresh Konar
Purpose: OEM-style prototype for ADAS risk prioritization
"""

# =========================
# STEP 0: Imports
# =========================
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import kagglehub

import osmnx as ox
import networkx as nx

import folium
from branca.colormap import LinearColormap

import requests


# =========================
# STEP 1: Dataset Download
# =========================
def download_dataset():
    print("[INFO] Downloading India Road Accident Dataset from Kaggle...")
    path = kagglehub.dataset_download(
        "data125661/india-road-accident-dataset"
    )
    print(f"[INFO] Dataset downloaded at: {path}")
    return path


# =========================
# STEP 2: Load Dataset
# =========================
def load_accident_data(dataset_path):
    print("[INFO] Loading CSV files...")

    csv_files = [
        f for f in os.listdir(dataset_path)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset directory")

    file_path = os.path.join(dataset_path, csv_files[0])
    df = pd.read_csv(file_path)

    print(f"[INFO] Loaded file: {csv_files[0]}")
    print(f"[INFO] Dataset shape: {df.shape}")

    return df


# =========================
# STEP 3: Initial Inspection
# =========================
def inspect_data(df):
    print("\n[INFO] Dataset Columns:")
    print(df.columns)

    print("\n[INFO] Missing Values:")
    print(df.isna().sum().sort_values(ascending=False))


# =========================
# STEP 4: Data Cleaning
# =========================
def clean_accident_data(df):
    print("\n[INFO] Cleaning accident data...")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Common latitude/longitude naming
    lat_cols = [c for c in df.columns if "lat" in c]
    lon_cols = [c for c in df.columns if "lon" in c]

    if not lat_cols or not lon_cols:
        raise ValueError(
            "Latitude/Longitude columns not found. "
            "Geocoding will be required in next step."
        )

    lat_col = lat_cols[0]
    lon_col = lon_cols[0]

    df = df.dropna(subset=[lat_col, lon_col])

    df = df[
        (df[lat_col].between(-90, 90)) &
        (df[lon_col].between(-180, 180))
    ]

    print(f"[INFO] Cleaned dataset shape: {df.shape}")

    return df, lat_col, lon_col


# =========================
# STEP 5: Convert to GeoDataFrame
# =========================
def convert_to_geodataframe(df, lat_col, lon_col):
    print("\n[INFO] Converting to GeoDataFrame...")

    geometry = [
        Point(xy) for xy in zip(df[lon_col], df[lat_col])
    ]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326"
    )

    print("[INFO] GeoDataFrame created successfully")
    return gdf


# =========================
# STEP 6: Save Clean Output
# =========================
def save_clean_data(gdf):
    os.makedirs("data", exist_ok=True)
    output_path = "data/clean_india_accidents.geojson"
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"[INFO] Clean GeoJSON saved at: {output_path}")

# =========================
# STEP 7: Load Road Network
# =========================
def load_road_network(gdf_accidents, buffer_km=5):
    print("\n[INFO] Downloading road network from OpenStreetMap...")

    # Use accident centroid as region center
    center = gdf_accidents.geometry.unary_union.centroid
    center_lat, center_lon = center.y, center.x

    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=buffer_km * 1000,
        network_type="drive"
    )

    print("[INFO] Road network downloaded")
    return G

# =========================
# STEP 8: Roads to GeoDataFrame
# =========================
def roads_to_gdf(G):
    print("\n[INFO] Converting road graph to GeoDataFrame...")

    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()

    # Keep only useful columns
    edges = edges[
        ["u", "v", "highway", "length", "geometry"]
    ]

    print(f"[INFO] Road segments: {len(edges)}")
    return edges

# =========================
# STEP 9: Accident Density
# =========================
def compute_accident_density(gdf_accidents, road_edges):
    print("\n[INFO] Mapping accidents to road segments...")

    # Ensure same CRS
    road_edges = road_edges.to_crs(epsg=4326)
    gdf_accidents = gdf_accidents.to_crs(epsg=4326)

    # Spatial join
    joined = gpd.sjoin(
        road_edges,
        gdf_accidents,
        how="left",
        predicate="intersects"
    )

    # Count accidents per road segment
    density = (
        joined
        .groupby(joined.index)
        .size()
        .rename("accident_count")
    )

    road_edges["accident_count"] = density
    road_edges["accident_count"] = road_edges["accident_count"].fillna(0)

    # Normalize by length (accidents per km)
    road_edges["accidents_per_km"] = (
        road_edges["accident_count"] /
        (road_edges["length"] / 1000)
    )

    print("[INFO] Accident density computed")
    return road_edges

# =========================
# STEP 10: Save Road Risk Data
# =========================
def save_road_risk_data(road_edges):
    os.makedirs("data", exist_ok=True)
    output_path = "data/road_accident_density.geojson"
    road_edges.to_file(output_path, driver="GeoJSON")
    print(f"[INFO] Road risk data saved at: {output_path}")


# =========================
# STEP 11: Normalize Metrics
# =========================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

# =========================
# STEP 12: Road Complexity
# =========================
def compute_road_complexity(road_edges):
    print("\n[INFO] Computing road complexity score...")

    # Shorter segments → usually junctions / curves
    road_edges["complexity_raw"] = 1 / (road_edges["length"] + 1)

    road_edges["road_complexity_score"] = normalize(
        road_edges["complexity_raw"]
    )

    return road_edges

# =========================
# STEP 13: Road Type Risk
# =========================
ROAD_TYPE_RISK = {
    "motorway": 0.9,
    "trunk": 0.8,
    "primary": 0.7,
    "secondary": 0.6,
    "tertiary": 0.5,
    "residential": 0.4,
    "service": 0.3
}

def compute_road_type_risk(road_edges):
    print("\n[INFO] Computing road type risk...")

    def map_risk(highway):
        if isinstance(highway, list):
            highway = highway[0]
        return ROAD_TYPE_RISK.get(highway, 0.5)

    road_edges["road_type_risk"] = road_edges["highway"].apply(map_risk)
    return road_edges


# =========================
# STEP 14: ADAS Risk Score
# =========================
def compute_adas_risk_score(road_edges):
    print("\n[INFO] Computing ADAS risk score...")

    road_edges["accident_density_score"] = normalize(
        road_edges["accidents_per_km"]
    )

    road_edges["adas_risk_score"] = (
        0.5 * road_edges["accident_density_score"] +
        0.3 * road_edges["road_complexity_score"] +
        0.2 * road_edges["road_type_risk"]
    )

    road_edges["adas_risk_score"] = (
        road_edges["adas_risk_score"] * 100
    ).round(2)

    return road_edges

# =========================
# STEP 15: Save Final Risk
# =========================
def save_final_risk_map(road_edges):
    os.makedirs("outputs/maps", exist_ok=True)
    path = "outputs/maps/adas_risk_map.geojson"
    road_edges.to_file(path, driver="GeoJSON")
    print(f"[INFO] Final ADAS risk map saved: {path}")

# =========================
# STEP 16: Risk Color Map
# =========================
def get_risk_color(score):
    if score < 30:
        return "green"
    elif score < 60:
        return "orange"
    else:
        return "red"

# =========================
# STEP 17: Create Heatmap
# =========================
def create_interactive_map(road_edges):
    print("\n[INFO] Creating interactive ADAS risk map...")

    center = road_edges.geometry.unary_union.centroid
    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=13,
        tiles="cartodbpositron"
    )

    for _, row in road_edges.iterrows():
        if row.geometry is None:
            continue

        coords = [
            (lat, lon)
            for lon, lat in row.geometry.coords
        ]

        popup_text = (
            f"<b>ADAS Risk Score:</b> {row['adas_risk_score']}<br>"
            f"<b>Accidents / km:</b> {round(row['accidents_per_km'], 2)}<br>"
            f"<b>Road Type:</b> {row['highway']}"
        )

        folium.PolyLine(
            locations=coords,
            color=get_risk_color(row["adas_risk_score"]),
            weight=5,
            opacity=0.8,
            popup=popup_text
        ).add_to(m)

    print("[INFO] Interactive map created")
    return m

# =========================
# STEP 18: Save Map
# =========================
def save_interactive_map(m):
    os.makedirs("outputs/maps", exist_ok=True)
    path = "outputs/maps/adas_risk_heatmap.html"
    m.save(path)
    print(f"[INFO] Interactive map saved at: {path}")

# =========================
# STEP 19: Weather Risk API
# =========================
def fetch_weather_risk(lat, lon):
    """
    Returns weather risk score between 0 and 1
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation_probability,relativehumidity_2m,temperature_2m"
        "&forecast_days=1"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        hourly = data.get("hourly", {})
        rain_prob = max(hourly.get("precipitation_probability", [0]))
        humidity = max(hourly.get("relativehumidity_2m", [0]))
        temperature = hourly.get("temperature_2m", [20])

        # Rain risk (0–1)
        rain_risk = rain_prob / 100.0

        # Fog risk proxy
        fog_risk = (
            0.7 if humidity > 85 and min(temperature) < 20 else 0.2
        )

        weather_risk = (0.6 * rain_risk) + (0.4 * fog_risk)
        return round(weather_risk, 2)

    except Exception as e:
        return 0.3  # conservative default

# =========================
# STEP 20: Add Weather Risk
# =========================
def add_weather_risk(road_edges):
    print("\n[INFO] Adding weather-based risk...")

    # Use centroid of whole region (OEM-style approximation)
    center = road_edges.geometry.unary_union.centroid
    weather_risk = fetch_weather_risk(center.y, center.x)

    road_edges["weather_risk"] = weather_risk
    print(f"[INFO] Weather risk applied: {weather_risk}")

    return road_edges

# =========================
# STEP 21: Final ADAS Score
# =========================
def compute_final_adas_risk(road_edges):
    print("\n[INFO] Computing final ADAS risk score (with weather)...")

    road_edges["final_adas_risk"] = (
        0.4 * road_edges["accident_density_score"] +
        0.25 * road_edges["road_complexity_score"] +
        0.2 * road_edges["road_type_risk"] +
        0.15 * road_edges["weather_risk"]
    )

    road_edges["final_adas_risk"] = (
        road_edges["final_adas_risk"] * 100
    ).round(2)

    return road_edges

# =========================
# STEP 22: Save Weather Map
# =========================
def save_weather_risk_map(road_edges):
    path = "outputs/maps/adas_risk_weather.geojson"
    road_edges.to_file(path, driver="GeoJSON")
    print(f"[INFO] Weather-aware risk map saved: {path}")


# =========================
# MAIN PIPELINE
# =========================
def main():
    dataset_path = download_dataset()
    df = load_accident_data(dataset_path)
    inspect_data(df)
    df_clean, lat_col, lon_col = clean_accident_data(df)
    gdf = convert_to_geodataframe(df_clean, lat_col, lon_col)
    save_clean_data(gdf)
    print("\n✅ STEP 1 COMPLETED: Accident data ready for ADAS risk analysis")

    # STEP 2: Road Network + Density
    G = load_road_network(gdf)
    road_edges = roads_to_gdf(G)
    road_risk = compute_accident_density(gdf, road_edges)
    save_road_risk_data(road_risk)
    print("\n✅ STEP 2 COMPLETED: India accident data for ADAS risk analysis")

    # STEP 3: ADAS Risk Engine
    road_risk = compute_road_complexity(road_risk)
    road_risk = compute_road_type_risk(road_risk)
    road_risk = compute_adas_risk_score(road_risk)
    save_final_risk_map(road_risk)
    print("\n✅ STEP 3 COMPLETED: GeoJSON ready for visualization")

    # STEP 4: Interactive Visualization
    m = create_interactive_map(road_risk)
    save_interactive_map(m)
    print("\n✅ STEP 4 COMPLETED: Interactive Leaflet map")

    # STEP 5: Weather-Based Risk
    road_risk = add_weather_risk(road_risk)
    road_risk = compute_final_adas_risk(road_risk)
    save_weather_risk_map(road_risk)
    print("\n✅ STEP 5 COMPLETED: Live environmental risk (rain/fog)")



if __name__ == "__main__":
    main()
