# 🚗 RoadSentinel – ADAS Heatmaps & Analysis

**Author:** Suresh Konar  
**Purpose:** OEM-style ADAS (Advanced Driver Assistance Systems) risk heatmap for India, including AEB (Autonomous Emergency Braking) and LDW (Lane Departure Warning) risk overlays, with weather and road-type weighting.

---

## 🔹 Description

The RoadSentinel – ADAS Heatmaps & Analysis is an interactive tool built with **Streamlit** that visualizes road accident risk across India. It provides:

- **ADAS Risk Heatmap:** Aggregate risk based on accident density.
- **AEB and LDW Risk Layers:** Separate risk overlays for autonomous emergency braking and lane departure warning systems.
- **Weather Risk:** Rain/fog risk multiplier applied to accident data.
- **Road-Type Weighting:** National Highways (NH), State Highways (SH), and Urban roads weighted differently to reflect realistic OEM analysis.
- **State / City Selection:** Zoom into individual cities or view all of India.
- **Interactive Analysis:** Separate plots for AEB vs LDW risk distribution.

---

## ⚙️ Features

1. **Interactive Map**
   - Heatmap of ADAS risk.
   - Separate scatter layers for AEB (red) and LDW (yellow) zones.
   - Toggleable by tabs for readability.

2. **Weather Risk**
   - Fetches rain/fog data from [Open-Meteo API](https://open-meteo.com/).
   - Applies a multiplier to risk scores dynamically.

3. **Road-Type Weighting**
   - NH: 1.2
   - SH: 1.0
   - Urban: 0.8
   - Weighted into risk calculations.

4. **Analytics**
   - KPIs: Number of grid cells, high-risk cells, max risk.
   - Bar charts comparing AEB vs LDW zones.

5. **Data Sources**
   - [India Road Accident Dataset](https://www.kaggle.com/datasets/data125661/india-road-accident-dataset)
   - Open-Meteo weather API.
   - Transparent: users can access raw data locally or via Kaggle.

6. **Cities / States Covered**
   - Maharashtra: Mumbai, Pune, Nagpur
   - Karnataka: Bengaluru, Mysore, Mangalore
   - Tamil Nadu: Chennai, Coimbatore, Madurai
   - Delhi: Delhi
   - **All India View:** Option to see aggregated data for the entire country.

---

## 📌 Technology Stack

| Layer          | Technology / Library          | Purpose                                           |
|----------------|-------------------------------|-------------------------------------------------|
| Frontend / UI  | Streamlit                     | Web app framework for interactive visualization |
| Map Visualization | PyDeck / Deck.gl            | Heatmaps and scatter overlays for ADAS risk     |
| Charts         | Plotly Express               | Bar charts for AEB / LDW analysis               |
| Data Handling  | Pandas / NumPy               | Data cleaning, aggregation, normalization       |
| API            | Requests                     | Fetch weather data                               |
| Data Download  | KaggleHub (optional)         | Download India road accident dataset            |

---

## ⚡ Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/username/adas-risk-heatmap.git
cd adas-risk-heatmap

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
