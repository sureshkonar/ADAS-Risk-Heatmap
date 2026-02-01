# RoadSentinel – ADAS Heatmaps & Analysis

**Author:** Suresh Konar
**Purpose:** OEM-style ADAS risk heatmap visualization for India with AEB/LDW overlays, weather & road-type weighting.

---

## 🛠 Features

* ADAS Risk Heatmap for selected cities or full India
* Separate AEB (Autonomous Emergency Braking) & LDW (Lane Departure Warning) risk layers
* Weather-based risk adjustment (rain/fog)
* Road-type weighting (NH / SH / Urban)
* Interactive PyDeck maps with hover tooltips
* Analysis charts for AEB / LDW risk zones
* KPIs: High Risk Cells, Grid Cells, Max Risk

---

## 🗺 Supported Locations

### States & Cities

* **Maharashtra:** Mumbai, Pune, Nagpur, Nashik, Aurangabad

* **Karnataka:** Bengaluru, Mysore, Mangalore, Hubli, Dharwad

* **Tamil Nadu:** Chennai, Coimbatore, Madurai, Salem, Tiruchirappalli

* **Delhi:** Delhi

* **West Bengal:** Kolkata, Howrah, Durgapur

* **Gujarat:** Ahmedabad, Surat, Vadodara

* **Madhya Pradesh:** Indore, Bhopal, Gwalior

* **Rajasthan:** Jaipur, Jodhpur, Udaipur

* **Full India option** available via dropdown to view the entire country dataset.

---

## ⚡ Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/username/adas-risk-heatmap.git
cd adas-risk-heatmap
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app:**

```bash
streamlit run app.py
```

---

## 📁 Data Requirements

* **Local CSV option:** Place the `India_Road_Accident_Dataset.csv` in the project folder.
* **Kaggle API option:** Ensure your Kaggle API key is available if using `kagglehub` to download the dataset automatically.

---

## ⚠️ Disclaimer

* Data sources are **publicly available**.
* Risk visualization is **indicative** and intended for **research / OEM analysis purposes only**.
* Weather data is **forecast-based** and may not reflect real-time conditions.
* Users should **not rely on this tool for actual driving or safety decisions**.

---

## 📝 Notes

* **AEB:** Autonomous Emergency Braking
* **LDW:** Lane Departure Warning
* **Grid Size:** Adjustable to control heatmap granularity.
* **Color Legend:**

  * Green: Low risk
  * Orange: Medium risk
  * Red: High risk

---

## 📚 Technology & Libraries

* **Python 3.10+**
* **Streamlit:** Web app UI
* **PyDeck:** Interactive maps & heatmaps
* **Plotly Express:** Graphical analysis for risk zones
* **Pandas / NumPy:** Data processing
* **Requests:** Fetch weather forecast
* **KaggleHub:** Download India road accident dataset

---

## 🔧 Usage

1. Select the **State** and **City** or choose **Full India**.
2. Adjust **Grid Size** to control heatmap resolution.
3. Enable or disable **Weather Risk** and **Road-Type Weighting**.
4. Run the pipeline to generate interactive maps and analysis.
5. Explore AEB and LDW zones via separate tabs.

---

## 💡 Future Enhancements

* Integration with real-time traffic data.
* Precise shapefile-based city boundaries.
* Advanced ADAS module predictions.
* Multi-day weather and hazard forecasting.

---

**Author Contact:** [msuresh9122002@gmail.com](mailto:msuresh9122002@gmail.com)
