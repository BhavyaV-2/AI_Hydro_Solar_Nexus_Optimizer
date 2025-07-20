# 🌊☀️ The AI Hydro-Solar Nexus Arbitrageur
### A Prescriptive AI-Powered System for the Co-Optimization of Water & Electricity Resources

---

### **Live Demo**
**[➡️ Click here to access the live interactive Streamlit application]**  *(After you deploy it, put the URL here)*

---

## 1. The Problem: The High-Stakes Gamble of Renewable Energy

India's sustainable future is built on two pillars: the immense power of our rivers (**Hydropower**) and the abundant energy of the sun (**Solar**). However, these critical resources are currently managed in isolation, forcing grid and reservoir operators into a daily, high-stakes gamble.

*   **The Solar Problem: Intermittency.** Solar power is powerful but unreliable. It vanishes at night and is severely weakened during the crucial monsoon season, creating massive instability on the grid.
*   **The Dam's Dilemma: The Flood-Drought Razor's Edge.** Hydropower is controllable, but operators must constantly choose between releasing too much water (risking future droughts for cities and farms) or storing too much water (risking catastrophic floods during heavy rainfall).

This uncoordinated approach forces a reliance on expensive and polluting fossil fuel "peaker plants" to balance the grid, creating inefficiency, wasting clean energy, and undermining our sustainability goals.

## 2. Our Solution: Transforming a Dam into an AI-Powered "Water Battery"

We have built an end-to-end AI system that solves this problem by forcing these two resources to work as a single, intelligent team. Our project, **The AI Hydro-Solar Nexus Arbitrageur**, is a prescriptive decision-support tool that transforms the hydropower dam into a giant, grid-scale "water battery" for the solar farm.

Our system ingests real-world data, uses a suite of sophisticated AI models to forecast future conditions, and then prescribes the **optimal, hour-by-hour schedule** for the dam's operations to:
1.  **Maximize Revenue** by performing energy arbitrage—storing water when solar is abundant (and power is cheap) and generating hydropower when the sun is gone (and power is expensive).
2.  **Enhance Grid Stability** by using controllable hydropower to smooth out the volatility of solar energy.
3.  **Guarantee Resilience** with core logic that actively prevents both flood and drought conditions.

This is a true "AI for Sustainable Infrastructure" solution, directly optimizing the management of **water** and **electricity** resources.

---

## 3. Technical Architecture & Innovation Pillars

Our project is a complete pipeline demonstrating a deep, multi-stage approach to a complex problem. Our innovation is built on four pillars:

### Pillar 1: Multi-Source Data Integration (The Foundation)
We successfully integrated three distinct, real-world datasets, each with its own timeline and format challenges:
*   **Kaggle Solar Data:** High-resolution generation and weather data from real solar plants (2020).
*   **India-WRIS Reservoir Data:** Official Government of India data on the Srisailam Reservoir (2015-2024).
*   **Indian Energy Exchange (IEX) Price Data:** Volatile, real-world hourly electricity market prices (2022-2024).
*   **Key Innovation:** We developed a robust data engineering pipeline to clean, align, and merge these disparate sources into a single, high-quality "Master Dataset" that forms the basis of our local expert models.

### Pillar 2: The Solar "Digital Twin" (AI Module 1)
*   **Objective:** To learn the timeless physics of solar generation (`Weather -> Power`).
*   **Model:** A novel **Hybrid CNN-LSTM Architecture with an Attention Mechanism**.
*   **Key Innovation:** We created a highly accurate forecaster with an **NMAE of 7.78%**. This model acts as a "Digital Twin," allowing our system to simulate realistic solar output for any weather scenario.

### Pillar 3: The "Local Market Expert" (AI Module 2)
*   **Objective:** To understand the unique economic behavior of the Indian power grid.
*   **Model:** An **XGBoost Regressor**, the state-of-the-art for tabular data.
*   **Key Innovation:** The model was trained exclusively on our merged Indian dataset, making it an expert on local market dynamics. The visualization proves its stunning accuracy in predicting real price volatility.

### Pillar 4: The Prescriptive "AI Operator" (The Final Application)
*   **Objective:** To integrate the expert forecasts into an optimal, actionable plan.
*   **Logic:** A prescriptive optimization algorithm with a clear rule hierarchy: **Safety (Flood/Drought Prevention) > Profit (Energy Arbitrage).**
*   **Key Innovation:** The final interactive **Streamlit dashboard**. This is not just a report; it's a "what-if" simulator that allows a user to define a scenario and receive an immediate, AI-generated operational schedule.

---

## 4. How to Run This Project

### A. The Research and Development Journey (The Notebooks)
The `notebooks/` directory contains the complete R&D history of this project.

1.  **`01_Solar_Forecasting.ipynb`:** Trains and saves the Solar Digital Twin model. Requires the Kaggle API key (`kaggle.json`).
2.  **`02_Reservoir_Analysis.ipynb`:** Shows our initial exploratory analysis of the raw reservoir data from India-WRIS.
3.  **`03_Final_Optimization_System.ipynb`:** The master development notebook. It cleans the IEX data, builds the Master Dataset, trains the Price Forecaster, and contains the logic for the final simulation dashboard.

### B. The Deployed Application (Streamlit)
The `streamlit_app/` folder contains the final, deployable application.

1.  **Place Assets:** Make sure the three required assets (`cnn_lstm_attention_model.keras`, `price_forecaster.json`, `Master_Indian_Data.csv`) are placed inside the `streamlit_app/assets/` directory.
2.  **Install Dependencies:** Navigate into the `streamlit_app` folder in your terminal and run `pip install -r requirements.txt`.
3.  **Run the App:** Run the command **`streamlit run app.py`**.

---

This project represents a complete journey from raw, messy data to a polished, intelligent, and impactful AI decision-support tool. It is our vision for a smarter, more sustainable, and more resilient energy future.
