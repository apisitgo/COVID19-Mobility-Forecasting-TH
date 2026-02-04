# COVID-19 Mobility Forecasting in Thailand

![Research Header](https://img.shields.io/badge/Model-ARIMA%20%7C%20Prophet%20%7C%20x--xgboost-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)

## 📝 Research Overview
This repository contains the official Python implementation for the study analyzing human mobility trends in Thailand during the COVID-19 pandemic (2020–2022). 

## 🎯 Objectives
* **Performance Benchmarking:** Evaluate the accuracy of **ARIMA**, **Facebook Prophet**, and an enhanced **x-xgboost** model.
* **Feature Impact Analysis:** Quantify the influence of government interventions (**Stringency Index**) versus epidemiological data (**Daily Cases**) using Feature Importance metrics (Weight, Gain, and Cover).
* **Robust Evaluation:** Utilize the **Rolling Origin Cross-Validation** (Walk-forward) technique to simulate real-world forecasting scenarios.

## 🛠️ Methodology

### 1. Data Preprocessing & Feature Engineering
* **Data Sources:** Google Community Mobility Reports, Stringency Index (Oxford Tracker), and COVID-19 daily case statistics.
* **Engineered Features:** Lockdown levels (categorized by severity), public holidays, and temporal variables (day of week, week number, seasonality).

### 2. Forecasting Framework
To ensure **Transparency and Reproducibility** (as recommended by peer reviewers), this project implements:
* **Rolling Origin Approach:** Instead of a single train-test split, we use a rolling window to evaluate model stability over time.
* **Model Configurations:**
    * **ARIMA:** Robust automated parameter selection tailored for small data samples in specific waves.
    * **Prophet:** Configured to handle additive seasonality and Thai public holiday effects.
    * **x-xgboost:** An optimized XGBRegressor utilizing 9 key features to capture non-linear behavioral shifts.
