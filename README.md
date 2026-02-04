# COVID-19 Mobility Forecasting in Thailand

![Research Header](https://img.shields.io/badge/Model-ARIMA%20%7C%20Prophet%20%7C%20x--xgboost-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)

## 📝 Research Overview
This repository contains the official Python implementation for the study analyzing human mobility trends in Thailand during the COVID-19 pandemic (2020–2022). 

## 🎯 Objectives
* **Performance Benchmarking:** Evaluate the accuracy of **ARIMA**, **Facebook Prophet**, and an enhanced **x-xgboost** model.
* **Feature Impact Analysis:** Quantify the influence of government interventions (**Stringency Index**) versus epidemiological data (**Daily Cases**) using Feature Importance metrics (Weight, Gain, and Cover).
* **Robust Evaluation:** Utilize the **Rolling Origin Cross-Validation** (Walk-forward) technique to simulate real-world forecasting scenarios.

## 📊 Data Sources and References
To ensure transparency, the dataset used in this research was integrated from the following reputable sources:

1. **Google COVID-19 Community Mobility Reports:** Used to measure movement trends across different categories of places (Retail, Grocery, Parks, Transit, Workplaces, and Residential).  
   *Source:* [Google Mobility Reports](https://www.google.com/covid19/mobility/)

2. **COVID-19 Epidemiological Data (JHU-CSSE):** Daily confirmed cases and death statistics were obtained from the **Johns Hopkins University Center for Systems Science and Engineering (JHU-CSSE)**.  
   *Source:* [JHU COVID-19 Dashboard / Our World in Data](https://github.com/CSSEGISandData/COVID-19)

3. **Oxford COVID-19 Government Response Tracker (OxCGRT):** The **Stringency Index** and other policy response metrics were sourced from the Blavatnik School of Government, University of Oxford.  
   *Source:* [Oxford COVID-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-dataset)
