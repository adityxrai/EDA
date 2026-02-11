🏠 House Price Analysis & Prediction System

An end-to-end Data Science project covering Exploratory Data Analysis (EDA), Statistical Analysis, Clustering, Anomaly Detection, and Machine Learning modeling on a real-world house price dataset (4600 records, 18 features).

📌 Project Overview

This project is divided into three major phases:

Phase I – Exploratory Data Analysis (EDA)

Phase II – Advanced Statistical Analysis, Clustering & Anomaly Detection

Phase III – Predictive Modeling & Model Optimization

The goal is to understand price-driving factors and build an accurate regression model to predict house prices.

🔎 Phase I – Exploratory Data Analysis
✅ Data Cleaning

Converted date column to datetime format

Handled missing values (median/mode imputation)

Removed duplicates

Detected and treated outliers using IQR method

Applied log transformation on price for normalization

📊 Key Insights from EDA

Highly Right-Skewed Price Distribution

Skewness ≈ 24.79

Log transformation significantly improved distribution.

Strong Correlations

sqft_living ↔ sqft_above → 0.87

sqft_living ↔ bathrooms → 0.76

sqft_living ↔ price → Strong positive relationship

Waterfront Properties

Average price of waterfront homes ≈ 2.6x higher

Strong statistical association (Chi-square p < 0.001)

City-Level Variation

Highest avg prices: Mercer Island, Bellevue, Seattle

Lower price clusters: Auburn, Federal Way

Time Series Trend

Stable upward monthly trend across 2014

No major seasonal volatility

📊 Phase II – Advanced Analysis
📈 Statistical Analysis

Computed skewness, kurtosis, quantiles

Identified 1,152 IQR-based outliers

Performed Chi-square tests for categorical relationships

🔬 Dimensionality Reduction

PCA (2D projection)

First two components explain:

PC1 = 33.7% variance

PC2 = 17.6% variance

🔍 Clustering Analysis

Used:

KMeans

Agglomerative Clustering

Gaussian Mixture Model (GMM)

🧠 KMeans Results

Optimal clusters: k = 9

Identified segments such as:

Luxury homes

Large-lot suburban houses

Renovated basement-heavy houses

Ultra-high-value outliers ($19M+ homes)

🚨 Anomaly Detection

Local Outlier Factor (LOF) → 92 anomalies

DBSCAN → 443 noise points

Extreme properties (large lot size, high price) flagged

🤖 Phase III – Predictive Modeling
🎯 Target

price

🧹 Feature Engineering

Extracted year, month, day

Extracted state & zipcode

Applied log transformation

OneHotEncoded categorical features

Standardized numeric features

📌 Models Implemented

Linear Regression

Ridge Regression

Random Forest

Gradient Boosting

Random Forest (Hyperparameter Tuned via RandomizedSearchCV)

📊 Model Performance
Model	RMSE	MAE	R²
Gradient Boosting	169,901	106,822	0.658
Linear Regression	170,503	109,749	0.656
Ridge	170,706	109,958	0.655
Random Forest (Tuned)	173,567	106,648	0.643
🏆 Best Overall Model:

Gradient Boosting Regressor

R² ≈ 0.66

RMSE ≈ 170k

🔑 Feature Importance (Permutation Importance)

Top price-driving features:

sqft_living (Most Important)

city_Seattle

sqft_above

zipcode

city_Bellevue

yr_built

view

bathrooms

📌 Key Insight:
House size and location are the strongest predictors of price.

📂 Project Structure
├── Phase I - EDA
├── Phase II - Clustering & Advanced Analysis
├── Phase III - Modeling
├── best_house_price_model.joblib
└── README.md

🛠 Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn, Plotly

Scikit-learn

SciPy

PCA

Clustering (KMeans, Agglomerative, GMM)

LOF & DBSCAN

RandomizedSearchCV

🚀 How to Run
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy joblib


Then run the notebook or script sequentially by phases.

📌 Business Insights Summary

Larger houses significantly increase price.

Waterfront properties drastically increase valuation.

Certain cities add strong location premium.

Renovation year positively impacts price clusters.

Log transformation improves predictive stability.

Tree-based models outperform linear ones slightly.

📦 Model Export

Saved final tuned model:

best_house_price_model.joblib


You can load it using:

import joblib
model = joblib.load("best_house_price_model.joblib")

🏁 Conclusion

This project demonstrates:

Strong EDA capability

Advanced statistical analysis

Clustering & anomaly detection

End-to-end ML pipeline building

Model evaluation & optimization

Feature importance interpretation
