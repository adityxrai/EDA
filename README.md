# 🏠 House Price Analysis & Prediction System  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Complete-success)
![License](https://img.shields.io/badge/Type-End--to--End%20ML-purple)

An end-to-end Data Science project covering **EDA, Statistical Analysis, Clustering, Anomaly Detection, and Machine Learning Modeling** on real-world housing data (4,600 records, 18 features).

---

## 📌 Project Objectives

✔ Understand key drivers of house prices  
✔ Perform advanced statistical & clustering analysis  
✔ Detect anomalies and extreme properties  
✔ Build optimized regression models  
✔ Interpret feature importance  

---

# 📊 Phase I — Exploratory Data Analysis (EDA)

### 🔎 Data Preprocessing
- Converted `date` to datetime
- Extracted `year`, `month`, `day`
- Handled missing values (median/mode imputation)
- Removed duplicates
- Applied IQR-based outlier filtering
- Log-transformed `price` for normalization

### 📈 Key Insights

- **Highly right-skewed price distribution** (Skew ≈ 24.79)
- Strong positive correlation:
  - `sqft_living ↔ sqft_above` → **0.87**
  - `sqft_living ↔ price` → Strong relationship
- Waterfront properties are **2.6× more expensive**
- Location premium observed in:
  - Mercer Island
  - Bellevue
  - Seattle

📌 **Conclusion:** House size and location dominate pricing behavior.

---

# 🔬 Phase II — Advanced Analytics

## 📊 Statistical Analysis
- Skewness & Kurtosis evaluation
- Quantile & IQR-based outlier detection
- Chi-square tests for categorical relationships

## 🧠 PCA (Dimensionality Reduction)

- PC1 → 33.7% variance
- PC2 → 17.6% variance
- Clear structural separation of high-value homes

---

## 🧩 Clustering Techniques Applied

| Algorithm | Purpose |
|-----------|----------|
| KMeans | Property segmentation |
| Agglomerative | Hierarchical grouping |
| GMM | Probabilistic clustering |
| DBSCAN | Density-based anomaly detection |
| LOF | Local outlier detection |

### 📌 KMeans Results
- Optimal clusters: **k = 9**
- Identified:
  - Luxury homes
  - Renovated basement-heavy houses
  - Large suburban properties
  - Ultra-high-value outliers ($19M+)

---

# 🤖 Phase III — Predictive Modeling

## 🎯 Target Variable
`price`

## 🛠 Feature Engineering
- OneHotEncoding for categorical variables
- StandardScaler for numeric features
- Log transformation of price
- Train-test split (80-20)

---

## 📈 Model Comparison

| Model | RMSE | MAE | R² |
|--------|--------|--------|--------|
| Gradient Boosting | **169,901** | 106,822 | **0.658** |
| Linear Regression | 170,503 | 109,749 | 0.656 |
| Ridge | 170,706 | 109,958 | 0.655 |
| Random Forest (Tuned) | 173,567 | 106,648 | 0.643 |

🏆 **Best Model: Gradient Boosting Regressor**

---

# 🔑 Feature Importance (Top Drivers)

1. 🏡 `sqft_living`
2. 📍 `city_Seattle`
3. 🏢 `sqft_above`
4. 📮 `zipcode`
5. 🌊 `view`
6. 🛁 `bathrooms`

📌 Size + Location = Price

---

# 🚨 Anomaly Detection

- LOF → 92 anomalies
- DBSCAN → 443 density-based outliers
- Extreme properties automatically identified

---

# 📦 Model Export

```python
best_house_price_model.joblib
