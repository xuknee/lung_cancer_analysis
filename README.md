# 🫁 Lung Cancer Data Analysis

## 📌 Overview
This project provides an **end-to-end EDA + predictive modeling** workflow for lung cancer patient data from Kaggle.  
Beyond demographics and clinical feature exploration, it trains an **XGBoost classifier** to predict survival and visualizes the **top 10 most important features**.

---

## 📊 Dataset
Source: [Lung Cancer Dataset by Khwaish Saxena](https://www.kaggle.com/datasets/khwaishsaxena/lung-cancer-dataset)

**To get started:**
1. Download `lung cancer.csv` from Kaggle.
2. Place it in your project folder or update the file path in the script/notebook.

---

## ⚙️ Requirements
- **Python:** 3.7+ (recommended: 3.8/3.9/3.10+)
- **Libraries:**
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - **`xgboost`** (modeling)

*Optional utilities (for split/metrics/profiling):*  
`scikit-learn`, `ydata_profiling`

---

## 📦 Installation
```bash
pip install pandas numpy matplotlib seaborn xgboost
# Optional (if you use train_test_split / accuracy_score / profiling):
pip install scikit-learn ydata-profiling
