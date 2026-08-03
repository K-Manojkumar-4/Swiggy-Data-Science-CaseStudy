# Swiggy Food Delivery Data Analysis & Machine Learning (2025)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

**End-to-end Data Science project** analyzing real-world Swiggy food delivery data across 28 Indian cities (January–August 2025). The project covers the complete data science lifecycle: data cleaning, feature engineering, exploratory data analysis (EDA), statistical analysis, unsupervised learning (K-Means clustering), supervised learning (Linear Regression), business insights, and actionable recommendations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Workflow](#workflow)
- [Key Findings](#key-findings)
- [Machine Learning Results](#machine-learning-results)
- [Business Recommendations](#business-recommendations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## Project Overview

This project performs a comprehensive analysis of nearly **200,000 dish listings** from Swiggy across major Indian cities. The goal is to uncover pricing patterns, rating behaviors, category popularity, geographic trends, and to segment dishes for strategic decision-making.

**Core Objectives:**
- Clean and prepare a large real-world food delivery dataset
- Perform rigorous exploratory and statistical analysis
- Segment dishes using K-Means clustering (Price + Rating + Popularity)
- Build a baseline Linear Regression model for price prediction
- Derive actionable business insights and recommendations for restaurants and platform stakeholders

---

## Dataset Description

| Attribute            | Description                                      | Type     |
|----------------------|--------------------------------------------------|----------|
| State                | Indian state                                     | Object   |
| City                 | City name                                        | Object   |
| Order Date           | Date of order listing                            | DateTime |
| Restaurant Name      | Name of the restaurant                           | Object   |
| Location             | Locality / area                                  | Object   |
| Category             | Menu category (Recommended, Main Course, etc.)   | Object   |
| Dish Name            | Name of the dish                                 | Object   |
| Price (INR)          | Price in Indian Rupees                           | Float    |
| Rating               | Average customer rating (1.0 – 5.0)              | Float    |
| Rating Count         | Number of ratings received                       | Integer  |

- **Size**: ~197,430 records × 10 original features
- **Time Period**: January 2025 – August 2025
- **Coverage**: 28 cities across India
- **Source**: Publicly available Swiggy-related dataset (2025)

---

## Project Structure

```
Swiggy-Food-Delivery-Data-Analysis-2025/
├── data/
│   └── swiggy_data2025.csv              # Raw dataset
├── notebooks/
│   └── 01_end_to_end_analysis.ipynb     # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py            # Cleaning & feature engineering
│   ├── eda_utils.py                     # Reusable EDA functions
│   └── modeling.py                      # Clustering & regression utilities
├── reports/
│   └── executive_summary.md             # Key insights summary
├── outputs/
│   └── figures/                         # Generated plots (saved during run)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/Swiggy-Food-Delivery-Data-Analysis-2025.git
cd Swiggy-Food-Delivery-Data-Analysis-2025

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Workflow

1. **Data Loading & Inspection**  
   Load CSV, examine schema, shape, dtypes, sample records.

2. **Data Cleaning & Feature Engineering**  
   - Missing value analysis (none found)  
   - Duplicate handling  
   - Date parsing (`Order Date` → datetime)  
   - Derived features: `Month`, `Month_num`, `Day`, `Year`  
   - Price outlier awareness

3. **Exploratory Data Analysis (EDA)**  
   - Univariate analysis (Price, Rating distributions)  
   - Bivariate analysis (Price vs Rating, City-wise metrics)  
   - Categorical analysis (Top categories, restaurants, cities)  
   - Geographic & temporal trends

4. **Statistical Analysis**  
   - Descriptive statistics  
   - Correlation analysis  
   - Distribution insights

5. **Machine Learning**  
   - **K-Means Clustering**: Segment dishes into meaningful groups based on Price, Rating, and Rating Count  
   - **Linear Regression**: Predict dish price using Rating and Rating Count (baseline model)

6. **Insights Generation & Recommendations**  
   Translate statistical and ML findings into business language.

---

## Key Findings

1. **Bengaluru** dominates the dataset with the highest number of dish listings.
2. Average dish price ≈ **₹268**, with the interquartile range roughly ₹139 – ₹329.
3. Extremely weak correlation between **Price** and **Rating** (near-zero linear relationship).
4. A significant portion of dishes have **zero or very low Rating Count**, indicating low review engagement.
5. High value-for-money dishes are predominantly priced under ₹200 while maintaining solid ratings.
6. Categories such as “Recommended”, Main Course, and Desserts are highly represented.
7. K-Means successfully identifies distinct segments (budget high-rated, premium, low-engagement, etc.).
8. Linear Regression using only Rating + Rating Count yields limited predictive power (low R²), confirming that price is driven by many other factors (cuisine type, location, restaurant positioning, etc.).

---

## Machine Learning Results

### K-Means Clustering
- Features used: `Price (INR)`, `Rating`, `Rating Count` (scaled)
- Optimal clusters determined via Elbow Method / Silhouette analysis
- Clear separation of dish segments useful for targeted promotions and menu optimization

### Linear Regression (Price Prediction)
| Metric              | Value   |
|---------------------|---------|
| Mean Absolute Error | ~123 INR |
| R² Score            | ~0.012  |

**Interpretation**: Rating and Rating Count alone are weak predictors of price. Additional features (cuisine, city tier, restaurant rating history, discounts, etc.) are required for a production-grade pricing model.

---

## Business Recommendations

- **Increase Review Collection**: Incentivize customers to rate dishes; low Rating Count reduces ranking visibility and trust.
- **Promote Value-for-Money Items**: Highlight high-rated dishes under ₹200 more aggressively on the home feed and search.
- **City-Specific Pricing Strategy**: Restaurants in high-average-price cities should expand mid and budget offerings.
- **Curate “Recommended” Section Carefully**: This category performs strongly — maintain quality control.
- **Segmented Campaigns**: Use cluster labels to design targeted offers (e.g., premium cluster upselling, budget cluster volume campaigns).
- **Feature Enrichment for Future Models**: Collect cuisine type, delivery time SLA, discount flags, restaurant chain status, and historical demand to improve predictive accuracy.

---

## Technologies Used

- **Python 3.10+**
- **Pandas / NumPy** – Data manipulation
- **Matplotlib / Seaborn / Plotly** – Visualization
- **scikit-learn** – K-Means, Linear Regression, preprocessing, metrics
- **SciPy / statsmodels** – Statistical support
- **Jupyter** – Interactive analysis

---

## How to Run

```bash
# Start Jupyter
jupyter notebook

# Open and run
notebooks/01_end_to_end_analysis.ipynb
```

All figures are saved automatically to `outputs/figures/`.  
Key numerical results and insights are also summarized in `reports/executive_summary.md`.

---

## Future Improvements

- Incorporate additional features (cuisine, delivery metrics, discounts)
- Try advanced models (XGBoost, LightGBM, neural nets) for price prediction
- Time-series analysis of ordering patterns
- Restaurant-level performance scoring and ranking
- Interactive Streamlit / Gradio dashboard
- Deploy clustering results as a recommendation engine prototype

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Author

**Data Science Portfolio Project**  
Built with real-world food delivery data • Full pipeline from raw data to business recommendations.

Feel free to star ⭐ the repository if you find it useful, and open issues for suggestions or improvements.

---

*Last updated: August 2025*
