# Customer Retention Prediction — "One-Click" Online Store

## Project Description

The "One-Click" online store has seen a decline in purchasing activity among its regular customers. Since acquiring new buyers is becoming less cost-effective, the company wants to focus on **retaining existing customers** through personalized offers, backed by data and modeling.

### Goals
- Build a model that predicts the probability of a decrease in a customer's purchasing activity.
- Segment customers based on behavior and predicted risk.
- Translate findings into concrete recommendations for improving retention.

## Data

The project uses four source tables:

| File | Description |
|---|---|
| `market_file.csv` | Customer-level marketing and behavioral features (activity, service type, category views, etc.) |
| `market_money.csv` | Revenue per customer, by month (current, previous, pre-previous) |
| `market_time.csv` | Time spent on the site per customer, by month |
| `money.csv` | Customer profit data (`;`-separated, comma decimals) |

Only customers with recorded purchases across **all** three tracked months were kept, to ensure the model is trained on genuinely active users.

## Methodology

1. **Data loading & preprocessing** — column name normalization, duplicate removal, and fixing implicit duplicates in categorical values (e.g. typos in service type and time period labels).
2. **Exploratory data analysis (EDA)** — distribution plots and boxplots for all key numeric and categorical features (marketing activity, duration, promotional purchase share, category views, service errors, revenue, time on site, profit, etc.), with outlier filtering (e.g. removing revenue outliers above 20,000).
3. **Table merging** — `market_file`, pivoted `market_money`, and pivoted `market_time` are joined into a single customer-level dataset.
4. **Correlation analysis** — a Phik correlation matrix is used to check for multicollinearity among mixed numeric/categorical features. No pair exceeded the critical threshold (0.9).
5. **Modeling pipeline** — a `scikit-learn` `Pipeline` with a `ColumnTransformer` handles:
   - One-hot encoding for nominal categorical features
   - Ordinal encoding for service type (standard vs. premium)
   - Scaling (`StandardScaler` / `MinMaxScaler`, tuned) for numeric features
6. **Model selection** — `RandomizedSearchCV` compares four algorithms: `KNeighborsClassifier`, `DecisionTreeClassifier`, `SVC`, and `LogisticRegression`, optimizing for **ROC-AUC** (chosen for its robustness to class imbalance and independence from a classification threshold).
7. **Feature importance** — SHAP values explain which features drive the final model's predictions.
8. **Customer segmentation** — a high-value customer segment (above-median promotional purchases, category views, and marketing activity) is isolated and split into low- vs. high-activity-probability groups using the trained model, then compared feature-by-feature.

## Results

- **Best model:** `SVC` with `kernel='sigmoid'`, `C=8`
- **ROC-AUC:** 0.86 (cross-validated), confirmed on the held-out test set
- **Most influential features:** promotional purchase share, average category views per visit, and marketing activity over 6 months
- **Least influential features:** popular product category (kitchenware/children's goods) and mailing list permission

### Segmentation Insight

Within the high-value customer segment:
- Customers with a **high** predicted probability of continued activity tend to make more promotional purchases but spend less time and view fewer pages.
- Customers with a **low** predicted probability view more pages/categories but make fewer promotional purchases — suggesting they may be struggling to find relevant products or offers.

### Recommendations

- Offer personalized promotions and product recommendations tailored to individual interests.
- Improve site navigation to help users reach relevant promotional items faster.
- Set up personalized notifications for customers in the low-activity-probability group to re-engage them before they churn.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
phik
```

Install extra dependencies (if not already available) with:
```bash
pip install shap phik
```

## Usage

Open and run `supervised_learning.ipynb` in Jupyter. The notebook is organized sequentially — data loading → preprocessing → EDA → merging → correlation analysis → modeling → feature importance → segmentation — and each section can be run top to bottom.

## Project Structure

```
.
├── supervised_learning.ipynb   # Main analysis and modeling notebook
├── market_file.csv             # Customer marketing/behavioral features
├── market_money.csv            # Monthly revenue per customer
├── market_time.csv             # Monthly time-on-site per customer
├── money.csv                   # Customer profit data
└── README.md
```
