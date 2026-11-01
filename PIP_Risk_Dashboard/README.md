# Workforce Analytics & PIP Risk Prediction

A Machine Learning project that explores workforce performance indicators and uses **Logistic Regression** to identify patterns associated with Performance Improvement Plan (PIP) requirements.

The project also includes an interactive dashboard built using **Streamlit** to present workforce insights, performance comparisons and model evaluation results.

## Project Overview

Organizations monitor several employee performance indicators, including attendance, KPI scores, quality scores and goal completion rates.

This project analyzes these indicators to understand how they are associated with PIP status and builds a binary classification model to identify potential PIP-related patterns.

**Important:** Model predictions are intended only as analytical signals for human review and employee support. They must never be used as automatic employment or PIP decisions.

## Objectives

- Perform data cleaning and exploratory data analysis (EDA).
- Analyze workforce and employee performance indicators.
- Identify patterns associated with PIP status.
- Build a Logistic Regression classification model.
- Evaluate the model using classification metrics.
- Develop an interactive Streamlit dashboard.

## Dataset Overview

- Total records: 6,315
- Total columns: 26
- Input features: 25
- Target variable: `pip_required`

**Target variable:**
- `0` – No PIP
- `1` – PIP Required

The dataset contains workforce-related information, including department, job role, employment type, attendance, KPI score, quality score, goal completion rate and training hours.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Logistic Regression
- Streamlit
- Joblib
- Jupyter Notebook

## Project Workflow

1. Data loading and inspection
2. Data cleaning and duplicate handling
3. Missing-value treatment
4. Exploratory Data Analysis
5. Feature preparation and preprocessing
6. Train-test split
7. Logistic Regression baseline model
8. Hyperparameter tuning using GridSearchCV
9. Model evaluation
10. Coefficient interpretation
11. Model saving using Joblib
12. Streamlit dashboard development

## Model Performance

The tuned Logistic Regression model achieved the following results on the held-out test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 80.77% |
| Precision | 51.41% |
| Recall | 79.68% |
| F1 Score | 62.50% |
| ROC-AUC | 88.33% |
| PR-AUC | 70.38% |

### Confusion Matrix

| | Predicted No PIP | Predicted PIP |
|---|---:|---:|
| Actual No PIP | 808 | 189 |
| Actual PIP | 51 | 200 |

The model identifies a substantial proportion of actual positive cases, with a recall of 79.68%. However, its precision of 51.41% indicates that many predicted PIP cases are false positives. This limitation must be considered before interpreting predictions.

## Key Business Insights

The dataset showed differences in average performance indicators between employees labelled No PIP and PIP Required.

| Indicator | No PIP | PIP Required |
|---|---:|---:|
| Attendance Percentage | 95.17 | 93.48 |
| KPI Score | 70.39 | 54.51 |
| Quality Score | 72.42 | 58.85 |
| Goal Completion Rate | 73.03 | 57.77 |
| Training Hours (Last 3 Months) | 18.20 | 17.09 |

Employees labelled No PIP had higher average values across these indicators.

These are descriptive associations within the dataset. They do not establish causation.

## Streamlit Dashboard

The interactive dashboard includes:

- Workforce overview and KPI cards
- Department-wise employee distribution
- PIP rate by department
- Employment type distribution
- Job role analysis
- Performance indicator comparison
- Key business insights
- Logistic Regression evaluation metrics
- Confusion matrix
- Model interpretation

## Project Structure

```text
PIP_Risk_Dashboard/
│
├── app.py
├── PIP_Risk_Dataset - LMS.csv
├── pip_logistic_regression_model.pkl
├── README.md
└── requirements.txt
```

Keep the dataset out of a public repository unless you have permission to share it. If necessary, provide instructions for obtaining it separately.

## How to Run the Dashboard

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project folder**

```bash
cd PIP_Risk_Dashboard
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run Streamlit**

```bash
python -m streamlit run app.py
```

## Limitations and Ethical Considerations

- The model's predictions are not definitive assessments of employee performance.
- False positives may incorrectly flag employees.
- Dataset patterns may reflect historical bias.
- Predictions should not determine PIP eligibility or employment outcomes.
- Human review, context and fairness checks are essential.
- Model performance may change when applied to different organizations or datasets.

## Conclusion

This project demonstrates an end-to-end beginner-level Machine Learning workflow, from data preparation and exploratory analysis to Logistic Regression modelling, evaluation and dashboard development.

It helped me understand how classification models can be used to explore real-world business problems while considering model limitations and responsible use of data.

**Model:** Logistic Regression  
**Dashboard:** Streamlit  
**Project Domain:** Workforce Analytics / Human Resources Analytics
