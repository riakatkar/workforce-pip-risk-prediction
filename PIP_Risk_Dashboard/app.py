
import os
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Workforce Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "PIP_Risk_Dataset - LMS.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pip_logistic_regression_model.pkl"
)


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

try:
    model = load_model()
    model_loaded = True
except Exception:
    model = None
    model_loaded = False


# --------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "department",
    "job_role",
    "employment_type",
    "pip_required"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

df["pip_required"] = pd.to_numeric(
    df["pip_required"],
    errors="coerce"
)

df = df.dropna(subset=["pip_required"]).copy()


# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("Workforce Filters")
st.sidebar.caption("Customize your analysis")

department_options = ["All"] + sorted(
    df["department"].dropna().astype(str).unique().tolist()
)

role_options = ["All"] + sorted(
    df["job_role"].dropna().astype(str).unique().tolist()
)

employment_options = ["All"] + sorted(
    df["employment_type"].dropna().astype(str).unique().tolist()
)

selected_department = st.sidebar.selectbox(
    "Department",
    department_options
)

selected_role = st.sidebar.selectbox(
    "Job Role",
    role_options
)

selected_employment = st.sidebar.selectbox(
    "Employment Type",
    employment_options
)


# --------------------------------------------------
# 5. APPLY FILTERS
# --------------------------------------------------

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["department"].astype(str)
        == selected_department
    ]

if selected_role != "All":
    filtered_df = filtered_df[
        filtered_df["job_role"].astype(str)
        == selected_role
    ]

if selected_employment != "All":
    filtered_df = filtered_df[
        filtered_df["employment_type"].astype(str)
        == selected_employment
    ]

if filtered_df.empty:
    st.warning("No employees found for the selected filters.")
    st.stop()


# --------------------------------------------------
# 6. DASHBOARD HEADER
# --------------------------------------------------

st.title("Workforce Analytics Dashboard")

st.caption(
    "Employee Overview | Workplace PIP Risk Prediction"
)

st.divider()


# --------------------------------------------------
# 7. KPI CARDS
# --------------------------------------------------

total_employees = len(filtered_df)

pip_cases = int(
    (filtered_df["pip_required"] == 1).sum()
)

pip_rate = (
    pip_cases / total_employees * 100
    if total_employees > 0 else 0
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Employees",
    f"{total_employees:,}"
)

col2.metric(
    "Employees Requiring PIP",
    f"{pip_cases:,}"
)

col3.metric(
    "PIP Rate",
    f"{pip_rate:.1f}%"
)

st.divider()


# --------------------------------------------------
# 8. DEPARTMENT ANALYSIS
# --------------------------------------------------

st.subheader("Department Analysis")

department_summary = (
    filtered_df.groupby("department")
    .agg(
        employees=("pip_required", "count"),
        pip_cases=("pip_required", "sum")
    )
    .reset_index()
)

department_summary["pip_rate"] = (
    department_summary["pip_cases"]
    / department_summary["employees"]
    * 100
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Employees by Department**")

    employee_chart = (
        department_summary
        .set_index("department")[["employees"]]
        .sort_values("employees")
    )

    st.bar_chart(
        employee_chart,
        horizontal=True
    )

with col2:
    st.markdown("**PIP Rate by Department (%)**")

    pip_chart = (
        department_summary
        .set_index("department")[["pip_rate"]]
        .sort_values("pip_rate")
    )

    st.bar_chart(
        pip_chart,
        horizontal=True
    )

st.divider()


# --------------------------------------------------
# 9. EMPLOYMENT TYPE DISTRIBUTION
# --------------------------------------------------

st.subheader("Employment Type Distribution")

employment_summary = (
    filtered_df["employment_type"]
    .value_counts()
    .sort_values()
    .to_frame("employees")
)

st.bar_chart(
    employment_summary,
    horizontal=True
)

st.divider()


# --------------------------------------------------
# 10. TOP 10 JOB ROLES
# --------------------------------------------------

st.subheader("Top 10 Job Roles by Employee Count")

role_summary = (
    filtered_df["job_role"]
    .value_counts()
    .head(10)
    .sort_values()
    .to_frame("employees")
)

st.bar_chart(
    role_summary,
    horizontal=True
)

st.divider()


# --------------------------------------------------
# 11. DEPARTMENT PERFORMANCE SUMMARY
# --------------------------------------------------

st.subheader("Department Performance Summary")

display_summary = department_summary.copy()

display_summary["pip_rate"] = (
    display_summary["pip_rate"].round(2)
)

display_summary = display_summary.rename(
    columns={
        "department": "Department",
        "employees": "Employees",
        "pip_cases": "PIP Cases",
        "pip_rate": "PIP Rate (%)"
    }
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()




# --------------------------------------------------
# 12. PERFORMANCE ANALYSIS
# --------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

st.subheader("Performance Analysis")

st.caption(
    "Explore how employee performance indicators "
    "are associated with PIP status."
)

# Performance features and their display names
performance_features = {
    "attendance_percentage": {
        "label": "Attendance Percentage",
        "unit": "%",
        "multiplier": 1
    },
    "kpi_score": {
        "label": "KPI Score",
        "unit": "points",
        "multiplier": 1
    },
    "quality_score": {
        "label": "Quality Score",
        "unit": "points",
        "multiplier": 1
    },
    "goal_completion_rate": {
        "label": "Goal Completion Rate",
        "unit": "%",
        "multiplier": 100
    },
    "training_hours_last_3m": {
        "label": "Training Hours (Last 3 Months)",
        "unit": "hours",
        "multiplier": 1
    }
}

available_features = [
    feature for feature in performance_features
    if feature in filtered_df.columns
]

if available_features:

    # Calculate averages by PIP status
    performance_summary = (
        filtered_df.groupby("pip_required")[available_features]
        .mean()
        .reindex([0, 1])
    )

    # Create a display-friendly summary
    display_performance = performance_summary.copy()

    for feature in available_features:
        multiplier = performance_features[feature]["multiplier"]
        display_performance[feature] *= multiplier

    display_performance.index = [
        "No PIP",
        "PIP Required"
    ]

    display_performance.columns = [
        performance_features[col]["label"]
        for col in available_features
    ]

    st.markdown("### Average Performance Indicators")

    st.dataframe(
        display_performance.round(2),
        use_container_width=True
    )

    st.divider()

    # Individual feature charts
    st.markdown("### Performance Indicator Comparison")

    for feature in available_features:

        details = performance_features[feature]

        chart_data = (
            performance_summary[feature]
            .reindex([0, 1])
            .dropna()
            * details["multiplier"]
        )

        if len(chart_data) != 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 3.2))

        # Dark background
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        bars = ax.barh(
            ["No PIP", "PIP Required"],
            chart_data.values,
            height=0.48,
            color=["#56B4E9", "#F39C6B"]
        )

        # Chart formatting
        ax.set_title(
            details["label"],
            color="white",
            fontsize=13,
            fontweight="bold",
            pad=15
        )

        ax.set_xlabel(
            f"Average ({details['unit']})",
            color="white"
        )

        ax.tick_params(
            axis="both",
            colors="white"
        )

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.2,
            color="white"
        )

        ax.set_axisbelow(True)

        # Remove unnecessary borders
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add values to bars
        max_value = max(chart_data.max(), 1)

        ax.set_xlim(0, max_value * 1.20)

        for bar, value in zip(bars, chart_data.values):

            ax.text(
                value + max_value * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                color="white",
                fontsize=10
            )

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    # --------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------

    st.divider()

    st.markdown("### Key Business Insights")

    insight_messages = []

    for feature in available_features:

        values = (
            performance_summary[feature]
            .reindex([0, 1])
        )

        no_pip = values.loc[0]
        pip_required = values.loc[1]

        if pd.isna(no_pip) or pd.isna(pip_required):
            continue

        difference = no_pip - pip_required
        details = performance_features[feature]

        if feature == "goal_completion_rate":
            difference *= 100

        if difference > 0:
            direction = "higher"
        elif difference < 0:
            direction = "lower"
        else:
            direction = "similar"

        insight_messages.append(
            f"**{details['label']}:** "
            f"No PIP employees have {direction} average values "
            f"({no_pip * details['multiplier']:.2f} vs. "
            f"{pip_required * details['multiplier']:.2f})."
        )

    for message in insight_messages:
        st.markdown(f"- {message}")

    st.info(
        "These findings show associations within the dataset. "
        "They do not establish causation or justify an automatic "
        "PIP decision. Individual cases require fair human review."
    )

else:
    st.info(
        "Performance indicator columns were not found "
        "in the dataset."
    )

st.divider()





# --------------------------------------------------
# 13. MODEL PERFORMANCE
# --------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

st.title("Model Performance")

st.caption(
    "Evaluation of the tuned Logistic Regression model "
    "on the held-out test dataset."
)

# Actual test results from the ML notebook
accuracy = 80.77
precision = 51.41
recall = 79.68
f1_score = 62.50
roc_auc = 88.33
pr_auc = 70.38

# --------------------------------------------------
# MODEL METRIC CARDS
# --------------------------------------------------

st.subheader("Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.2f}%")
col2.metric("Precision", f"{precision:.2f}%")
col3.metric("Recall", f"{recall:.2f}%")

col4, col5, col6 = st.columns(3)

col4.metric("F1 Score", f"{f1_score:.2f}%")
col5.metric("ROC-AUC", f"{roc_auc:.2f}%")
col6.metric("PR-AUC", f"{pr_auc:.2f}%")

st.divider()

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("Confusion Matrix")

# Actual values from the held-out test evaluation
confusion_matrix = np.array([
    [808, 189],
    [51, 200]
])

fig, ax = plt.subplots(figsize=(7, 4))

fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")

image = ax.imshow(
    confusion_matrix,
    cmap="Blues"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["No PIP", "PIP Required"],
    color="white"
)

ax.set_yticklabels(
    ["No PIP", "PIP Required"],
    color="white"
)

ax.set_xlabel("Predicted Label", color="white")
ax.set_ylabel("Actual Label", color="white")

ax.set_title(
    "Confusion Matrix",
    color="white",
    fontsize=14,
    fontweight="bold"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(confusion_matrix[i, j]),
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold"
        )

ax.tick_params(colors="white")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# --------------------------------------------------
# CONFUSION MATRIX INTERPRETATION
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix.ravel()

col1, col2 = st.columns(2)

col1.metric("True Negatives", f"{tn:,}")
col2.metric("True Positives", f"{tp:,}")

col3, col4 = st.columns(2)

col3.metric("False Positives", f"{fp:,}")
col4.metric("False Negatives", f"{fn:,}")

st.divider()

# --------------------------------------------------
# MODEL INTERPRETATION
# --------------------------------------------------

st.subheader("Model Interpretation")

st.markdown(
    f"""
    - **Accuracy:** The model correctly classified
      {accuracy:.2f}% of test observations.
    - **Precision:** {precision:.2f}% of predicted
      PIP cases were actual positive cases.
    - **Recall:** The model identified {recall:.2f}%
      of actual positive cases.
    - **F1 Score:** The balance between precision
      and recall was {f1_score:.2f}%.
    - **ROC-AUC:** The model achieved {roc_auc:.2f}%,
      indicating good ranking ability across thresholds.
    - **PR-AUC:** The model achieved {pr_auc:.2f}%.
    """
)

st.warning(
    "The model identifies potential patterns, not employees "
    "who should automatically receive a PIP. Human review, "
    "context and fairness checks are essential."
)

st.divider()
