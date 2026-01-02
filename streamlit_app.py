import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
import plotly.express as px
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

sat_model = joblib.load("outputs/models/satisfaction_model.pkl")
X_test = pd.read_csv("outputs/data/X_test.csv")
y_test = pd.read_csv("outputs/data/y_test.csv")["satisfaction_class"]
eff_model = joblib.load("outputs/models/effectiveness_model.pkl")
EX_test = pd.read_csv("outputs/data/EX_test.csv")
Ey_test = pd.read_csv("outputs/data/Ey_test.csv")["effectiveness_class"]


st.set_page_config(
    page_title="Drug Review Analytics Dashboard",
    layout="wide"
)
DATA_PATH = Path("data/newdrugupdated_data.csv")
PLOT_DIR = Path("outputs")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("💊 Drug Review Analytics Dashboard")

if not DATA_PATH.exists():
    st.error(f"Data not found at {DATA_PATH.resolve()}")
    st.stop()

df = load_data()

st.sidebar.header("🔍 Filters")

drug_filter = st.sidebar.multiselect("Select Drug(s)", sorted(df["Drug"].dropna().unique()))

condition_filter = st.sidebar.multiselect(
    "Select Condition(s)",
    sorted(df["Condition"].dropna().unique())
)

sentiment_col = "sentiment" if "sentiment" in df.columns else "sentiment_label"
sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment(s)",
    sorted(df[sentiment_col].dropna().unique())
)


filtered_df = df.copy()

if drug_filter:
    filtered_df = filtered_df[filtered_df["Drug"].isin(drug_filter)]

if condition_filter:
    filtered_df = filtered_df[filtered_df["Condition"].isin(condition_filter)]

if sentiment_filter:
    filtered_df = filtered_df[filtered_df[sentiment_col].isin(sentiment_filter)]


#KPIs
st.subheader("Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Reviews", len(filtered_df))
c2.metric("Unique Drugs", filtered_df["Drug"].nunique())
c3.metric("Unique Conditions", filtered_df["Condition"].nunique())

st.markdown("---")

st.subheader("Distribution of Review Length (Words)")

# review length 

fig = px.histogram(filtered_df, x="review_length", nbins=50, marginal="rug", labels={"review_length": "Word Count"},)
fig.update_layout(xaxis_title="Word Count",yaxis_title="Number of Reviews",bargap=0.1)

st.markdown("---")

#Sentiment distribution
st.subheader("Sentiment Distribution")

sentiment_counts = filtered_df[sentiment_col].value_counts()
st.bar_chart(sentiment_counts)

st.markdown("---")

#Top Drugs by Review
st.subheader("Top 15 Most Reviewed Drugs")

top_drugs_df = (filtered_df["Drug"].value_counts().head(15).reset_index())
top_drugs_df.columns = ["Drug", "Review_Count"]

# percentage contribution
total_dreviews = top_drugs_df["Review_Count"].sum()
top_drugs_df["Percentage"] = (top_drugs_df["Review_Count"] / total_dreviews * 100).round(1)

#treemap
fig = px.treemap(top_drugs_df, path=["Drug"], values="Review_Count", custom_data=["Percentage"],)
fig.update_traces(hovertemplate= "<b>%{label}</b><br>" + "Reviews: %{value}<br>" + "Share: %{customdata[0]}%")
fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

#Top Conditions by Review
st.subheader("Top 15 Most Reviewed Conditions")

top_conditions_df = (filtered_df["Condition"].value_counts().head(15).reset_index())
top_conditions_df.columns = ["Condition", "Review_Count"]

# percentage contribution
total_creviews = top_conditions_df["Review_Count"].sum()
top_conditions_df["Percentage"] = (top_conditions_df["Review_Count"] / total_creviews * 100).round(1)

#treemap
fig = px.treemap(top_conditions_df, path=["Condition"], values="Review_Count", custom_data=["Percentage"],)
fig.update_traces(hovertemplate= "<b>%{label}</b><br>" + "Reviews: %{value}<br>" + "Share: %{customdata[0]}%")
fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Overall Sentiment Category Distribution")

sentiment_counts = (filtered_df["sentiment_label"].value_counts().reset_index())

sentiment_counts.columns = ["Sentiment", "Count"]
sentiment_counts["Percentage"] = (sentiment_counts["Count"] / sentiment_counts["Count"].sum() * 100).round(1)

st.bar_chart(sentiment_counts.set_index("Sentiment")["Count"])

st.markdown("---")

st.subheader("Overall Satisfaction Category Distribution")

sat_counts = (filtered_df["satisfaction_class"].value_counts().reset_index())

sat_counts.columns = ["Satisfaction", "Count"]
sat_counts["Percentage"] = (sat_counts["Count"] / sat_counts["Count"].sum() * 100).round(1)

st.bar_chart(sat_counts.set_index("Satisfaction")["Count"])

st.markdown("---")

st.subheader("Overall Effectiveness Category Distribution")

eff_counts = (filtered_df["effectiveness_class"].value_counts().reset_index())

eff_counts.columns = ["Effectiveness", "Count"]
eff_counts["Percentage"] = (eff_counts["Count"] / eff_counts["Count"].sum() * 100).round(1)

st.bar_chart(eff_counts.set_index("Effectiveness")["Count"])

st.markdown("---")

y_pred = sat_model.predict(X_test)
st.subheader("Satisfaction Classification Performance")

sat_accuracy = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{sat_accuracy:.3f}")

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)

st.subheader("Classification Report")
st.dataframe(report_df, use_container_width=True)

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Satisfaction Classification")
st.pyplot(fig)

st.subheader("Satisfsaction Model Agreement (Cohen’s Kappa)")

kappa = cohen_kappa_score(y_test, y_pred)

st.metric(
    label="Cohen’s Kappa",
    value=f"{kappa:.3f}",
    help="Measures agreement beyond chance (−1 to 1)"
)

st.subheader("Top Predictors of Satisfaction")

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": sat_model.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")

fig.update_layout(
    yaxis=dict(autorange="reversed"),
    xaxis_title="Feature Importance",
    yaxis_title="Feature"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

Ey_pred = eff_model.predict(EX_test)
st.subheader("Effectiveness Classification Performance")

eff_accuracy = accuracy_score(Ey_test, Ey_pred)
st.metric("Accuracy", f"{eff_accuracy:.3f}")

report_dict = classification_report(Ey_test, Ey_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)

st.subheader("Classification Report")
st.dataframe(report_df, use_container_width=True)

st.subheader("Confusion Matrix")

cm = confusion_matrix(Ey_test, Ey_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Effectiveness Classification")
st.pyplot(fig)

st.subheader("Effectiveness Model Agreement (Cohen’s Kappa)")

kappa = cohen_kappa_score(y_test, y_pred)

st.metric(
    label="Cohen’s Kappa",
    value=f"{kappa:.3f}",
    help="Measures agreement beyond chance (−1 to 1)"
)

st.subheader("Top Predictors of Effectiveness")

importance_df = pd.DataFrame({
    "Feature": EX_test.columns,
    "Importance": eff_model.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")

fig.update_layout(
    yaxis=dict(autorange="reversed"),
    xaxis_title="Feature Importance",
    yaxis_title="Feature"
)

st.plotly_chart(fig, use_container_width=True)
