# Drug Review Analytics Pipeline, Predictions, and Interactive Dashboard
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Dagster](https://img.shields.io/badge/Dagster-Orchestration-6C5CE7?style=flat-square&logo=dagster)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?style=flat-square&logo=postgresql)

This project implements a CRISP-DM methodology and an end-to-end data analytics workflow for analysing pharmaceutical drug reviews, combining:

- **Dagster** for reproducible data orchestration  
- **Jupyter Notebook** for data preprocessing and modelling  
- **Streamlit** for a fully interactive analytics dashboard  

The system separates **heavy data processing and modelling** from **interactive visualisation**.

---

## Project Overview

The project analyses drug review multi-modal data to understand:

- Review volume across drugs and medical conditions
- Sentiment and satisfaction distributions
- Key words driving positive and negative sentiment
- Machine learning model performance for:
  - Satisfaction classification
  - Effectiveness prediction
- Model interpretability using feature importance and agreement metrics
- Model Evaluation using Kappa

---

## Architecture

Jupyter Notebook → Dagster Pipeline → Cleaned Outputs
↓
Streamlit Dashboard

---

### Design Principles
- **Dagster** handles:
  - Data preprocessing
  - Feature engineering
  - Model training (offline)
  - Saving clean datasets and trained models
- **Jupyter Notebook** handles:
  - Data preprocessing
  - Feature engineering
  - Model training
  - Static Visual Generation for Report
- **Streamlit** handles:
  - Filtering and slicing data
  - Live, interactive visualisations
  - Model evaluation dashboards

---

## Technologies Used

- **Python**
- **Dagster** – data orchestration
- **Pandas / NumPy** – data processing
- **Scikit-learn** – machine learning
- **NLTK / Text Processing** – sentiment preparation
- **Matplotlib / Seaborn** – report figures
- **Plotly** – interactive charts
- **Streamlit** – dashboard
- **Conda** – environment management
