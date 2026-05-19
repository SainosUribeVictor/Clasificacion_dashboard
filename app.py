import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Modelos ML",
    layout="wide"
)

st.title("Modelos de Clasificación")

# =========================
# CARGAR CSV
# =========================
metrics_df = pd.read_csv("model_metrics2.csv")
report_df = pd.read_csv("classification_reports.csv")
cm_df = pd.read_csv("confusion_matrices.csv")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Filtros")

modelo = st.sidebar.selectbox(
    "Selecciona un modelo",
    metrics_df["Modelo"].unique()
)

# =========================
# FILTRAR DATOS
# =========================
metrics_model = metrics_df[
    metrics_df["Modelo"] == modelo
]

report_model = report_df[
    report_df["Modelo"] == modelo
]

cm_model = cm_df[
    cm_df["Modelo"] == modelo
]

# =========================
# METRICAS GENERALES
# =========================
st.header("Métricas Generales")

col1, col2, col3, col4 = st.columns(4)

accuracy = metrics_model["Accuracy"].values[0]
precision = metrics_model["Precision"].values[0]
recall = metrics_model["Recall"].values[0]
f1 = metrics_model["F1"].values[0]

col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1", f"{f1:.3f}")

# =========================
# COMPARACION ENTRE MODELOS
# =========================
st.header("Comparación de Modelos")

fig = px.bar(
    metrics_df,
    x="Modelo",
    y=["Accuracy", "Precision", "Recall", "F1"],
    barmode="group",
    title="Comparación General"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# CLASSIFICATION REPORT
# =========================
st.header("Reporte")

st.dataframe(report_model)

# =========================
# GRAFICA POR CLASE
# =========================
st.subheader("Desempeño por Clase")

fig2 = px.bar(
    report_model,
    x="Clase",
    y=["Precision", "Recall", "F1-Score"],
    barmode="group",
    title=f"Métricas por Clase - {modelo}"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# MATRIZ DE CONFUSION
# =========================
st.header("Matriz de Confusión")

pivot_cm = cm_model.pivot_table(
    index="Clase_Real",
    columns="Clase_Predicha",
    values="Cantidad",
    fill_value=0
)

fig3, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    pivot_cm,
    annot=True,
    fmt=".0f",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de Confusión - {modelo}")

st.pyplot(fig3)