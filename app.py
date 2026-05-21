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

# =========================
# CARGAR CSV
# =========================
metrics_df = pd.read_csv("model_metrics2.csv")
report_df = pd.read_csv("classification_reports.csv")
cm_df = pd.read_csv("confusion_matrices.csv")
roc_df = pd.read_csv("roc_curves.csv")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navegación")

pagina = st.sidebar.radio(
    "Ir a:",
    ["Introducción", "Dashboard Modelos"]
)

# =========================================================
# PAGINA INTRODUCCION
# =========================================================
if pagina == "Introducción":

    st.title("Salud Mental y Machine Learning")

    st.markdown("""
    <style>
    .card {
        padding: 20px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
    }

    .blue {
        background: linear-gradient(135deg, #1E3A8A, #2563EB);
    }

    .green {
        background: linear-gradient(135deg, #065F46, #10B981);
    }

    .purple {
        background: linear-gradient(135deg, #5B21B6, #8B5CF6);
    }

    .orange {
        background: linear-gradient(135deg, #C2410C, #F97316);
    }

    .red {
        background: linear-gradient(135deg, #991B1B, #EF4444);
    }

    .title-card {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .text-card {
        font-size: 16px;
        line-height: 1.6;
    }

    .bubble {
        background-color: #F3F4F6;
        padding: 12px 18px;
        border-radius: 30px;
        display: inline-block;
        margin: 8px;
        font-weight: bold;
        color: #111827;
    }
    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # OBJETIVO
    # =====================================================
    st.markdown("""
    <div class="card blue">
        <div class="title-card">Objetivo del Proyecto</div>
        <div class="text-card">
        Desarrollar modelos de Machine Learning capaces de clasificar
        condiciones de salud mental utilizando variables laborales,
        psicológicas y sociales presentes en el dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # VARIABLES
    # =====================================================
    st.subheader("Variables Analizadas")

    st.markdown("""
    <div class="bubble">Estrés Laboral</div>
    <div class="bubble">Acceso a Tratamiento</div>
    <div class="bubble">Apoyo del Empleador</div>
    <div class="bubble">Ambiente Laboral</div>
    <div class="bubble">Historial Psicológico</div>
    <div class="bubble">Condiciones de Trabajo</div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================================
    # EDA
    # =====================================================
    st.markdown("""
    <div class="card green">
        <div class="title-card">Exploratory Data Analysis (EDA)</div>
        <div class="text-card">
        Antes del entrenamiento se realizó un análisis exploratorio completo
        para comprender la estructura del dataset y preparar los datos correctamente.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
         Valores nulos  
         Distribuciones  
         Correlaciones
        """)

    with col2:
        st.info("""
         Variables categóricas  
         Variables numéricas  
         Outliers
        """)

    with col3:
        st.info("""
         Balance de clases  
         Limpieza de datos  
         Codificación
        """)

    # =====================================================
    # SMOTE
    # =====================================================
    st.markdown("""
    <div class="card purple">
        <div class="title-card">Balanceo de Clases con SMOTE</div>
        <div class="text-card">
        El dataset presentaba un fuerte desbalance entre clases.
        Algunas categorías tenían miles de registros mientras otras
        contaban con muy pocos ejemplos.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("""
     SMOTE permitió generar datos sintéticos para clases minoritarias,
    equilibrando el dataset y mejorando el desempeño de los modelos.
    """)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Problema Detectado", "Desbalance")

    with col5:
        st.metric("Técnica Aplicada", "SMOTE")

    with col6:
        st.metric("Resultado", "Dataset Balanceado")

    # =====================================================
    # MODELOS
    # =====================================================
    st.markdown("""
    <div class="card orange">
        <div class="title-card">Modelos Evaluados</div>
        <div class="text-card">
        Se compararon distintos algoritmos de clasificación para identificar
        cuál ofrece mejores resultados en precisión y capacidad de generalización.
        </div>
    </div>
    """, unsafe_allow_html=True)

    modelos = [
        "SVM",
        "KNN",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "Gamma Pydra",
        "CatBoost"
    ]

    cols = st.columns(4)

    for i, modelo_card in enumerate(modelos):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="bubble">{modelo_card}</div>
            """, unsafe_allow_html=True)

    # =====================================================
    # CONCLUSION
    # =====================================================
    st.markdown("""
    <div class="card red">
        <div class="title-card">Conclusiones</div>
        <div class="text-card">
        El proyecto demostró la importancia del preprocesamiento,
        balanceo de datos y comparación de modelos para obtener
        predicciones más confiables y precisas.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col7, col8 = st.columns(2)

    with col7:
        st.success("""
        ### Resultados Principales

        - Dataset correctamente preparado
        - Mejor equilibrio entre clases
        - Mejora en Recall y F1-Score
        - Comparación efectiva entre modelos
        """)

    with col8:
        st.warning("""
        ### Mejor Modelo

        🏆 **CatBoost**

        Destacó por:
        - Mejor precisión
        - Menor sobreajuste
        - Buen manejo de variables categóricas
        - Mejor capacidad de clasificación
        """)

    st.balloons()

# =========================================================
# PAGINA DASHBOARD
# =========================================================
elif pagina == "Dashboard Modelos":

    st.title("Dashboard de Modelos de Clasificación")

    # =========================
    # FILTRO MODELOS
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

    roc_model = roc_df[
        roc_df["Modelo"] == modelo
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
    st.header("Reporte de Clasificación")

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

    # =========================
    # CURVAS ROC
    # =========================
    st.header(f"Curvas ROC - {modelo}")

    fig4 = px.line(
        roc_model,
        x="FPR",
        y="TPR",
        color="Clase",
        hover_data=["AUC", "Threshold"],
        title=f"Curvas ROC - {modelo}"
    )

    # Línea diagonal
    fig4.add_shape(
        type="line",
        line=dict(dash="dash"),
        x0=0,
        y0=0,
        x1=1,
        y1=1
    )

    fig4.update_layout(
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        height=650
    )

    st.plotly_chart(fig4, use_container_width=True)