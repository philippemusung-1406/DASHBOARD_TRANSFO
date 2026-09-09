import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard Inspection Transformateurs MT",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Tableau de Bord - Inspection des Transformateurs MT")
st.markdown(
    "Analyse interactive de la maintenance et du suivi de l'état du parc de transformateurs."
)

# --- CHARGEMENT INTELLIGENT ET SÉCURISÉ DES DONNÉES ---
BASE_DIR = Path(__file__).resolve().parent


def find_dataset():
    """Recherche automatique du fichier dataset dans les dossiers courants."""
    possible_names = [
        "transfo dataset.xlsx",
        "transfo_dataset.xlsx",
        "transfo dataset.xls",
    ]
    search_paths = [
        BASE_DIR,
        BASE_DIR.parent,
        Path.cwd(),
        Path.home() / "Downloads",
        Path.home() / "Téléchargements",
    ]

    for path in search_paths:
        for name in possible_names:
            full_path = path / name
            if full_path.exists():
                return full_path
    return None


@st.cache_data
def load_and_clean_data(file_source):
    """Charge et nettoie les colonnes et espaces parasites."""
    df = pd.read_excel(file_source)

    # Nettoyage des noms de colonnes (suppression des espaces au début/fin)
    df.columns = df.columns.str.strip()

    # Nettoyage des espaces dans les colonnes textuelles
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Conversion de la colonne date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


# Barre latérale : Gestion du fichier
st.sidebar.header("📁 Source des Données")
uploaded_file = st.sidebar.file_uploader(
    "Charger un autre fichier Excel", type=["xlsx", "xls"]
)

dataset_path = find_dataset()

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    st.sidebar.success("Fichier chargé via l'interface !")
elif dataset_path is not None:
    df = load_and_clean_data(dataset_path)
    st.sidebar.info(f"Fichier détecté : `{dataset_path.name}`")
else:
    st.error(
        "⚠️ Fichier 'transfo dataset.xlsx' introuvable. Veuillez le téléverser via la barre latérale."
    )
    st.stop()

# --- FILTRES DE LA BARRE LATÉRALE ---
st.sidebar.header("🔍 Filtres d'Analyse")

# Filtre Transformateur
transfo_options = ["Tous"] + sorted(list(df["transfo_id"].unique()))
selected_transfo = st.sidebar.selectbox(
    "Sélectionner un transformateur", transfo_options
)

# Filtre Niveau d'huile
huile_options = ["Tous"] + sorted(list(df["niveau_huile(°c)"].unique()))
selected_huile = st.sidebar.selectbox("Niveau d'huile", huile_options)

# Filtrage dynamique
filtered_df = df.copy()
if selected_transfo != "Tous":
    filtered_df = filtered_df[filtered_df["transfo_id"] == selected_transfo]
if selected_huile != "Tous":
    filtered_df = filtered_df[filtered_df["niveau_huile(°c)"] == selected_huile]

# --- INDICATEURS CLÉS (KPIs) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Nombre d'inspections", len(filtered_df))

with kpi2:
    temp_avg = (
        filtered_df["temp_huile(°c)"].mean() if not filtered_df.empty else 0
    )
    st.metric("Temp. Huile Moyenne", f"{temp_avg:.1f} °C")

with kpi3:
    silica_avg = (
        filtered_df["silicagel(%)"].mean() if not filtered_df.empty else 0
    )
    st.metric("Silicagel Moyen", f"{silica_avg:.1f} %")

with kpi4:
    fuites = (
        len(filtered_df[filtered_df["buchings"] == "fuite"])
        if not filtered_df.empty
        else 0
    )
    st.metric("Alertes Fuite Buchings", fuites)

st.markdown("---")

# --- VISUALISATIONS GRAPHIQUES ---
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("📈 Évolution de la Température de l'Huile")
    fig_temp = px.line(
        filtered_df,
        x="date",
        y="temp_huile(°c)",
        color="transfo_id",
        markers=True,
        labels={
            "date": "Date d'inspection",
            "temp_huile(°c)": "Température (°C)",
        },
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with col_graph2:
    st.subheader("📊 Niveau de Silicagel (%) par Equipement")
    fig_silica = px.bar(
        filtered_df,
        x="transfo_id",
        y="silicagel(%)",
        color="niveau_huile(°c)",
        barmode="group",
        labels={"transfo_id": "Transformateur", "silicagel(%)": "Silicagel (%)"},
    )
    st.plotly_chart(fig_silica, use_container_width=True)

col_graph3, col_graph4 = st.columns(2)

with col_graph3:
    st.subheader("⚠️ État des Relais Buchholz")
    fig_buchholz = px.pie(
        filtered_df,
        names="relais_buchh",
        title="Répartition des états des relais Buchholz",
        hole=0.4,
    )
    st.plotly_chart(fig_buchholz, use_container_width=True)

with col_graph4:
    st.subheader("🔌 Aspect Général vs Buchings")
    fig_aspect = px.histogram(
        filtered_df,
        x="aspet _gen",
        color="buchings",
        barmode="group",
        labels={"aspet _gen": "Aspect Général", "count": "Nombre"},
    )
    st.plotly_chart(fig_aspect, use_container_width=True)

# --- TABLEAU DE DONNÉES ET TÉLÉCHARGEMENT ---
st.subheader("📋 Extraits des Données Filtrées")
st.dataframe(filtered_df, use_container_width=True)

# Bouton de téléchargement des données filtrées
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Télécharger les données filtrées (CSV)",
    data=csv_data,
    file_name="inspections_transformateurs_filtres.csv",
    mime="text/csv",
)
