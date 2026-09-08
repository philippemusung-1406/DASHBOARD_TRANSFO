import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Inspection Transformateurs MT",
    page_icon="⚡",
    layout="wide"
)

# Titre principal
st.title("⚡ Table de Bord d'Inspection des Transformateurs")
st.markdown("Visualisation et suivi de l'état des transformateurs moyenne tension.")

# Chargement des données avec mise en cache
@st.cache_data
def load_data():
    # Nettoyage des espaces éventuels dans les noms de colonnes
    df = pd.read_excel('transfo_dataset.xlsx')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- BARRE LATÉRALE : FILTRES ---
st.sidebar.header("🔍 Filtres de recherche")

# Filtre par transformateur
transfo_list = ['Tous'] + sorted(list(df['transfo_id'].unique()))
selected_transfo = st.sidebar.selectbox("Sélectionner un transformateur", transfo_list)

# Filtre par état du niveau d'huile
niveau_huile_list = ['Tous'] + list(df['niveau_huile(°c)'].unique())
selected_huile = st.sidebar.selectbox("Niveau d'huile", niveau_huile_list)

# Filtrage du dataframe
filtered_df = df.copy()

if selected_transfo != 'Tous':
    filtered_df = filtered_df[filtered_df['transfo_id'] == selected_transfo]

if selected_huile != 'Tous':
    filtered_df = filtered_df[filtered_df['niveau_huile(°c)'] == selected_huile]

# --- INDICATEURS CLÉS (KPIs) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Inspections", len(filtered_df))
with col2:
    st.metric("Temp. Huile Moyenne", f"{filtered_df['temp_huile(°c)'].mean():.1f} °C")
with col3:
    st.metric("Silicagel Moyen", f"{filtered_df['silicagel(%)'].mean():.1f} %")
with col4:
    fuites = len(filtered_df[filtered_df['buchings'].str.contains('fuite', case=False, na=False)])
    st.metric("Cas de Fuites (Buchings)", fuites)

st.markdown("---")

# --- GRAPHIQUES INTERACTIFS ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🌡️ Évolution Température Huile")
    fig_temp = px.line(
        filtered_df, 
        x='date', 
        y='temp_huile(°c)', 
        color='transfo_id',
        title="Température de l'huile dans le temps"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with col_right:
    st.subheader("💧 Niveau Silicagel (%)")
    fig_silica = px.box(
        filtered_df, 
        x='transfo_id', 
        y='silicagel(%)',
        title="Distribution du silicagel par transformateur"
    )
    st.plotly_chart(fig_silica, use_container_width=True)

# Graphique de répartition des anomalies (Buchings / Relais Buchholz)
st.subheader("⚠️ Répartition des États des Buchings")
fig_buching = px.histogram(
    filtered_df, 
    x='buchings', 
    color='aspet _gen', 
    barmode='group',
    title="État des Buchings vs Aspect Général"
)
st.plotly_chart(fig_buching, use_container_width=True)

# --- TABLEAU DE DONNÉES FILTRÉES ---
st.subheader("📋 Données détaillées d'inspection")
st.dataframe(filtered_df, use_container_width=True)