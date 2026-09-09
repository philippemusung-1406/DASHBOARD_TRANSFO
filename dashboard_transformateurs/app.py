from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. CONFIGURATION DE LA PAGE & THÈME
# ==========================================
st.set_page_config(
    page_title="GMAO - Monitor Transfo MT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injection de CSS personnalisé pour un design industriel / haut de gamme
# Et suppression de l'outil d'édition Streamlit (menu et footer)
st.markdown(
    """
    <style>
    /* Masquer le menu d'édition Streamlit et le footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Arrière-plan global et police */
    .main {
        background-color: #0E1117;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Style des cartes KPI */
    .metric-card {
        background: linear-gradient(135deg, #1E2640 0%, #111827 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .metric-card.alert {
        border-left: 5px solid #EF4444;
    }
    .metric-card.warning {
        border-left: 5px solid #F59E0B;
    }
    .metric-card.success {
        border-left: 5px solid #10B981;
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-sub {
        color: #6B7280;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* Style des conteneurs de graphiques */
    .chart-container {
        background-color: #1A202C;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. CHARGEMENT & NETTOYAGE DES DONNÉES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent


def find_dataset():
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
def load_data(source):
    df = pd.read_excel(source)
    df.columns = df.columns.str.strip()

    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Ajout d'une colonne de score de criticité
    df["critique"] = False
    mask_alert = (
        (df["temp_huile(°c)"] >= 50)
        | (df["niveau_huile(°c)"] == "rouge")
        | (df["buchings"] == "fuite")
        | (df["relais_buchh"] == "fuite")
    )
    df.loc[mask_alert, "critique"] = True

    return df


# Barre latérale : Chargement
st.sidebar.image(
    "https://img.icons8.com/fluent/96/lightning-bolt.png", width=60
)
st.sidebar.title("GMAO Transfo MT")
st.sidebar.caption("Système de Supervision & Inspection")

dataset_file = find_dataset()
uploaded_file = st.sidebar.file_uploader(
    "🔄 Mettre à jour le dataset", type=["xlsx", "xls"]
)

if uploaded_file:
    df = load_data(uploaded_file)
elif dataset_file:
    df = load_data(dataset_file)
else:
    st.error(
        "⚠️ Base de données introuvable. Veuillez téléverser 'transfo dataset.xlsx'."
    )
    st.stop()

# ==========================================
# 3. FILTRES DYNAMIQUES (SIDEBAR)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtres d'Analyse")

# Filtre de période
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Période d'inspection",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Filtre par Transformateur
transfo_list = ["Tous les équipements"] + sorted(list(df["transfo_id"].unique()))
selected_transfo = st.sidebar.selectbox("Équipement", transfo_list)

# Application des filtres
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= date_range[0])
        & (filtered_df["date"].dt.date <= date_range[1])
    ]

if selected_transfo != "Tous les équipements":
    filtered_df = filtered_df[filtered_df["transfo_id"] == selected_transfo]

# ==========================================
# 4. EN-TÊTE & SYNTHÈSE GLOBALE (KPIS)
# ==========================================
st.title("⚡ Table de Bord d'Inspection des Transformateurs")
st.markdown(
    "Supervision temps réel de la santé des transformateurs Moyenne Tension."
)

total_inspections = len(filtered_df)
anomalies_count = len(filtered_df[filtered_df["critique"] == True])
taux_conformite = (
    ((total_inspections - anomalies_count) / total_inspections * 100)
    if total_inspections > 0
    else 100
)
temp_max = (
    filtered_df["temp_huile(°c)"].max() if not filtered_df.empty else 0
)

# Affichage des cartes KPI HTML
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Inspections</div>
            <div class="metric-value">{total_inspections}</div>
            <div class="metric-sub">Relevés enregistrés</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with k2:
    status_class = "success" if taux_conformite >= 85 else "warning"
    st.markdown(
        f"""
        <div class="metric-card {status_class}">
            <div class="metric-title">Taux de Conformité</div>
            <div class="metric-value">{taux_conformite:.1f}%</div>
            <div class="metric-sub">Équipements sans anomalie</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with k3:
    alert_class = "alert" if anomalies_count > 0 else "success"
    st.markdown(
        f"""
        <div class="metric-card {alert_class}">
            <div class="metric-title">Anomalies Détectées</div>
            <div class="metric-value">{anomalies_count}</div>
            <div class="metric-sub">Cas d'alerte ou de fuite</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with k4:
    temp_class = "alert" if temp_max >= 50 else "success"
    st.markdown(
        f"""
        <div class="metric-card {temp_class}">
            <div class="metric-title">Temp. Max Enregistrée</div>
            <div class="metric-value">{temp_max} °C</div>
            <div class="metric-sub">Seuil critique : 50 °C</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. ONGLETS DE NAVIGATION
# ==========================================
tab_overview, tab_alerts, tab_individual, tab_data = st.tabs(
    [
        "📊 Vue d'ensemble",
        "🚨 Centre d'Alertes & Maintenance",
        "🔍 Diagnostic d'un Equipement",
        "📋 Registre Brut",
    ]
)

# Palette de couleurs personnalisée pour Plotly
COLOR_PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

# --- TAB 1 : VUE D'ENSEMBLE ---
with tab_overview:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🌡️ Évolution Température de l'Huile (°C)")
        fig_temp = px.line(
            filtered_df,
            x="date",
            y="temp_huile(°c)",
            color="transfo_id",
            markers=True,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_temp.add_hline(
            y=50,
            line_dash="dot",
            line_color="red",
            annotation_text="Seuil Alerte (50°C)",
        )
        fig_temp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_b:
        st.subheader("💧 Distribution de l'Humidité (Silicagel %)")
        fig_silica = px.histogram(
            filtered_df,
            x="silicagel(%)",
            color="transfo_id",
            barmode="overlay",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_silica.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_silica, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("🟢 État du Niveau d'Huile")
        fig_huile = px.pie(
            filtered_df,
            names="niveau_huile(°c)",
            color="niveau_huile(°c)",
            color_discrete_map={
                "vert": "#10B981",
                "jaune": "#F59E0B",
                "rouge": "#EF4444",
            },
            hole=0.4,
            template="plotly_dark",
        )
        fig_huile.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_huile, use_container_width=True)

    with col_d:
        st.subheader("⚡ Relation Puissance (kVA) vs Température")
        fig_scatter = px.scatter(
            filtered_df,
            x="puissance(kva)",
            y="temp_huile(°c)",
            size="silicagel(%)",
            color="aspet _gen",
            hover_name="transfo_id",
            template="plotly_dark",
            color_discrete_map={"propre": "#3B82F6", "sale": "#EF4444"},
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 2 : CENTRE D'ALERTES & MAINTENANCE ---
with tab_alerts:
    st.subheader("🚨 Équipements Nécessitant une Intervention Immédiate")

    critical_df = filtered_df[filtered_df["critique"] == True]

    if critical_df.empty:
        st.success(
            "✅ Aucune anomalie critique détectée sur la période sélectionnée."
        )
    else:
        st.warning(
            f"Attention : {len(critical_df)} relevé(s) d'inspection indiquent un état critique."
        )

        st.dataframe(
            critical_df[
                [
                    "date",
                    "transfo_id",
                    "temp_huile(°c)",
                    "niveau_huile(°c)",
                    "buchings",
                    "relais_buchh",
                    "aspet _gen",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("### 📊 Répartition des Défauts")
        col_alt1, col_alt2 = st.columns(2)

        with col_alt1:
            fig_def_b = px.bar(
                critical_df,
                x="transfo_id",
                color="buchings",
                title="Défauts sur Buchings par Transformateur",
                template="plotly_dark",
                color_discrete_map={"propre": "#10B981", "fuite": "#EF4444"},
            )
            fig_def_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_def_b, use_container_width=True)

        with col_alt2:
            fig_def_r = px.bar(
                critical_df,
                x="transfo_id",
                color="relais_buchh",
                title="Défauts sur Relais Buchholz",
                template="plotly_dark",
                color_discrete_map={
                    "propre": "#10B981",
                    "sale": "#F59E0B",
                    "fuite": "#EF4444",
                },
            )
            fig_def_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_def_r, use_container_width=True)

# --- TAB 3 : DIAGNOSTIC INDIVIDUEL ---
with tab_individual:
    st.subheader("🔍 Fiche Technique et Historique")

    transfo_selected_single = st.selectbox(
        "Choisir un transformateur spécifique :",
        sorted(list(df["transfo_id"].unique())),
        key="single_transfo_select",
    )

    single_df = df[df["transfo_id"] == transfo_selected_single].sort_values(
        "date"
    )

    if not single_df.empty:
        latest = single_df.iloc[-1]

        # Fiche d'information
        f1, f2, f3, f4 = st.columns(4)
        f1.info(f"**Puissance :** {latest['puissance(kva)']} kVA")
        f2.info(
            f"**Tension Primaire :** {latest['tension_pri(v)']} V"
        )
        f3.info(
            f"**Tension Secondaire :** {latest['tension_sec(v)']} V"
        )
        f4.info(f"**Dernier contrôle :** {latest['date'].strftime('%Y-%m-%d')}")

        st.markdown("---")

        # Graphique d'historique de l'équipement
        fig_single = go.Figure()
        fig_single.add_trace(
            go.Scatter(
                x=single_df["date"],
                y=single_df["temp_huile(°c)"],
                mode="lines+markers",
                name="Température (°C)",
                line=dict(color="#EF4444", width=3),
            )
        )
        fig_single.add_trace(
            go.Bar(
                x=single_df["date"],
                y=single_df["silicagel(%)"],
                name="Silicagel (%)",
                marker_color="#3B82F6",
                opacity=0.4,
            )
        )
        fig_single.update_layout(
            title=f"Historique d'inspection - {transfo_selected_single}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_single, use_container_width=True)

# --- TAB 4 : REGISTRE BRUT & EXPORT ---
with tab_data:
    st.subheader("📋 Registre d'Inspection des Transformateurs")

    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Exporter les données sélectionnées (CSV)",
        data=csv,
        file_name="rapport_inspection_transformateurs.csv",
        mime="text/csv",
    )
