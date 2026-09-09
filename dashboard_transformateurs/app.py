from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION DE LA PAGE & THÈME CSS
# ==========================================
st.set_page_config(
    page_title="GMAO - Analytics & Fiabilité Transfo MT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Masquer le menu d'édition Streamlit et le footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Arrière-plan global */
    .main {
        background-color: #0E1117;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Style des cartes KPI */
    .metric-card {
        background: linear-gradient(135deg, #1E2640 0%, #111827 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    .metric-card.alert { border-left: 5px solid #EF4444; }
    .metric-card.warning { border-left: 5px solid #F59E0B; }
    .metric-card.success { border-left: 5px solid #10B981; }
    
    .metric-title {
        color: #9CA3AF;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #6B7280;
        font-size: 0.75rem;
        margin-top: 3px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. CHARGEMENT & PRÉPARATION DES DONNÉES
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

    # Tri pour calculs de séries temporelles
    df = df.sort_values(["transfo_id", "date"]).reset_index(drop=True)

    # Calcul de la variation du silicagel
    df["silica_diff"] = df.groupby("transfo_id")["silicagel(%)"].diff().fillna(0)

    # Indicateur global de criticité
    df["critique"] = False
    mask_alert = (
        (df["temp_huile(°c)"] >= 50)
        | (df["niveau_huile(°c)"] == "rouge")
        | (df["buchings"] == "fuite")
        | (df["relais_buchh"] == "fuite")
    )
    df.loc[mask_alert, "critique"] = True

    return df


# Entête Sidebar
st.sidebar.image(
    "https://img.icons8.com/fluent/96/lightning-bolt.png", width=50
)
st.sidebar.title("GMAO Transfo MT")
st.sidebar.caption("Plateforme d'Analyse & Prédiction")

dataset_file = find_dataset()
uploaded_file = st.sidebar.file_uploader(
    "🔄 Importer Dataset Excel", type=["xlsx", "xls"]
)

if uploaded_file:
    df = load_data(uploaded_file)
elif dataset_file:
    df = load_data(dataset_file)
else:
    st.error(
        "⚠️ Base de données introuvable. Veuillez téléverser le fichier Excel."
    )
    st.stop()

# ==========================================
# 3. FILTRES DYNAMIQUES
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtres d'Analyse")

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Période",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

transfo_list = ["Tous les équipements"] + sorted(list(df["transfo_id"].unique()))
selected_transfo = st.sidebar.selectbox("Équipement", transfo_list)

filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= date_range[0])
        & (filtered_df["date"].dt.date <= date_range[1])
    ]

if selected_transfo != "Tous les équipements":
    filtered_df = filtered_df[filtered_df["transfo_id"] == selected_transfo]

# ==========================================
# 4. EN-TÊTE & KPIs GLOBAUX
# ==========================================
st.title("⚡ Dashboard Avancé & Analyse de Fiabilité Transfo MT")
st.markdown(
    "Surveillance, Analyse Contingente, Modélisation Prédictive des Fuites et Calculs de Fiabilité (MTTF/MTTR)."
)

total_inspections = len(filtered_df)
anomalies_count = len(filtered_df[filtered_df["critique"] == True])
fuites_buchings = len(filtered_df[filtered_df["buchings"] == "fuite"])
silica_var_moy = (
    filtered_df["silica_diff"].abs().mean() if not filtered_df.empty else 0
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">Inspections Filtrées</div>
            <div class="metric-value">{total_inspections}</div>
            <div class="metric-sub">Relevés analysés</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    status_c = "alert" if anomalies_count > 0 else "success"
    st.markdown(
        f"""<div class="metric-card {status_c}">
            <div class="metric-title">Anomalies Globales</div>
            <div class="metric-value">{anomalies_count}</div>
            <div class="metric-sub">Équipements en alerte</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    b_class = "alert" if fuites_buchings > 0 else "success"
    st.markdown(
        f"""<div class="metric-card {b_class}">
            <div class="metric-title">Fuites Buchings</div>
            <div class="metric-value">{fuites_buchings}</div>
            <div class="metric-sub">Traversées défectueuses</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="metric-card warning">
            <div class="metric-title">Var. Moyenne Silicagel</div>
            <div class="metric-value">Δ {silica_var_moy:.2f}%</div>
            <div class="metric-sub">Flottement inter-inspection</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. ONGLETS D'ANALYSE
# ==========================================
tab_overview, tab_contingency, tab_evolution, tab_reliability, tab_data = (
    st.tabs(
        [
            "📊 Vue d'ensemble",
            "🧮 Contingence & Croisements",
            "📈 Évolution Temporelle Transfo",
            "⏳ Fiabilité, MTTF/MTTR & IA",
            "📋 Registre des Données",
        ]
    )
)

# --- TAB 1 : VUE D'ENSEMBLE ---
with tab_overview:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🌡️ Température d'Huile par Transformateur (°C)")
        fig_temp = px.box(
            filtered_df,
            x="transfo_id",
            y="temp_huile(°c)",
            color="aspet _gen",
            template="plotly_dark",
            color_discrete_map={"propre": "#3B82F6", "sale": "#EF4444"},
        )
        fig_temp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_b:
        st.subheader("💧 Répartition de l'État du Niveau d'Huile")
        fig_huile = px.histogram(
            filtered_df,
            x="niveau_huile(°c)",
            color="buchings",
            barmode="group",
            template="plotly_dark",
            color_discrete_map={"propre": "#10B981", "fuite": "#EF4444"},
        )
        fig_huile.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_huile, use_container_width=True)

# --- TAB 2 : TABLEAUX DE CONTINGENCE STYLISÉS ---
with tab_contingency:
    st.subheader(
        "🧮 Tableau de Contingence & Profils Lignes (Aspect Général vs Buchings)"
    )
    st.markdown(
        "Analyse de dépendance entre la propreté externe (Aspect Général) et la présence de fuites sur les traversées (Buchings)."
    )

    # Tableau des fréquences brutes
    ct_raw = pd.crosstab(
        filtered_df["aspet _gen"],
        filtered_df["buchings"],
        margins=True,
        margins_name="Total",
    )

    # Tableau des profils lignes (%)
    ct_prop = (
        pd.crosstab(
            filtered_df["aspet _gen"],
            filtered_df["buchings"],
            normalize="index",
        )
        * 100
    )

    col_ct1, col_ct2 = st.columns(2)

    with col_ct1:
        st.markdown("#### 🔢 Fréquences Absolues (Nombre d'Inspections)")
        st.dataframe(
            ct_raw.style.background_gradient(cmap="Blues").format("{:d}"),
            use_container_width=True,
        )

    with col_ct2:
        st.markdown("#### 📊 Profil Ligne (% de fuites par état d'aspect)")
        st.dataframe(
            ct_prop.style.background_gradient(cmap="OrRd").format("{:.2f} %"),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("📉 Représentation Graphique de la Contingence")
    fig_ct = px.bar(
        ct_prop.reset_index(),
        x="aspet _gen",
        y=["fuite", "propre"],
        title="Proportion des états de Buchings selon l'Aspect Général",
        labels={
            "aspet _gen": "Aspect Général",
            "value": "Proportion (%)",
            "variable": "État Buchings",
        },
        template="plotly_dark",
        color_discrete_map={"propre": "#10B981", "fuite": "#EF4444"},
    )
    fig_ct.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_ct, use_container_width=True)

# --- TAB 3 : ÉVOLUTION SPÉCIFIQUE DU SILICAGEL ET NIVEAU D'HUILE ---
with tab_evolution:
    st.subheader("📈 Suivi Chronologique par Transformateur")

    transfo_target = st.selectbox(
        "Sélectionner un transformateur à analyser :",
        sorted(list(df["transfo_id"].unique())),
        key="ev_transfo",
    )

    df_single = df[df["transfo_id"] == transfo_target].sort_values("date")

    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        st.markdown("#### 🧪 Évolution du Silicagel (%) & Deltas")
        fig_sil_ev = go.Figure()
        fig_sil_ev.add_trace(
            go.Scatter(
                x=df_single["date"],
                y=df_single["silicagel(%)"],
                mode="lines+markers",
                name="Silicagel (%)",
                line=dict(color="#3B82F6", width=3),
            )
        )
        fig_sil_ev.add_trace(
            go.Bar(
                x=df_single["date"],
                y=df_single["silica_diff"],
                name="Variation Δ (%)",
                marker_color="#F59E0B",
                opacity=0.6,
            )
        )
        fig_sil_ev.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sil_ev, use_container_width=True)

    with col_ev2:
        st.markdown("#### 🛢️ Historique du Niveau d'Huile")
        color_map_huile = {"vert": "#10B981", "jaune": "#F59E0B", "rouge": "#EF4444"}
        fig_oil_ev = px.scatter(
            df_single,
            x="date",
            y="niveau_huile(°c)",
            color="niveau_huile(°c)",
            size="temp_huile(°c)",
            color_discrete_map=color_map_huile,
            template="plotly_dark",
        )
        fig_oil_ev.update_traces(marker=dict(size=14))
        fig_oil_ev.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_oil_ev, use_container_width=True)

    # Analyse des variations du Silicagel
    st.markdown("#### 🔬 Analyse Statistiques des Variations de Silicagel")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric(
        "Moyenne du Silicagel", f"{df_single['silicagel(%)'].mean():.1f} %"
    )
    col_stat2.metric(
        "Écart-type Silicagel", f"{df_single['silicagel(%)'].std():.2f} %"
    )
    col_stat3.metric(
        "Variation Max (Δ)",
        f"{df_single['silica_diff'].abs().max():.1f} %",
    )

# --- TAB 4 : FIABILITÉ, MTTF/MTTR & IA PRÉDICTIVE ---
with tab_reliability:
    st.subheader("🔮 Prédiction Probabiliste de Fuite (Modèle IA)")
    st.markdown(
        "Ce modèle de Machine Learning évalue la probabilité qu'une traversée (Buchings) présente une fuite en fonction des paramètres de fonctionnement."
    )

    # Entraînement du modèle de régression logistique
    X = df[
        [
            "puissance(kva)",
            "tension_pri(v)",
            "temp_huile(°c)",
            "silicagel(%)",
            "aspet _gen",
            "niveau_huile(°c)",
        ]
    ]
    y = (df["buchings"] == "fuite").astype(int)

    X_encoded = pd.get_dummies(X, drop_first=True)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_encoded, y)

    # Simulateur interactif
    st.markdown("##### 🎛️ Simuler les Conditions d'un Transformateur")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sim_puiss = sc1.selectbox(
        "Puissance (kVA)", sorted(df["puissance(kva)"].unique())
    )
    sim_temp = sc2.slider(
        "Température Huile (°C)",
        int(df["temp_huile(°c)"].min()),
        int(df["temp_huile(°c)"].max()),
        40,
    )
    sim_silica = sc3.slider(
        "Silicagel (%)",
        int(df["silicagel(%)"].min()),
        int(df["silicagel(%)"].max()),
        80,
    )
    sim_aspect = sc4.selectbox(
        "Aspect Général", sorted(df["aspet _gen"].unique())
    )

    # Préparation du vecteur de test
    input_dict = {
        "puissance(kva)": [sim_puiss],
        "tension_pri(v)": [15000],
        "temp_huile(°c)": [sim_temp],
        "silicagel(%)": [sim_silica],
        "aspet _gen": [sim_aspect],
        "niveau_huile(°c)": ["vert"],
    }
    input_df = pd.DataFrame(input_dict)
    input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(
        columns=X_encoded.columns, fill_value=0
    )

    prob_fuite = model.predict_proba(input_encoded)[0][1] * 100
    prob_propre = 100 - prob_fuite

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
            <div class="metric-card {'alert' if prob_fuite > 40 else 'success'}">
                <div class="metric-title">Probabilité de Fuite (Buchings)</div>
                <div class="metric-value">{prob_fuite:.1f} %</div>
                <div class="metric-sub">Résultat du modèle prédictif</div>
            </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
            <div class="metric-card success">
                <div class="metric-title">Probabilité d'État Propre</div>
                <div class="metric-value">{prob_propre:.1f} %</div>
                <div class="metric-sub">Résultat du modèle prédictif</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⏳ Analyse de Survie & Fiabilité (Indicateurs MTTF & MTTR)")

    # Calculs de fiabilité
    total_obs = len(filtered_df)
    n_fuites = len(filtered_df[filtered_df["buchings"] == "fuite"])
    n_niveau_crit = len(
        filtered_df[filtered_df["niveau_huile(°c)"] == "rouge"]
    )

    # MTTF estimé en nombre de cycles d'inspection
    mttf_buchings = (
        (total_obs / n_fuites)
        if n_fuites > 0
        else total_obs
    )
    mttf_niveau = (
        (total_obs / n_niveau_crit)
        if n_niveau_crit > 0
        else total_obs
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric(
            "MTTF Estimé (Fuites Buchings)",
            f"{mttf_buchings:.1f} cycles",
            help="Temps moyen avant l'apparition d'une fuite aux traversées.",
        )
    with col_f2:
        st.metric(
            "MTTF Estimé (Niveau Huile Rouge)",
            f"{mttf_niveau:.1f} cycles",
            help="Temps moyen avant baisse critique du niveau d'huile.",
        )
    with col_f3:
        st.metric(
            "Taux de Défaillance (λ)",
            f"{(n_fuites / total_obs):.4f} /cycle",
            help="Proportion de pannes constatées par inspection.",
        )

    # Courbe de Survie Empirique (Kaplan-Meier simplifié)
    st.markdown("##### 📉 Courbe de Fiabilité Estimée $R(t)$")
    t_steps = np.arange(1, 20)
    lambda_param = n_fuites / total_obs if total_obs > 0 else 0.05
    reliability_curve = np.exp(-lambda_param * t_steps)

    fig_surv = go.Figure()
    fig_surv.add_trace(
        go.Scatter(
            x=t_steps,
            y=reliability_curve * 100,
            mode="lines+markers",
            name="Fiabilité R(t)",
            line=dict(color="#10B981", width=3),
        )
    )
    fig_surv.update_layout(
        title="Probabilité de maintien en bon état en fonction des cycles d'inspection",
        xaxis_title="Nombre d'inspections successives",
        yaxis_title="Fiabilité (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_surv, use_container_width=True)

# --- TAB 5 : REGISTRE DE DONNÉES ---
with tab_data:
    st.subheader("📋 Vue Intégrale des Inspections")
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Exporter le Registre Filtré (CSV)",
        data=csv_data,
        file_name="registre_transformateurs_gmao.csv",
        mime="text/csv",
    )
