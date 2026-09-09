from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import streamlit as st

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION DE LA PAGE & THÈME CSS
# ==========================================
st.set_page_config(
    page_title="GMAO - Analytics & Previsions Transfo MT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background-color: #0E1117;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
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

    df = df.sort_values(["transfo_id", "date"]).reset_index(drop=True)
    df["silica_diff"] = df.groupby("transfo_id")["silicagel(%)"].diff().fillna(0)

    df["critique"] = False
    mask_alert = (
        (df["temp_huile(°c)"] >= 50)
        | (df["niveau_huile(°c)"] == "rouge")
        | (df["buchings"] == "fuite")
        | (df["relais_buchh"] == "fuite")
    )
    df.loc[mask_alert, "critique"] = True

    return df


st.sidebar.image(
    "https://img.icons8.com/fluent/96/lightning-bolt.png", width=50
)
st.sidebar.title("GMAO Transfo MT")
st.sidebar.caption("Plateforme d'Analyse & Prédictions")

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
# 3. FILTRES DYNAMIQUES & BARRE DE RECHERCHE
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Recherche & Filtres d'Analyse")

# Barre de recherche textuelle globale
search_query = st.sidebar.text_input(
    "🔎 Barre de recherche globale",
    placeholder="Tapez un mot-clé (ex: TFO_01, fuite, sale, rouge...)",
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Période d'inspection",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

transfo_list = ["Tous les équipements"] + sorted(list(df["transfo_id"].unique()))
selected_transfo = st.sidebar.selectbox("Équipement ciblable", transfo_list)

filtered_df = df.copy()

# Application du filtre par date
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= date_range[0])
        & (filtered_df["date"].dt.date <= date_range[1])
    ]

# Application du filtre par équipement
if selected_transfo != "Tous les équipements":
    filtered_df = filtered_df[filtered_df["transfo_id"] == selected_transfo]

# Application de la recherche globale textuelle
if search_query:
    q = search_query.lower()
    match_mask = filtered_df.astype(str).apply(
        lambda row: row.str.lower().str.contains(q).any(), axis=1
    )
    filtered_df = filtered_df[match_mask]

# ==========================================
# 4. EN-TÊTE & KPIs GLOBAUX
# ==========================================
st.title("⚡ Dashboard & Prévisions Prédictives des Transformateurs MT")
st.markdown(
    "Surveillance opérationnelle, analyse de contingence et **prévision des risques de fuites aux traversées à 1 et 2 mois**."
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
            <div class="metric-title">Anomalies Relevées</div>
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
            <div class="metric-sub">Traversées actuellement défectueuses</div>
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
tab_overview, tab_contingency, tab_evolution, tab_predictions, tab_data = (
    st.tabs(
        [
            "📊 Vue d'ensemble",
            "🧮 Contingence & Croisements",
            "📈 Évolution Temporelle Transfo",
            "🔮 Prédictions à 1 & 2 Mois (IA)",
            "📋 Registre des Données",
        ]
    )
)

# --- TAB 1 : VUE D'ENSEMBLE ---
with tab_overview:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🌡️ Température d'Huile par Transformateur (°C)")
        if not filtered_df.empty:
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
        else:
            st.info("Aucune donnée disponible.")

    with col_b:
        st.subheader("💧 Répartition de l'État du Niveau d'Huile")
        if not filtered_df.empty:
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
        else:
            st.info("Aucune donnée disponible.")

# --- TAB 2 : TABLEAUX DE CONTINGENCE STYLISÉS & ROBUSTES ---
with tab_contingency:
    st.subheader(
        "🧮 Tableau de Contingence & Profils Lignes (Aspect Général vs Buchings)"
    )
    st.markdown(
        "Analyse de dépendance entre la propreté externe (Aspect Général) et la présence de fuites sur les traversées (Buchings)."
    )

    if not filtered_df.empty:
        ct_raw = pd.crosstab(
            filtered_df["aspet _gen"],
            filtered_df["buchings"],
            margins=True,
            margins_name="Total",
        )

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
                ct_raw.style.format("{:d}"),
                use_container_width=True,
            )

        with col_ct2:
            st.markdown("#### 📊 Profil Ligne (% de fuites par état d'aspect)")
            st.dataframe(
                ct_prop.style.format("{:.2f} %"),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("📉 Représentation Graphique de la Contingence")

        ct_prop_clean = ct_prop.reset_index().rename_axis(None, axis=1)
        value_vars = [c for c in ct_prop_clean.columns if c != "aspet _gen"]
        df_melted = pd.melt(
            ct_prop_clean,
            id_vars=["aspet _gen"],
            value_vars=value_vars,
            var_name="Etat_Buchings",
            value_name="Proportion",
        )

        fig_ct = px.bar(
            df_melted,
            x="aspet _gen",
            y="Proportion",
            color="Etat_Buchings",
            title="Proportion des états de Buchings selon l'Aspect Général (%)",
            labels={
                "aspet _gen": "Aspect Général",
                "Proportion": "Proportion (%)",
                "Etat_Buchings": "État Buchings",
            },
            barmode="group",
            template="plotly_dark",
            color_discrete_map={"propre": "#10B981", "fuite": "#EF4444"},
        )
        fig_ct.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.warning("Aucune donnée correspondant à votre recherche.")

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

# --- TAB 4 : PRÉDICTIONS RÉELLES À 1 & 2 MOIS (TOUS LES TRANSFORMATEURS) ---
with tab_predictions:
    st.subheader("🔮 Prévision des Risques de Fuite des Buchings (M+1 & M+2)")
    st.markdown(
        "Ce modèle entraine une **Régression Logistique** sur l'historique complet pour estimer la probabilité que chaque transformateur développe une fuite aux traversées (Buchings) dans **1 mois** et **2 mois**."
    )

    # Préparation du modèle de Machine Learning
    X_features = [
        "puissance(kva)",
        "tension_pri(v)",
        "temp_huile(°c)",
        "silicagel(%)",
        "aspet _gen",
        "niveau_huile(°c)",
    ]
    X_train = df[X_features]
    y_train = (df["buchings"] == "fuite").astype(int)

    X_encoded = pd.get_dummies(X_train, drop_first=True)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_encoded, y_train)

    # Récupération du dernier état connu de CHAQUE transformateur
    latest_df = (
        df.sort_values("date").groupby("transfo_id").last().reset_index()
    )

    forecast_results = []

    for _, row in latest_df.iterrows():
        t_id = row["transfo_id"]

        # État M+1 (Hypothèse de dégradation naturelle : légère hausse temp, baisse silicagel)
        feat_m1 = {
            "puissance(kva)": row["puissance(kva)"],
            "tension_pri(v)": row["tension_pri(v)"],
            "temp_huile(°c)": min(row["temp_huile(°c)"] + 2, 70),
            "silicagel(%)": max(row["silicagel(%)"] - 5, 20),
            "aspet _gen": row["aspet _gen"],
            "niveau_huile(°c)": row["niveau_huile(°c)"],
        }

        # État M+2 (Poursuite de dégradation)
        feat_m2 = {
            "puissance(kva)": row["puissance(kva)"],
            "tension_pri(v)": row["tension_pri(v)"],
            "temp_huile(°c)": min(row["temp_huile(°c)"] + 4, 75),
            "silicagel(%)": max(row["silicagel(%)"] - 10, 10),
            "aspet _gen": "sale"
            if row["aspet _gen"] == "sale"
            else row["aspet _gen"],
            "niveau_huile(°c)": row["niveau_huile(°c)"],
        }

        df_m1 = pd.DataFrame([feat_m1])
        df_m2 = pd.DataFrame([feat_m2])

        enc_m1 = pd.get_dummies(df_m1, drop_first=True).reindex(
            columns=X_encoded.columns, fill_value=0
        )
        enc_m2 = pd.get_dummies(df_m2, drop_first=True).reindex(
            columns=X_encoded.columns, fill_value=0
        )

        p_fuite_m1 = model.predict_proba(enc_m1)[0][1] * 100
        p_fuite_m2 = model.predict_proba(enc_m2)[0][1] * 100

        # Classification du niveau de risque M+2
        if p_fuite_m2 >= 50:
            risk_level = "🔴 Critique"
        elif p_fuite_m2 >= 25:
            risk_level = "🟠 Moyen"
        else:
            risk_level = "🟢 Faible"

        forecast_results.append(
            {
                "Transformateur": t_id,
                "Dernier Etat Buchings": row["buchings"],
                "Aspect Actuel": row["aspet _gen"],
                "Prob. Fuite M+1 (%)": round(p_fuite_m1, 1),
                "Prob. Fuite M+2 (%)": round(p_fuite_m2, 1),
                "Niveau de Risque (M+2)": risk_level,
            }
        )

    df_forecast = pd.DataFrame(forecast_results).sort_values(
        "Prob. Fuite M+2 (%)", ascending=False
    )

    # Affichage du Tableau Prédictif
    st.markdown("#### 📋 Tableau de Prévision des Risques par Transformateur")
    st.dataframe(df_forecast, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Comparatif des Probabilités de Fuite à M+1 et M+2")

    fig_prev = px.bar(
        df_forecast,
        x="Transformateur",
        y=["Prob. Fuite M+1 (%)", "Prob. Fuite M+2 (%)"],
        barmode="group",
        title="Évolution de la probabilité prédictive de fuite des traversées",
        template="plotly_dark",
        color_discrete_sequence=["#F59E0B", "#EF4444"],
    )
    fig_prev.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_prev, use_container_width=True)

# --- TAB 5 : REGISTRE DE DONNÉES ---
with tab_data:
    st.subheader("📋 Vue Intégrale des Inspections (Filtrée)")
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Exporter le Registre Filtré (CSV)",
        data=csv_data,
        file_name="registre_transformateurs_gmao.csv",
        mime="text/csv",
    )
