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
    page_title="GMAO - Analytics & Prévisions Transfo MT",
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
# INITIALISATION DE LA SESSION DE STOCK
# ==========================================
if "stock_data" not in st.session_state:
    st.session_state.stock_data = [
        {"Code": "PR-001", "Désignation": "Gel de Silice (Kg)", "Stock Actuel": 45, "Stock Min": 50, "Prix Unitaire ($)": 15},
        {"Code": "PR-002", "Désignation": "Joint Traversée Buchings", "Stock Actuel": 12, "Stock Min": 10, "Prix Unitaire ($)": 85},
        {"Code": "PR-003", "Désignation": "Huile Minérale Isolante (L)", "Stock Actuel": 200, "Stock Min": 300, "Prix Unitaire ($)": 6},
        {"Code": "PR-004", "Désignation": "Relais Buchholz Flotteur", "Stock Actuel": 3, "Stock Min": 5, "Prix Unitaire ($)": 450},
        {"Code": "PR-005", "Désignation": "Indicateur Niveau Huile", "Stock Actuel": 8, "Stock Min": 4, "Prix Unitaire ($)": 120},
    ]

# ==========================================
# 2. FONCTIONS DE CHARGEMENT & PRÉPARATION
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
def process_data(source):
    df = pd.read_excel(source)
    df.columns = df.columns.str.strip()

    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["transfo_id", "date"]).reset_index(drop=True)

    # Calcul des variations inter-inspections
    df["var_silicagel(%)"] = df.groupby("transfo_id")["silicagel(%)"].diff().fillna(0)
    df["var_temp_huile(°c)"] = df.groupby("transfo_id")["temp_huile(°c)"].diff().fillna(0)

    # Suivi des changements d'état qualitatifs
    df["prev_aspet_gen"] = df.groupby("transfo_id")["aspet _gen"].shift(1)
    df["var_aspet_gen"] = np.where(
        df["prev_aspet_gen"].isna(),
        "Initial",
        np.where(
            df["aspet _gen"] == df["prev_aspet_gen"],
            "Inchangé",
            df["prev_aspet_gen"] + " ➔ " + df["aspet _gen"],
        ),
    )

    df["prev_buchings"] = df.groupby("transfo_id")["buchings"].shift(1)
    df["var_buchings"] = np.where(
        df["prev_buchings"].isna(),
        "Initial",
        np.where(
            df["buchings"] == df["prev_buchings"],
            "Inchangé",
            df["prev_buchings"] + " ➔ " + df["buchings"],
        ),
    )

    df["prev_niveau_huile"] = df.groupby("transfo_id")["niveau_huile(°c)"].shift(1)
    df["var_niveau_huile"] = np.where(
        df["prev_niveau_huile"].isna(),
        "Initial",
        np.where(
            df["niveau_huile(°c)"] == df["prev_niveau_huile"],
            "Inchangé",
            df["prev_niveau_huile"] + " ➔ " + df["niveau_huile(°c)"],
        ),
    )

    df["critique"] = False
    mask_alert = (
        (df["temp_huile(°c)"] >= 50)
        | (df["niveau_huile(°c)"] == "rouge")
        | (df["buchings"] == "fuite")
        | (df.get("relais_buchh", pd.Series([""] * len(df))) == "fuite")
        | (df["silicagel(%)"] <= 40)
    )
    df.loc[mask_alert, "critique"] = True

    return df


# ==========================================
# FONCTION D'AFFICHAGE ET D'EXPORT DES ALERTES
# ==========================================
def afficher_alertes(df_data, tab_prefix="main"):
    """Analyse les données, affiche les alertes et génère des tableaux téléchargeables en CSV."""
    if df_data.empty:
        return

    derniers = df_data.sort_values("date").groupby("transfo_id").last().reset_index()

    df_silicagel = derniers[derniers["silicagel(%)"] <= 40]
    df_huile_jaune = derniers[derniers["niveau_huile(°c)"].astype(str).str.lower() == "jaune"]
    df_huile_rouge = derniers[derniers["niveau_huile(°c)"].astype(str).str.lower() == "rouge"]
    
    mask_fuite = (derniers["buchings"].astype(str).str.lower() == "fuite")
    if "relais_buchh" in derniers.columns:
        mask_fuite = mask_fuite | (derniers["relais_buchh"].astype(str).str.lower() == "fuite")
    df_fuites = derniers[mask_fuite]

    has_alerts = not (df_silicagel.empty and df_huile_jaune.empty and df_huile_rouge.empty and df_fuites.empty)

    if has_alerts:
        st.markdown("### 🚨 Centre d'Alertes & Actions Recommandées")

        cols_export = ["transfo_id", "date", "silicagel(%)", "niveau_huile(°c)", "buchings", "temp_huile(°c)"]
        cols_export_exist = [c for c in cols_export if c in derniers.columns]

        if not df_silicagel.empty:
            st.error(f"⚠️ **Alerte Silicagel (≤ 40%) - {len(df_silicagel)} transformateur(s) à remplacer**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df_silicagel[cols_export_exist].rename(columns={
                    "transfo_id": "Transformateur", "date": "Date Relevé",
                    "silicagel(%)": "Silicagel (%)", "niveau_huile(°c)": "Niveau Huile",
                    "buchings": "Buchings", "temp_huile(°c)": "Temp (°C)"
                }), use_container_width=True)
            with c2:
                csv_sil = df_silicagel[cols_export_exist].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger CSV (Silicagel)",
                    data=csv_sil,
                    file_name="alertes_silicagel_remplacement.csv",
                    mime="text/csv",
                    key=f"btn_sil_{tab_prefix}"
                )

        if not df_huile_jaune.empty:
            st.warning(f"🟡 **Alerte Niveau d'Huile (Jaune) - {len(df_huile_jaune)} transformateur(s) : Programmer l'appoint d'huile**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df_huile_jaune[cols_export_exist].rename(columns={
                    "transfo_id": "Transformateur", "date": "Date Relevé",
                    "silicagel(%)": "Silicagel (%)", "niveau_huile(°c)": "Niveau Huile",
                    "buchings": "Buchings", "temp_huile(°c)": "Temp (°C)"
                }), use_container_width=True)
            with c2:
                csv_hj = df_huile_jaune[cols_export_exist].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger CSV (Huile Jaune)",
                    data=csv_hj,
                    file_name="alertes_niveau_huile_jaune.csv",
                    mime="text/csv",
                    key=f"btn_hj_{tab_prefix}"
                )

        if not df_huile_rouge.empty:
            st.error(f"🔴 **Alerte Niveau d'Huile (Rouge) - {len(df_huile_rouge)} transformateur(s) : Appoint d'huile Nécessaire**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df_huile_rouge[cols_export_exist].rename(columns={
                    "transfo_id": "Transformateur", "date": "Date Relevé",
                    "silicagel(%)": "Silicagel (%)", "niveau_huile(°c)": "Niveau Huile",
                    "buchings": "Buchings", "temp_huile(°c)": "Temp (°C)"
                }), use_container_width=True)
            with c2:
                csv_hr = df_huile_rouge[cols_export_exist].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger CSV (Huile Rouge)",
                    data=csv_hr,
                    file_name="alertes_niveau_huile_rouge.csv",
                    mime="text/csv",
                    key=f"btn_hr_{tab_prefix}"
                )

        if not df_fuites.empty:
            st.error(f"💧 **Alerte Fuite Décelée - {len(df_fuites)} transformateur(s) à traiter (éliminer les fuites)**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df_fuites[cols_export_exist].rename(columns={
                    "transfo_id": "Transformateur", "date": "Date Relevé",
                    "silicagel(%)": "Silicagel (%)", "niveau_huile(°c)": "Niveau Huile",
                    "buchings": "Buchings", "temp_huile(°c)": "Temp (°C)"
                }), use_container_width=True)
            with c2:
                csv_f = df_fuites[cols_export_exist].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger CSV (Fuites)",
                    data=csv_f,
                    file_name="alertes_fuites_transformateurs.csv",
                    mime="text/csv",
                    key=f"btn_f_{tab_prefix}"
                )

        st.markdown("---")


# ==========================================
# 3. BARRE LATÉRALE & EN-TÊTE
# ==========================================
st.sidebar.image("https://img.icons8.com/fluent/96/lightning-bolt.png", width=50)
st.sidebar.title("GMAO Transfo MT")
st.sidebar.caption("Plateforme d'Analyse & Prédictions")

col_title, col_search_box = st.columns([1.3, 1.7])

with col_title:
    st.title("⚡ Dashboard & Prévisions MT")
    st.caption("Surveillance opérationnelle et analyse détaillée des variations.")

with col_search_box:
    st.markdown("### 🔍 Chargement & Recherche Rapide")

    uploaded_file = st.file_uploader(
        "📂 Téléverser un fichier Excel à examiner (.xlsx, .xls)",
        type=["xlsx", "xls"],
        key="main_excel_uploader",
    )

    if uploaded_file:
        df = process_data(uploaded_file)
        st.success("✅ Fichier personnalisé chargé avec succès !")
    else:
        dataset_file = find_dataset()
        if dataset_file:
            df = process_data(dataset_file)
        else:
            st.error("⚠️ Aucun fichier détecté. Veuillez charger un fichier Excel ci-dessus.")
            st.stop()

    sc1, sc2 = st.columns(2)
    with sc1:
        transfo_list = ["Tous les équipements"] + sorted(list(df["transfo_id"].unique()))
        selected_transfo = st.selectbox("Équipement ciblable", transfo_list)
    with sc2:
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.date_input(
            "Période d'inspection",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

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
# 4. TABLEAU DE SYNTHÈSE DES VARIATIONS
# ==========================================
if selected_transfo != "Tous les équipements":
    st.markdown(f"### 📋 Rapport Synthétique d'Aspects & Variations : **{selected_transfo}**")

    disp_cols = [
        "date", "transfo_id", "aspet _gen", "var_aspet_gen",
        "silicagel(%)", "var_silicagel(%)", "buchings", "var_buchings",
        "niveau_huile(°c)", "var_niveau_huile", "temp_huile(°c)", "var_temp_huile(°c)",
    ]

    df_display = filtered_df[disp_cols].rename(
        columns={
            "date": "Date Inspection", "transfo_id": "Transformateur",
            "aspet _gen": "Aspect Général", "var_aspet_gen": "Var. Aspect Général",
            "silicagel(%)": "Silicagel (%)", "var_silicagel(%)": "Δ Silicagel (%)",
            "buchings": "Buchings", "var_buchings": "Var. Buchings",
            "niveau_huile(°c)": "Niveau Huile", "var_niveau_huile": "Var. Niveau Huile",
            "temp_huile(°c)": "Temp Huile (°C)", "var_temp_huile(°c)": "Δ Temp (°C)",
        }
    )

    st.dataframe(
        df_display.style.format({"Δ Silicagel (%)": "{:+.1f}", "Δ Temp (°C)": "{:+.1f}"}),
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. KPIS GLOBAUX
# ==========================================
total_inspections = len(filtered_df)
anomalies_count = len(filtered_df[filtered_df["critique"] == True])
fuites_buchings = len(filtered_df[filtered_df["buchings"] == "fuite"])
silica_var_moy = (
    filtered_df["var_silicagel(%)"].abs().mean() if not filtered_df.empty else 0
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
# 6. ONGLETS D'ANALYSE
# ==========================================
(
    tab_overview,
    tab_contingency,
    tab_evolution,
    tab_predictions,
    tab_mtbf,
    tab_spc,
    tab_stock,
    tab_kpi_maint,
    tab_data,
) = st.tabs(
    [
        "📊 Vue d'ensemble",
        "🧮 Contingence & Croisements",
        "📈 Évolution Temporelle Transfo",
        "🔮 Prédictions IA",
        "📉 Suivi Pannes (MTBF/MTTR)",
        "🎯 Suivi SPC (Cp/Cpk)",
        "📦 Pièces de Rechange",
        "📊 Dashboard KPI Maintenance",
        "📋 Registre des Données",
    ]
)

# --- TAB 1 : VUE D'ENSEMBLE ---
with tab_overview:
    afficher_alertes(filtered_df, tab_prefix="overview")

    nb_total_transfos = filtered_df["transfo_id"].nunique()
    transfos_critiques = filtered_df[filtered_df["critique"] == True]["transfo_id"].nunique()
    health_score = max(0, int(((nb_total_transfos - transfos_critiques) / max(1, nb_total_transfos)) * 100))

    st.markdown("#### 🛡️ Indice de Santé Global du Parc")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("Indice de Santé (Health Score)", f"{health_score} %", delta=f"{'-' if health_score < 80 else '+'}{100-health_score}% risque")
    with h2:
        st.metric("Transformateurs en Alerte", f"{transfos_critiques} / {nb_total_transfos}")
    with h3:
        st.metric("Taux d'Anomalie Global", f"{round((transfos_critiques/max(1, nb_total_transfos))*100, 1)} %")

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.subheader("🌡️ Distribution des Températures")
        if not filtered_df.empty:
            fig_temp = px.box(
                filtered_df,
                x="transfo_id",
                y="temp_huile(°c)",
                color="aspet _gen",
                template="plotly_dark",
                color_discrete_map={"propre": "#3B82F6", "sale": "#EF4444"},
            )
            fig_temp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Aucune donnée disponible.")

    with col_b:
        st.subheader("💧 Niveau d'Huile vs Buchings")
        if not filtered_df.empty:
            fig_huile = px.histogram(
                filtered_df,
                x="niveau_huile(°c)",
                color="buchings",
                barmode="group",
                template="plotly_dark",
                color_discrete_map={"propre": "#10B981", "fuite": "#EF4444"},
            )
            fig_huile.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_huile, use_container_width=True)
        else:
            st.info("Aucune donnée disponible.")

    with col_c:
        st.subheader("🍩 Répartition des Anomalies")
        if not filtered_df.empty:
            nb_fuite_b = len(filtered_df[filtered_df["buchings"] == "fuite"])
            nb_temp_haute = len(filtered_df[filtered_df["temp_huile(°c)"] >= 50])
            nb_silicagel_bas = len(filtered_df[filtered_df["silicagel(%)"] <= 40])
            
            df_defauts = pd.DataFrame({
                "Type": ["Fuite Buchings", "Surchauffe (>50°C)", "Silicagel Bas (≤40%)"],
                "Nombre": [nb_fuite_b, nb_temp_haute, nb_silicagel_bas]
            })
            
            fig_donut = px.pie(
                df_defauts, 
                names="Type", 
                values="Nombre", 
                hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=["#EF4444", "#F59E0B", "#3B82F6"]
            )
            fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Aucune donnée disponible.")

    st.markdown("---")

    st.subheader("⚠️ Top 5 des Équipements à Intervenir en Priorité")
    df_top_critique = filtered_df[filtered_df["critique"] == True].groupby("transfo_id").last().reset_index()
    if not df_top_critique.empty:
        cols_prio = ["transfo_id", "date", "temp_huile(°c)", "silicagel(%)", "buchings", "niveau_huile(°c)", "aspet _gen"]
        st.dataframe(df_top_critique[cols_prio].rename(columns={
            "transfo_id": "Transformateur",
            "date": "Dernière Inspection",
            "temp_huile(°c)": "Temp (°C)",
            "silicagel(%)": "Silicagel (%)",
            "buchings": "État Buchings",
            "niveau_huile(°c)": "Niveau Huile",
            "aspet _gen": "Aspect Général"
        }), use_container_width=True)
    else:
        st.success("✅ Aucun transformateur en état critique détecté pour la sélection actuelle.")

# --- TAB 2 : CONTINGENCE & CROISEMENTS ---
with tab_contingency:
    st.subheader("🧮 Tableau de Contingence & Profils Lignes (Aspect Général vs Buchings)")
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
            st.dataframe(ct_raw.style.format("{:d}"), use_container_width=True)

        with col_ct2:
            st.markdown("#### 📊 Profil Ligne (% de fuites par état d'aspect)")
            st.dataframe(ct_prop.style.format("{:.2f} %"), use_container_width=True)

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
        fig_ct.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.warning("Aucune donnée correspondant à votre recherche.")

# --- TAB 3 : ÉVOLUTION TEMPORELLE ---
with tab_evolution:
    st.subheader("📈 Suivi Chronologique Détaillé par Transformateur")
    
    transfo_target = st.selectbox(
        "Sélectionner un transformateur à analyser :",
        sorted(list(df["transfo_id"].unique())),
        key="ev_transfo",
    )

    df_single = df[df["transfo_id"] == transfo_target].sort_values("date")
    
    afficher_alertes(df_single, tab_prefix="evolution")

    if not df_single.empty:
        st.markdown("#### 🌡️ 1. Historique Température d'Huile & Silicagel (%)")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_temp_ev = go.Figure()
            fig_temp_ev.add_trace(
                go.Scatter(
                    x=df_single["date"],
                    y=df_single["temp_huile(°c)"],
                    mode="lines+markers",
                    name="Temp Huile (°C)",
                    line=dict(color="#EF4444", width=3),
                )
            )
            fig_temp_ev.add_hline(
                y=50, line_dash="dash", line_color="#F59E0B", annotation_text="Seuil d'alerte (50°C)"
            )
            fig_temp_ev.update_layout(
                title="Évolution de la Température d'Huile (°C)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_temp_ev, use_container_width=True)

        with col_t2:
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
            fig_sil_ev.add_hline(
                y=40, line_dash="dash", line_color="#EF4444", annotation_text="Seuil de remplacement (40%)"
            )
            fig_sil_ev.add_trace(
                go.Bar(
                    x=df_single["date"],
                    y=df_single["var_silicagel(%)"],
                    name="Variation Δ (%)",
                    marker_color="#F59E0B",
                    opacity=0.6,
                )
            )
            fig_sil_ev.update_layout(
                title="Évolution du Silicagel (%) & Deltas",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sil_ev, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 🔍 2. Historique des États (Buchings & Aspect Général)")
        col_ev3, col_ev4 = st.columns(2)
        with col_ev3:
            fig_buch = px.scatter(
                df_single,
                x="date",
                y="buchings",
                color="buchings",
                title="Historique des États des Buchings (Traversées)",
                color_discrete_map={"propre": "#10B981", "fuite": "#EF4444"},
                template="plotly_dark",
            )
            fig_buch.update_traces(marker=dict(size=14, symbol="square"))
            fig_buch.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_buch, use_container_width=True)

        with col_ev4:
            fig_asp = px.scatter(
                df_single,
                x="date",
                y="aspet _gen",
                color="aspet _gen",
                title="Historique Aspect Général",
                color_discrete_map={"propre": "#3B82F6", "sale": "#EF4444"},
                template="plotly_dark",
            )
            fig_asp.update_traces(marker=dict(size=14, symbol="diamond"))
            fig_asp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_asp, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 🛡️ 3. Historique Relais Buchholz & Niveau d'Huile")
        col_ev5, col_ev6 = st.columns(2)
        with col_ev5:
            if "relais_buchh" in df_single.columns:
                fig_buchh = px.scatter(
                    df_single,
                    x="date",
                    y="relais_buchh",
                    color="relais_buchh",
                    title="Historique Relais Buchholz",
                    color_discrete_map={"propre": "#10B981", "fuite": "#EF4444", "sans": "#9CA3AF"},
                    template="plotly_dark",
                )
                fig_buchh.update_traces(marker=dict(size=14, symbol="circle"))
                fig_buchh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_buchh, use_container_width=True)
            else:
                st.info("Colonne 'relais_buchh' non trouvée dans le dataset.")

        with col_ev6:
            color_map_huile = {"vert": "#10B981", "jaune": "#F59E0B", "rouge": "#EF4444"}
            fig_oil_ev = px.scatter(
                df_single,
                x="date",
                y="niveau_huile(°c)",
                color="niveau_huile(°c)",
                size="temp_huile(°c)",
                title="Historique du Niveau d'Huile (Taille = Température)",
                color_discrete_map=color_map_huile,
                template="plotly_dark",
            )
            fig_oil_ev.update_traces(marker=dict(size=14))
            fig_oil_ev.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_oil_ev, use_container_width=True)

    else:
        st.warning("Aucune donnée disponible pour cet équipement.")

# --- TAB 4 : PRÉDICTIONS RÉELLES (IA) ---
with tab_predictions:
    st.subheader("🔮 Prévision des Risques de Fuite des Buchings (M+1 & M+2)")
    st.markdown("Prévisions réalisées par modèle prédictif sur l'état futur des traversées.")

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

    latest_df = filtered_df.sort_values("date").groupby("transfo_id").last().reset_index()

    forecast_results = []
    for _, row in latest_df.iterrows():
        t_id = row["transfo_id"]

        feat_m1 = {
            "puissance(kva)": row["puissance(kva)"],
            "tension_pri(v)": row["tension_pri(v)"],
            "temp_huile(°c)": min(row["temp_huile(°c)"] + 2, 70),
            "silicagel(%)": max(row["silicagel(%)"] - 5, 20),
            "aspet _gen": row["aspet _gen"],
            "niveau_huile(°c)": row["niveau_huile(°c)"],
        }

        feat_m2 = {
            "puissance(kva)": row["puissance(kva)"],
            "tension_pri(v)": row["tension_pri(v)"],
            "temp_huile(°c)": min(row["temp_huile(°c)"] + 4, 75),
            "silicagel(%)": max(row["silicagel(%)"] - 10, 10),
            "aspet _gen": "sale" if row["aspet _gen"] == "sale" else row["aspet _gen"],
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

        risk_level = (
            "🔴 Critique"
            if p_fuite_m2 >= 50
            else ("🟠 Moyen" if p_fuite_m2 >= 25 else "🟢 Faible")
        )

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

    if forecast_results:
        df_forecast = pd.DataFrame(forecast_results).sort_values(
            "Prob. Fuite M+2 (%)", ascending=False
        )
        st.markdown("#### 📋 Tableau de Prévision par Transformateur")
        st.dataframe(df_forecast, use_container_width=True)

        fig_prev = px.bar(
            df_forecast,
            x="Transformateur",
            y=["Prob. Fuite M+1 (%)", "Prob. Fuite M+2 (%)"],
            barmode="group",
            title="Évolution prédictive des risques de fuite aux traversées",
            template="plotly_dark",
            color_discrete_sequence=["#F59E0B", "#EF4444"],
        )
        fig_prev.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)
        st.plotly_chart(fig_prev, use_container_width=True)
    else:
        st.info("Aucune donnée pour la prédiction.")

# --- TAB 5 : SUIVI DES PANNES (MTBF / MTTR) ---
with tab_mtbf:
    st.subheader("📉 Indicateurs de Fiabilité : MTBF & MTTR par Transformateur")
    st.markdown(
        "Calculs basés sur le temps moyen de bon fonctionnement entre défaillances (**MTBF**) et le temps moyen de réparation (**MTTR**)."
    )

    mtbf_data = []
    for transfo_id, group in filtered_df.groupby("transfo_id"):
        group = group.sort_values("date")
        anomalies = group[group["critique"] == True]
        nb_pannes = len(anomalies)

        total_days = (
            (group["date"].max() - group["date"].min()).days
            if len(group) > 1
            else 30
        )
        total_hours = max(total_days * 24, 720)

        downtime_hours = nb_pannes * 8
        operating_hours = max(total_hours - downtime_hours, 1)

        mtbf = (
            round(operating_hours / nb_pannes, 1)
            if nb_pannes > 0
            else operating_hours
        )
        mttr = round(downtime_hours / nb_pannes, 1) if nb_pannes > 0 else 0
        disponibilite = round((operating_hours / total_hours) * 100, 2)

        mtbf_data.append(
            {
                "Transformateur": transfo_id,
                "Nombre de Pannes / Alertes": nb_pannes,
                "Heures Fonct. Total (h)": operating_hours,
                "Temps d'Arrêt Total (h)": downtime_hours,
                "MTBF (h)": mtbf,
                "MTTR (h)": mttr,
                "Disponibilité (%)": disponibilite,
            }
        )

    df_mtbf = pd.DataFrame(mtbf_data).sort_values("Disponibilité (%)")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("MTBF Moyen Parc", f"{round(df_mtbf['MTBF (h)'].mean(), 1)} h")
    with m2:
        st.metric("MTTR Moyen Parc", f"{round(df_mtbf['MTTR (h)'].mean(), 1)} h")
    with m3:
        st.metric(
            "Taux de Disponibilité Moyen",
            f"{round(df_mtbf['Disponibilité (%)'].mean(), 1)} %",
        )

    st.markdown("---")
    st.markdown("#### 📋 Synthèse Fiabilité par Équipement")
    st.dataframe(df_mtbf, use_container_width=True)

    fig_mtbf = px.bar(
        df_mtbf,
        x="Transformateur",
        y=["MTBF (h)", "MTTR (h)"],
        barmode="group",
        title="Comparatif MTBF vs MTTR par Transformateur (Heures)",
        template="plotly_dark",
        color_discrete_sequence=["#10B981", "#EF4444"],
    )
    fig_mtbf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)
    st.plotly_chart(fig_mtbf, use_container_width=True)

# --- TAB 6 : SUIVI SPC (INDICES Cp / Cpk) ---
with tab_spc:
    st.subheader("🎯 Maîtrise Statistique des Procédés (SPC / MSP)")
    st.markdown(
        "Évaluation de la capabilité du procédé d'exploitation des transformateurs ($C_p$ et $C_{pk}$)."
    )

    metric_spc = st.radio(
        "Sélectionner le paramètre physique à contrôler :",
        ["Température Huile (°C)", "Taux de Silicagel (%)"],
        horizontal=True,
    )

    if metric_spc == "Température Huile (°C)":
        data_spc = filtered_df["temp_huile(°c)"].dropna()
        usl, lsl = 50.0, 20.0
    else:
        data_spc = filtered_df["silicagel(%)"].dropna()
        usl, lsl = 100.0, 40.0

    if len(data_spc) > 5:
        mean_val = data_spc.mean()
        std_val = data_spc.std()

        cp = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
        cpu = (usl - mean_val) / (3 * std_val) if std_val > 0 else 0
        cpl = (mean_val - lsl) / (3 * std_val) if std_val > 0 else 0
        cpk = min(cpu, cpl)

        ucl = mean_val + 3 * std_val
        lcl = max(0, mean_val - 3 * std_val)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Moyenne (μ)", f"{mean_val:.2f}")
        with c2:
            st.metric("Écart-Type (σ)", f"{std_val:.2f}")
        with c3:
            st.metric("Indice Capabilité Cp", f"{cp:.2f}")
        with c4:
            st.metric(
                "Indice Capabilité Cpk",
                f"{cpk:.2f}",
                delta="Conforme" if cpk >= 1.33 else "Incapable/Resserrer",
            )

        st.markdown("---")
        st.markdown("#### 📈 Carte de Contrôle X-bar (Limites Statistiques UCL / LCL)")

        fig_spc = go.Figure()
        fig_spc.add_trace(
            go.Scatter(
                y=data_spc.values,
                mode="lines+markers",
                name="Valeur Relevée",
                line=dict(color="#3B82F6"),
            )
        )
        fig_spc.add_hline(
            y=mean_val,
            line_color="#10B981",
            annotation_text=f"Moyenne ({mean_val:.1f})",
        )
        fig_spc.add_hline(
            y=ucl,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"UCL (+3σ = {ucl:.1f})",
        )
        fig_spc.add_hline(
            y=lcl,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"LCL (-3σ = {lcl:.1f})",
        )
        fig_spc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_spc, use_container_width=True)
    else:
        st.info("Données insuffisantes pour calculer la capabilité SPC.")

# ==========================================
# ⚡ TAB 7 : GESTION DYNAMIQUE DES PIÈCES DE RECHANGE
# ==========================================
with tab_stock:
    st.subheader("📦 Gestion Dynamique & Interactive des Pièces de Rechange")
    st.markdown("Saisissez directement vos nouvelles pièces ci-dessous : l'application effectue l'analyse, déclenche les alertes et enregistre les données.")

    # Formulaire d'insertion de nouvelle pièce
    with st.expander("➕ Insérer une nouvelle pièce de rechange", expanded=True):
        with st.form("form_add_stock", clear_on_submit=True):
            col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
            with col_f1:
                new_code = st.text_input("Code Pièce", value=f"PR-00{len(st.session_state.stock_data)+1}")
            with col_f2:
                new_designation = st.text_input("Désignation", placeholder="Ex: Joint de cuve")
            with col_f3:
                new_stock_actuel = st.number_input("Stock Actuel", min_value=0, value=10, step=1)
            with col_f4:
                new_stock_min = st.number_input("Stock Min (Alerte)", min_value=0, value=5, step=1)
            with col_f5:
                new_pu = st.number_input("Prix Unitaire ($)", min_value=0.0, value=50.0, step=5.0)

            btn_ajouter = st.form_submit_button("💾 Enregistrer la pièce dans le stock")

            if btn_ajouter:
                if new_designation.strip() != "":
                    new_item = {
                        "Code": new_code.strip(),
                        "Désignation": new_designation.strip(),
                        "Stock Actuel": int(new_stock_actuel),
                        "Stock Min": int(new_stock_min),
                        "Prix Unitaire ($)": float(new_pu),
                    }
                    st.session_state.stock_data.append(new_item)
                    st.success(f"✅ Pièce '{new_designation}' ajoutée avec succès !")
                else:
                    st.error("⚠️ Veuillez saisir une désignation valide.")

    # Transformation des données en DataFrame et calculs automatiques
    df_stock = pd.DataFrame(st.session_state.stock_data)

    df_stock["Statut"] = np.where(
        df_stock["Stock Actuel"] < df_stock["Stock Min"],
        "⚠️ Recommander",
        "✅ Suffisant",
    )
    df_stock["Valeur Stock ($)"] = df_stock["Stock Actuel"] * df_stock["Prix Unitaire ($)"]

    st.markdown("---")

    # Metrics
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Valeur Totale du Stock", f"{df_stock['Valeur Stock ($)'].sum():,.2f} $")
    with s2:
        recom_count = len(df_stock[df_stock["Stock Actuel"] < df_stock["Stock Min"]])
        st.metric("Articles en Alerte (Sous le Min)", recom_count, delta=f"{recom_count} à commander", delta_color="inverse")
    with s3:
        st.metric("Total Références en Stock", len(df_stock))

    st.markdown("#### 📋 Etat du Stock mis à jour")
    
    # Mise en forme visuelle des lignes selon alerte
    st.dataframe(
        df_stock.style.apply(
            lambda row: ["background-color: #3b1719; color: #f87171" if row["Statut"] == "⚠️ Recommander" else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
    )

    c_exp, c_reset = st.columns([3, 1])
    with c_exp:
        csv_stock = df_stock.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Exporter le stock à jour (CSV)",
            data=csv_stock,
            file_name="gestion_stock_pieces_rechange.csv",
            mime="text/csv",
        )
    with c_reset:
        if st.button("🔄 Réinitialiser le Stock par Défaut"):
            st.session_state.stock_data = [
                {"Code": "PR-001", "Désignation": "Gel de Silice (Kg)", "Stock Actuel": 45, "Stock Min": 50, "Prix Unitaire ($)": 15},
                {"Code": "PR-002", "Désignation": "Joint Traversée Buchings", "Stock Actuel": 12, "Stock Min": 10, "Prix Unitaire ($)": 85},
                {"Code": "PR-003", "Désignation": "Huile Minérale Isolante (L)", "Stock Actuel": 200, "Stock Min": 300, "Prix Unitaire ($)": 6},
                {"Code": "PR-004", "Désignation": "Relais Buchholz Flotteur", "Stock Actuel": 3, "Stock Min": 5, "Prix Unitaire ($)": 450},
                {"Code": "PR-005", "Désignation": "Indicateur Niveau Huile", "Stock Actuel": 8, "Stock Min": 4, "Prix Unitaire ($)": 120},
            ]
            st.rerun()

# --- TAB 8 : TABLEAU DE BORD KPI MAINTENANCE ---
with tab_kpi_maint:
    st.subheader("📊 Tableau de Bord Stratégique KPI Maintenance")
    st.markdown(
        "Vision synthétique des performances globales du service de maintenance."
    )

    nb_transfo_total = filtered_df["transfo_id"].nunique()
    total_releves = len(filtered_df)
    anomalies_totales = len(filtered_df[filtered_df["critique"] == True])

    taux_dispo_parc = round(
        ((nb_transfo_total - filtered_df[filtered_df["critique"] == True]["transfo_id"].nunique()) / max(1, nb_transfo_total)) * 100, 1
    )
    taux_preventif = round(
        ((total_releves - anomalies_totales) / max(1, total_releves)) * 100, 1
    )
    cout_defaillance_est = anomalies_totales * 450

    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1:
        st.metric("Disponibilité Opérationnelle", f"{taux_dispo_parc} %")
    with kp2:
        st.metric("Taux de Maintenance Préventive", f"{taux_preventif} %")
    with kp3:
        st.metric("Coût Estimé des Défaillances", f"{cout_defaillance_est:,} $")
    with kp4:
        st.metric("TRS / OEE Estimé Parc", f"{round(taux_dispo_parc * 0.92, 1)} %")

    st.markdown("---")

    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.markdown("#### 🔄 Ratio Maintenance Préventive vs Curative")
        df_maint_ratio = pd.DataFrame(
            {
                "Type Maintenance": ["Préventive (Conforme)", "Curative (Alertes)"],
                "Volume": [total_releves - anomalies_totales, anomalies_totales],
            }
        )
        fig_pie_maint = px.pie(
            df_maint_ratio,
            names="Type Maintenance",
            values="Volume",
            hole=0.4,
            template="plotly_dark",
            color_discrete_sequence=["#10B981", "#EF4444"],
        )
        fig_pie_maint.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie_maint, use_container_width=True)

    with col_kpi2:
        st.markdown("#### 🚨 Répartition Mensuelle des Incidents")
        filtered_df["Mois"] = filtered_df["date"].dt.to_period("M").astype(str)
        df_incidents_mois = (
            filtered_df[filtered_df["critique"] == True]
            .groupby("Mois")
            .size()
            .reset_index(name="Nombre Incidents")
        )

        fig_bar_inc = px.bar(
            df_incidents_mois,
            x="Mois",
            y="Nombre Incidents",
            title="Évolution Mensuelle des Incidents Détectés",
            template="plotly_dark",
            color_discrete_sequence=["#F59E0B"],
        )
        fig_bar_inc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar_inc, use_container_width=True)

# --- TAB 9 : REGISTRE DE DONNÉES ---
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
