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
        df[col] = df[col].astype(str).str.strip()

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
    )
    df.loc[mask_alert, "critique"] = True

    return df


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

# Application des filtres de sélection
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
        "date",
        "transfo_id",
        "aspet _gen",
        "var_aspet_gen",
        "silicagel(%)",
        "var_silicagel(%)",
        "buchings",
        "var_buchings",
        "niveau_huile(°c)",
        "var_niveau_huile",
        "temp_huile(°c)",
        "var_temp_huile(°c)",
    ]

    df_display = filtered_df[disp_cols].rename(
        columns={
            "date": "Date Inspection",
            "transfo_id": "Transformateur",
            "aspet _gen": "Aspect Général",
            "var_aspet_gen": "Var. Aspect Général",
            "silicagel(%)": "Silicagel (%)",
            "var_silicagel(%)": "Δ Silicagel (%)",
            "buchings": "Buchings",
            "var_buchings": "Var. Buchings",
            "niveau_huile(°c)": "Niveau Huile",
            "var_niveau_huile": "Var. Niveau Huile",
            "temp_huile(°c)": "Temp Huile (°C)",
            "var_temp_huile(°c)": "Δ Temp (°C)",
        }
    )

    st.dataframe(
        df_display.style.format(
            {"Δ Silicagel (%)": "{:+.1f}", "Δ Temp (°C)": "{:+.1f}"}
        ),
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
tab_overview, tab_contingency, tab_evolution, tab_predictions, tab_data = st.tabs(
    [
        "📊 Vue d'ensemble",
        "🧮 Contingence & Croisements",
        "📈 Évolution Temporelle Transfo",
        "🔮 Prédictions à 1 & 2 Mois (IA)",
        "📋 Registre des Données",
    ]
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

# --- TAB 2 : CONTINGENCE & CROISEMENTS ---
with tab_contingency:
    st.subheader(
        "🧮 Tableau de Contingence & Profils Lignes (Aspect Général vs Buchings)"
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
        fig_ct.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.warning("Aucune donnée correspondant à votre recherche.")

# --- TAB 3 : ÉVOLUTION TEMPORELLE (ENRICHIE) ---
with tab_evolution:
    st.subheader("📈 Suivi Chronologique Détaillé par Transformateur")
    
    transfo_target = st.selectbox(
        "Sélectionner un transformateur à analyser :",
        sorted(list(df["transfo_id"].unique())),
        key="ev_transfo",
    )

    df_single = df[df["transfo_id"] == transfo_target].sort_values("date")

    if not df_single.empty:
        # --- SECTIONS HISTORIQUES demandées ---
        
        # 1. Historique Température d'Huile & Silicagel
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
            # Ligne de seuil critique (50°C)
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

        # 2. Historique États des Buchings & Aspect Général
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
            fig_buch.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
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
            fig_asp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_asp, use_container_width=True)

        st.markdown("---")

        # 3. Historique Relais Buchholz & Niveau d'Huile
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
                fig_buchh.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
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
            fig_oil_ev.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_oil_ev, use_container_width=True)

    else:
        st.warning("Aucune donnée disponible pour cet équipement.")

# --- TAB 4 : PRÉDICTIONS RÉELLES (IA) ---
with tab_predictions:
    st.subheader("🔮 Prévision des Risques de Fuite des Buchings (M+1 & M+2)")
    st.markdown(
        "Prévisions réalisées par modèle prédictif sur l'état futur des traversées."
    )

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

    latest_df = df.sort_values("date").groupby("transfo_id").last().reset_index()

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
