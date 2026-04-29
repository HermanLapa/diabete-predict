import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Diabète Predict",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS personnalisé
st.markdown("""
<style>
    /* Style du header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Style des cartes */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Style des résultats */
    .result-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .result-low {
        background: linear-gradient(135deg, #2ed573 0%, #26ae60 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .result-moderate {
        background: linear-gradient(135deg, #ffa502 0%, #e67e22 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    
    /* Style des métriques */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        background: #f8f9fa;
        border-radius: 15px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CHARGEMENT DU MODÈLE ====================
@st.cache_resource
def charger_modele():
    """Charge le modèle entraîné"""
    try:
        with open('modele_diabete.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Modèle non trouvé ! Exécute d'abord 'python train.py'")
        st.stop()

modele = charger_modele()

# ==================== INITIALISATION DES DONNÉES ====================
if not os.path.exists('data'):
    os.makedirs('data')

fichier_donnees = 'data/patients.csv'

if not os.path.exists(fichier_donnees):
    df_vide = pd.DataFrame(columns=[
        'date', 'age', 'glucose', 'bmi', 'pression', 'risque'
    ])
    df_vide.to_csv(fichier_donnees, index=False)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🩺 Diabète Predict</h1>
    <p>Prédiction du risque de diabète basée sur vos données cliniques</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=80)
    st.markdown("## 📋 Navigation")
    page = st.radio(
        "Choisissez une section",
        ["🎯 Prédiction", "📊 Statistiques", "📈 Graphiques", "📁 Historique"],
        format_func=lambda x: x
    )
    st.markdown("---")
    st.caption("💡 **Astuce** : Chaque prédiction est automatiquement sauvegardée")

# ==================== PAGE 1 : PRÉDICTION ====================
if page == "🎯 Prédiction":
    st.markdown("## 🎯 Évaluation personnalisée")
    st.markdown("*Remplissez le formulaire ci-dessous pour évaluer votre risque de diabète*")
    
    col_form, col_info = st.columns([1, 1])
    
    with col_form:
        with st.container():
            st.markdown("### 📝 Informations")
            
            age = st.slider(
                "📅 Âge (années)",
                min_value=18, max_value=100, value=35,
                help="Votre âge actuel"
            )
            
            glucose = st.slider(
                "🩸 Taux de glucose (mg/dL)",
                min_value=70, max_value=250, value=100,
                help="Glycémie à jeun. Normal : 70-99, Pré-diabète : 100-125, Diabète : >126"
            )
            
            bmi = st.slider(
                "⚖️ IMC (kg/m²)",
                min_value=15.0, max_value=45.0, value=25.0, step=0.5,
                help="IMC = poids(kg) / taille²(m). Normal : 18.5-24.9"
            )
            
            pression = st.slider(
                "❤️ Pression artérielle (mm Hg)",
                min_value=60, max_value=140, value=75,
                help="Tension artérielle. Normale : <80"
            )
            
            st.markdown("---")
            
            if st.button("🔍 PRÉDIRE MON RISQUE", type="primary", use_container_width=True):
                # Préparation des données
                input_data = np.array([[age, glucose, bmi, pression]])
                input_scaled = modele['scaler'].transform(input_data)
                
                # Prédiction
                risque = modele['model'].predict(input_scaled)[0]
                probabilite = modele['model'].predict_proba(input_scaled)[0][1]
                
                # Sauvegarde
                nouveau_patient = pd.DataFrame([{
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'age': age,
                    'glucose': glucose,
                    'bmi': bmi,
                    'pression': pression,
                    'risque': "Élevé" if risque == 1 else "Faible"
                }])
                
                df_existant = pd.read_csv(fichier_donnees)
                df_update = pd.concat([df_existant, nouveau_patient], ignore_index=True)
                df_update.to_csv(fichier_donnees, index=False)
                
                # Stockage dans la session
                st.session_state['resultat'] = {
                    'risque': risque,
                    'probabilite': probabilite,
                    'age': age,
                    'glucose': glucose,
                    'bmi': bmi,
                    'pression': pression
                }
    
    with col_info:
        if 'resultat' in st.session_state:
            res = st.session_state['resultat']
            
            if res['risque'] == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h2>⚠️ RISQUE ÉLEVÉ</h2>
                    <p style="font-size: 2.5rem;">{res['probabilite']:.0%}</p>
                    <p>Probabilité de développer un diabète</p>
                    <hr>
                    <p>📌 Recommandations :<br>
                    - Consultez rapidement un médecin<br>
                    - Surveillez votre alimentation<br>
                    - Faites de l'exercice régulièrement</p>
                </div>
                """, unsafe_allow_html=True)
            elif res['probabilite'] > 0.4:
                st.markdown(f"""
                <div class="result-moderate">
                    <h2>⚠️ RISQUE MODÉRÉ</h2>
                    <p style="font-size: 2rem;">{res['probabilite']:.0%}</p>
                    <p>Probabilité de développer un diabète</p>
                    <hr>
                    <p>📌 Recommandations :<br>
                    - Bilan de santé préventif<br>
                    - Réduisez les sucres rapides<br>
                    - Contrôlez votre poids</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h2>✅ RISQUE FAIBLE</h2>
                    <p style="font-size: 2rem;">{res['probabilite']:.0%}</p>
                    <p>Probabilité de développer un diabète</p>
                    <hr>
                    <p>📌 Recommandations :<br>
                    - Maintenez une bonne hygiène de vie<br>
                    - Contrôle annuel recommandé<br>
                    - Continuez ainsi !</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Jauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['probabilite'] * 100,
                title={"text": "Niveau de risque (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c"},
                    "steps": [
                        {"range": [0, 40], "color": "#2ed573"},
                        {"range": [40, 70], "color": "#ffa502"},
                        {"range": [70, 100], "color": "#ff6b6b"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=50))
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.markdown("""
            <div class="card">
                <h3>📊 À propos de l'application</h3>
                <p>Cette application utilise un modèle d'intelligence artificielle pour évaluer votre risque de diabète.</p>
                <p><strong>Paramètres analysés :</strong></p>
                <ul>
                    <li>Âge</li>
                    <li>Taux de glucose</li>
                    <li>IMC (Indice de Masse Corporelle)</li>
                    <li>Pression artérielle</li>
                </ul>
                <p>Remplissez le formulaire et cliquez sur "Prédire mon risque"</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== PAGE 2 : STATISTIQUES ====================
elif page == "📊 Statistiques":
    st.markdown("## 📊 Statistiques globales")
    
    df = pd.read_csv(fichier_donnees)
    
    if len(df) > 1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Total</h3>
                <p style="font-size: 2rem; margin:0;">{}</p>
                <p>participants</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            risque_eleve = len(df[df['risque'] == 'Élevé'])
            st.markdown("""
            <div class="metric-card">
                <h3>⚠️ À risque</h3>
                <p style="font-size: 2rem; margin:0;">{}</p>
                <p>personnes ({:.0f}%)</p>
            </div>
            """.format(risque_eleve, risque_eleve/len(df)*100), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📅 Âge moyen</h3>
                <p style="font-size: 2rem; margin:0;">{:.0f}</p>
                <p>ans</p>
            </div>
            """.format(df['age'].mean()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>⚖️ IMC moyen</h3>
                <p style="font-size: 2rem; margin:0;">{:.1f}</p>
                <p>kg/m²</p>
            </div>
            """.format(df['bmi'].mean()), unsafe_allow_html=True)
        
        # Graphiques de statistiques
        st.markdown("---")
        st.markdown("### 📈 Analyse détaillée")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            # Moyennes par groupe
            moyennes = df.groupby('risque')[['glucose', 'bmi', 'age']].mean().reset_index()
            fig_moy = px.bar(moyennes, x='risque', y=['glucose', 'bmi', 'age'],
                            title="Moyennes par groupe de risque",
                            barmode='group',
                            labels={'value': 'Valeur moyenne', 'variable': 'Paramètre'})
            st.plotly_chart(fig_moy, use_container_width=True)
        
        with col_s2:
            # Distribution des risques par âge
            df['age_groupe'] = pd.cut(df['age'], bins=[18, 35, 50, 65, 100],
                                      labels=['18-35', '35-50', '50-65', '65+'])
            risque_age = pd.crosstab(df['age_groupe'], df['risque'])
            fig_risk_age = px.bar(risque_age, title="Répartition des risques par tranche d'âge",
                                 labels={'value': 'Nombre', 'age_groupe': 'Tranche d\'âge'})
            st.plotly_chart(fig_risk_age, use_container_width=True)
        
    else:
        st.info("📊 Collectez plus de données pour voir les statistiques (minimum 2 participants)")

# ==================== PAGE 3 : GRAPHIQUES ====================
elif page == "📈 Graphiques":
    st.markdown("## 📈 Distribution des données")
    st.markdown("*Chaque graphique montre la répartition d'une variable différente*")
    
    df = pd.read_csv(fichier_donnees)
    
    if len(df) > 1:
        # Graphique 1 : Âge
        st.markdown("### 📅 Distribution de l'ÂGE")
        fig1 = px.histogram(df, x='age', nbins=15,
                           title="Répartition des âges des participants",
                           labels={'age': 'Âge (années)', 'count': "Nombre"},
                           color_discrete_sequence=['#667eea'])
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        col_age1, col_age2, col_age3 = st.columns(3)
        with col_age1: st.metric("Âge moyen", f"{df['age'].mean():.0f} ans")
        with col_age2: st.metric("Âge minimum", f"{df['age'].min()} ans")
        with col_age3: st.metric("Âge maximum", f"{df['age'].max()} ans")
        
        st.markdown("---")
        
        # Graphique 2 : Glucose
        st.markdown("### 🩸 Distribution du GLUCOSE")
        fig2 = px.histogram(df, x='glucose', nbins=15,
                           title="Répartition des niveaux de glucose",
                           labels={'glucose': 'Glucose (mg/dL)', 'count': "Nombre"},
                           color_discrete_sequence=['#ff6b6b'])
        fig2.add_vline(x=126, line_dash="dash", line_color="red", 
                       annotation_text="Seuil diabète (126)")
        st.plotly_chart(fig2, use_container_width=True)
        
        col_gluc1, col_gluc2, col_gluc3 = st.columns(3)
        with col_gluc1: st.metric("Glucose moyen", f"{df['glucose'].mean():.0f} mg/dL")
        with col_gluc2: st.metric("Normal (<100)", f"{len(df[df['glucose']<100])} pers.")
        with col_gluc3: st.metric("Élevé (>126)", f"{len(df[df['glucose']>126])} pers.")
        
        st.markdown("---")
        
        # Graphique 3 : IMC
        st.markdown("### ⚖️ Distribution de l'IMC")
        fig3 = px.histogram(df, x='bmi', nbins=12,
                           title="Répartition de l'IMC",
                           labels={'bmi': 'IMC (kg/m²)', 'count': "Nombre"},
                           color_discrete_sequence=['#2ed573'])
        fig3.add_vrect(x0=0, x1=18.5, fillcolor="lightblue", opacity=0.3)
        fig3.add_vrect(x0=18.5, x1=25, fillcolor="green", opacity=0.3)
        fig3.add_vrect(x0=25, x1=30, fillcolor="orange", opacity=0.3)
        fig3.add_vrect(x0=30, x1=50, fillcolor="red", opacity=0.3)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("📌 Zones : 🔵 Maigreur | 🟢 Normal | 🟠 Surpoids | 🔴 Obésité")
        
        col_bmi1, col_bmi2, col_bmi3 = st.columns(3)
        with col_bmi1: st.metric("IMC moyen", f"{df['bmi'].mean():.1f}")
        with col_bmi2: st.metric("Normal (18.5-25)", f"{len(df[(df['bmi']>=18.5)&(df['bmi']<25)])} pers.")
        with col_bmi3: st.metric("Surpoids/Obésité", f"{len(df[df['bmi']>=25])} pers.")
        
        st.markdown("---")
        
        # Graphique 4 : Pression
        st.markdown("### ❤️ Distribution de la PRESSION ARTÉRIELLE")
        fig4 = px.histogram(df, x='pression', nbins=12,
                           title="Répartition de la pression artérielle",
                           labels={'pression': 'Pression (mm Hg)', 'count': "Nombre"},
                           color_discrete_sequence=['#ffa502'])
        fig4.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="Normale")
        fig4.add_vline(x=90, line_dash="dash", line_color="red", annotation_text="Élevée")
        st.plotly_chart(fig4, use_container_width=True)
        
        col_press1, col_press2, col_press3 = st.columns(3)
        with col_press1: st.metric("Pression moyenne", f"{df['pression'].mean():.0f} mm Hg")
        with col_press2: st.metric("Normale (<80)", f"{len(df[df['pression']<80])} pers.")
        with col_press3: st.metric("Élevée (>90)", f"{len(df[df['pression']>90])} pers.")
        
        st.markdown("---")
        
        # Graphique 5 : Risque
        st.markdown("### ⚠️ Distribution du RISQUE")
        risque_counts = df['risque'].value_counts().reset_index()
        risque_counts.columns = ['Risque', 'Nombre']
        fig5 = px.bar(risque_counts, x='Risque', y='Nombre',
                     title="Répartition du risque de diabète",
                     labels={'Risque': 'Niveau de risque', 'Nombre': "Nombre"},
                     color='Risque',
                     color_discrete_map={'Faible': '#2ed573', 'Élevé': '#ff6b6b'})
        st.plotly_chart(fig5, use_container_width=True)
        
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1: st.metric("Risque faible", f"{len(df[df['risque']=='Faible'])} personnes")
        with col_risk2: st.metric("Risque élevé", f"{len(df[df['risque']=='Élevé'])} personnes")
        
    else:
        st.info("📊 Aucune donnée suffisante. Faites d'abord des prédictions dans la section 'Prédiction'")

# ==================== PAGE 4 : HISTORIQUE ====================
else:
    st.markdown("## 📋 Historique des données collectées")
    
    df = pd.read_csv(fichier_donnees)
    
    if len(df) > 0:
        # Filtres
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtre_risque = st.selectbox("Filtrer par risque", ["Tous", "Faible", "Élevé"])
        with col_f2:
            tri = st.selectbox("Trier par", ["Date (récent)", "Date (ancien)", "Âge (croissant)", "Âge (décroissant)"])
        
        # Application des filtres
        if filtre_risque != "Tous":
            df_display = df[df['risque'] == filtre_risque]
        else:
            df_display = df.copy()
        
        # Tri
        if tri == "Date (récent)":
            df_display = df_display.sort_values('date', ascending=False)
        elif tri == "Date (ancien)":
            df_display = df_display.sort_values('date', ascending=True)
        elif tri == "Âge (croissant)":
            df_display = df_display.sort_values('age', ascending=True)
        else:
            df_display = df_display.sort_values('age', ascending=False)
        
        st.dataframe(df_display, use_container_width=True)
        
        # Export
        st.markdown("---")
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            csv = df.to_csv(index=False)
            st.download_button("📥 Télécharger toutes les données (CSV)", csv, "donnees_diabete.csv")
        
        with col_exp2:
            if st.button("🗑️ Réinitialiser toutes les données", type="secondary"):
                df_vide = pd.DataFrame(columns=['date', 'age', 'glucose', 'bmi', 'pression', 'risque'])
                df_vide.to_csv(fichier_donnees, index=False)
                st.success("✅ Données réinitialisées !")
                st.rerun()
        

      # Statistiques sur l'historique
        st.markdown("---")
        st.markdown("### 📊 Résumé des données")
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Nombre total", len(df))
        with col_sum2:
            st.metric("Date première", df['date'].min() if len(df) > 0 else "-")
        with col_sum3:
            st.metric("Date dernière", df['date'].max() if len(df) > 0 else "-")
        
    else:
        st.info("📋 Aucune donnée collectée pour le moment")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>🩺 Diabète Predict - Application de prédiction du risque de diabète</p>
    <p style="font-size: 0.8rem;">⚠️ Cette application est un outil d'aide à la décision. Consultez toujours un médecin pour un diagnostic officiel.</p>
</div>
""", unsafe_allow_html=True)