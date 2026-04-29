import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("=" * 50)
print("🩺 ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION DU DIABÈTE")
print("=" * 50)

# Création des données d'entraînement (dataset élargi pour meilleure précision)
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 28, 33, 38, 43, 48, 53, 58, 63, 26, 31, 36, 41, 46, 51, 56, 61],
    'glucose': [85, 90, 95, 110, 125, 140, 155, 170, 180, 88, 92, 105, 115, 130, 145, 160, 175, 82, 95, 108, 120, 135, 150, 165, 178],
    'bmi': [22, 23, 24, 26, 28, 30, 32, 34, 35, 21, 23, 25, 27, 29, 31, 33, 34, 22, 24, 26, 28, 30, 32, 33, 35],
    'pression': [70, 72, 74, 76, 78, 80, 82, 85, 88, 68, 71, 73, 75, 77, 79, 81, 84, 69, 72, 74, 76, 78, 80, 83, 86],
    'diabete': [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
})

print(f"\n📊 Dataset chargé : {len(data)} patients")
print(data.head())

# Séparation des features (X) et target (y)
X = data[['age', 'glucose', 'bmi', 'pression']]
y = data['diabete']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraînement du modèle Random Forest
print("\n🔄 Entraînement du modèle en cours...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_scaled, y)

# Évaluation de la précision
train_score = model.score(X_scaled, y)
print(f"\n✅ Précision du modèle : {train_score:.2%}")

# Sauvegarde du modèle et du scaler
with open('modele_diabete.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("\n💾 Modèle sauvegardé dans 'modele_diabete.pkl'")
print("\n" + "=" * 50)
print("🎉 ENTRAÎNEMENT TERMINÉ !")
print("📌 Lance l'application avec : streamlit run app.py")
print("=" * 50)