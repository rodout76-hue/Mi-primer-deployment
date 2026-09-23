# ============================================================
# PROYECTO FINAL — CLASIFICACIÓN DE CORREO SPAM
# 1) Crear dataset sintético
# 2) Entrenar modelo
# 3) Guardar modelo
# 4) App Streamlit para predicción
# ============================================================

# ------------------------------------------------------------
# 1. CREAR DATASET SINTÉTICO
# ------------------------------------------------------------
import pandas as pd
import numpy as np

# Creamos un dataset inventado con características típicas de correo
# - num_palabras: número de palabras del correo
# - num_links: número de enlaces
# - num_mayus: número de palabras en MAYÚSCULAS
# - tiene_oferta: 1 si contiene palabras tipo "oferta", "gratis", etc.
# - spam: etiqueta (1 = spam, 0 = no spam)

np.random.seed(42)

n_samples = 500

data = {
    "num_palabras": np.random.randint(20, 300, n_samples),
    "num_links": np.random.randint(0, 10, n_samples),
    "num_mayus": np.random.randint(0, 50, n_samples),
    "tiene_oferta": np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

# Regla inventada para generar la etiqueta spam:
# - muchos links
# - muchas mayúsculas
# - tiene_oferta = 1
spam_score = (
    0.3 * (df["num_links"] > 3).astype(int) +
    0.3 * (df["num_mayus"] > 20).astype(int) +
    0.4 * df["tiene_oferta"]
)

df["spam"] = (spam_score > 0.5).astype(int)

print("Primeras filas del dataset sintético:")
print(df.head())

# ------------------------------------------------------------
# 2. ENTRENAR MODELO DE CLASIFICACIÓN
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

X = df[["num_palabras", "num_links", "num_mayus", "tiene_oferta"]]
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)