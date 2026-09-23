
# ============================================================
# 4. APP STREAMLIT — CLASIFICACIÓN DE CORREO SPAM
# Guarda este bloque en un archivo separado llamado app.py
# ============================================================

import streamlit as st
import numpy as np
import joblib

# ------------------------------------------------------------
# Cargar modelo
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model_spam.pkl")
    return model

model = load_model()

# ------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Correo Spam",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Clasificación de Correo Spam")
st.write("Introduce las características del correo y el modelo te dirá si es spam o no.")

# ------------------------------------------------------------
# Inputs del usuario
# ------------------------------------------------------------
st.header("📥 Características del correo")

num_palabras = st.number_input("Número de palabras", 0, 1000, 120)
num_links = st.number_input("Número de enlaces", 0, 50, 2)
num_mayus = st.number_input("Número de palabras en MAYÚSCULAS", 0, 200, 10)
tiene_oferta = st.selectbox("¿Contiene palabras tipo 'oferta', 'gratis', 'promoción'?", ["No", "Sí"])

tiene_oferta_bin = 1 if tiene_oferta == "Sí" else 0

# ------------------------------------------------------------
# Botón de predicción
# ------------------------------------------------------------
if st.button("🔍 Clasificar correo"):
    input_data = np.array([
        num_palabras,
        num_links,
        num_mayus,
        tiene_oferta_bin
    ]).reshape(1, -1)

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][pred]

    if pred == 1:
        st.error(f"⚠️ El modelo predice: **SPAM** (probabilidad {prob:.2f})")
    else:
        st.success(f"✅ El modelo predice: **NO SPAM** (probabilidad {prob:.2f})")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.write("---")
st.write("Proyecto final · Clasificación de correo spam · Ángela")
