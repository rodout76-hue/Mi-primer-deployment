# 📧 Proyecto Final — Deployment ML con Streamlit
## Clasificador de Correo Spam (Dataset Sintético)

Este proyecto implementa un modelo de Machine Learning para clasificar correos electrónicos como **spam** o **no spam**.  
El dataset ha sido **creado de forma sintética** para fines educativos, simulando características típicas de correos reales.

La aplicación está desplegada con **Streamlit**, permitiendo al usuario introducir características de un correo y obtener una predicción en tiempo real.

---

## 🧩 Dataset Sintético

El dataset contiene 500 muestras generadas artificialmente con las siguientes variables:

- `num_palabras`: número total de palabras del correo  
- `num_links`: cantidad de enlaces presentes  
- `num_mayus`: número de palabras en MAYÚSCULAS  
- `tiene_oferta`: indicador (0/1) de si contiene palabras como "oferta", "gratis", "promoción"  
- `spam`: etiqueta objetivo generada mediante una regla lógica

---

## 🤖 Modelo de Machine Learning

Se utiliza un **RandomForestClassifier** de scikit-learn:

- Entrenado sobre el dataset sintético  
- Evaluado con un split 80/20  
- Guardado como `model_spam.pkl` para su uso en la app Streamlit

---

## 🖥️ Aplicación Streamlit

La app permite:

- Introducir manualmente las características del correo
- Ejecutar el modelo entrenado
- Mostrar si el correo es **spam** o **no spam**
- Mostrar la probabilidad asociada a la predicción

Archivo principal: `app.py`

---

## 📦 Estructura del proyecto


---

## 🚀 Despliegue en Streamlit Cloud

1. Subir el repositorio a GitHub  
2. Ir a https://streamlit.io/cloud  
3. Seleccionar el repositorio  
4. Elegir `app.py` como archivo principal  
5. Streamlit instalará automáticamente las dependencias  
6. La app quedará disponible en una URL pública

---

## 👩‍💻 Autor

Proyecto realizado por **Ángela**  
Máster Data Science — Nodd3r

