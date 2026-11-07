
import streamlit as st
import numpy as np

from src.data import load_raw_data, preprocess
from src.predict import load_model, predict


@st.cache_resource
def load_resources():
    X, y = load_raw_data()
    _, _, _, _, scaler = preprocess(X, y)
    model = load_model()
    return model, scaler


st.set_page_config(
    page_title="Mi Inteligencia Artificial",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Mi Inteligencia Artificial")
st.write("Introducí dos números y mi IA los clasificará.")

try:
    model, scaler = load_resources()

    st.subheader("Datos de entrada")

    numero1 = st.number_input("Primer número", value=0.0)
    numero2 = st.number_input("Segundo número", value=0.5)

    if st.button("Analizar con IA"):
        X = np.array([[numero1, numero2]], dtype=np.float32)
        clases, probabilidades = predict(model, X, scaler)

        st.success(f"Resultado: Clase {int(clases[0])}")
        st.metric(
            "Confianza del modelo",
            f"{float(probabilidades[0].max()) * 100:.2f}%"
        )

        st.write("Probabilidades por clase:")
        st.write({
            "Clase 0": f"{probabilidades[0][0] * 100:.2f}%",
            "Clase 1": f"{probabilidades[0][1] * 100:.2f}%"
        })

except Exception as error:
    st.error("No se pudo cargar el modelo.")
    st.error(str(error))
