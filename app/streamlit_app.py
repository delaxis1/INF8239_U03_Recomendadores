"""Interfaz local: contenido y perfiles de usuario, con fuente y límites visibles."""

import joblib
import streamlit as st
from inf8239_u03.config import ROOT, MOVIELENS_DIR
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import ContentRecommender

st.set_page_config(page_title="Recomendador - Alexis De La Cruz", layout="wide")
st.title("Motor recomendador de películas")
st.caption("INF-8239 - Ejercicio 05 - Alexis De La Cruz (100805370)")
st.info(
    "Fuente: MovieLens / GroupLens. Datos de investigación. Una puntuación no garantiza satisfacción; los usuarios nuevos reciben popularidad no personalizada."
)


@st.cache_resource
def load_content():
    ratings, movies = load_movielens(MOVIELENS_DIR)
    return movies, ContentRecommender().fit(movies)


try:
    movies, content = load_content()
except FileNotFoundError:
    st.error(
        "Ejecute uv run python scripts/run_exercise05.py para descargar los datos."
    )
    st.stop()
mode = st.radio("Método", ["Contenido de una película", "Perfil de usuario"])
if mode == "Contenido de una película":
    title = st.selectbox("Referencia", sorted(movies.title))
    if st.button("Recomendar"):
        st.dataframe(content.recommend(title), hide_index=True)
        st.caption(
            "TF-IDF de géneros y coseno. Géneros iguales pueden empatar y crear sobre-especialización."
        )
else:
    path = ROOT / "models/serving.joblib"
    if not path.exists():
        st.warning("Complete el experimento para crear el modelo local.")
        st.stop()

    @st.cache_resource
    def serving():
        return joblib.load(path)

    bundle = serving()
    scorer = bundle["scorer"]
    uid = int(
        st.number_input(
            "Usuario; use 999999 para probar cold start", min_value=1, value=1, step=1
        )
    )
    if st.button("Mostrar Top-10"):
        rec = scorer.recommend(uid, bundle["choice"], bundle["mf"])
        table = movies.set_index("movieId").loc[rec].reset_index()
        st.dataframe(table, hide_index=True)
        st.caption("Método seleccionado en validación: " + bundle["choice"])
        if uid not in scorer.seen:
            st.warning(
                "Usuario nuevo: "
                + (
                    "conteo de popularidad"
                    if bundle["choice"] == "count"
                    else "popularidad suavizada"
                )
                + ", sin personalización."
            )
        else:
            st.caption(
                "Se excluyen "
                + str(len(scorer.seen[uid]))
                + " películas del historial de entrenamiento."
            )
