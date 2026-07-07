# -*- coding: utf-8 -*-
"""
Application Streamlit — Classification d'images satellites

Charge le modèle VGG16 fine-tuné (model.json + best_model.h5) et permet
de classer une image satellite parmi 30 catégories.

Lancement :
    streamlit run streamlit_sat.py
"""

import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import model_from_json

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Classification d'images satellites",
    page_icon="🛰️",
    layout="centered",
)

MODEL_DIR = "models"
MODEL_ARCHITECTURE = os.path.join(MODEL_DIR, "model.json")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "best_model.h5")
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center",
    "Church", "Commercial", "DenseResidential", "Desert", "Farmland",
    "Forest", "Industrial", "Meadow", "MediumResidential", "Mountain",
    "Park", "Parking", "Playground", "Pond", "Port", "RailwayStation",
    "Resort", "River", "School", "SparseResidential", "Square", "Stadium",
    "StorageTanks", "Viaduct",
]


# ----------------------------------------------------------------------------
# Chargement du modèle (mis en cache pour ne pas recharger à chaque interaction)
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_ARCHITECTURE) or not os.path.exists(MODEL_WEIGHTS):
        return None
    with open(MODEL_ARCHITECTURE, "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(MODEL_WEIGHTS)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convertit une image PIL en tenseur prêt pour le modèle."""
    array = np.asarray(image.convert("RGB")).astype("float32")
    array = cv2.resize(array, IMAGE_SIZE)
    array /= 255.0
    return array.reshape(1, *IMAGE_SIZE, 3)


def predict(model, image: Image.Image):
    tensor = preprocess_image(image)
    probabilities = model.predict(tensor, verbose=0)[0]
    top5_idx = np.argsort(probabilities)[::-1][:5]
    return [(CLASS_NAMES[i], float(probabilities[i])) for i in top5_idx]


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------

st.title("🛰️ Classification d'images satellites")
st.caption(
    "Modèle VGG16 fine-tuné, entraîné sur 30 classes de paysages "
    "(aéroport, plage, pont, zone résidentielle, forêt, stade, etc.)"
)

with st.sidebar:
    st.header("À propos")
    st.write(
        "Cette application charge un modèle VGG16 pré-entraîné puis "
        "fine-tuné sur un jeu de données d'images satellites haute "
        "résolution, et classe l'image importée parmi 30 catégories."
    )
    st.write("**Classes reconnues :**")
    st.write(", ".join(CLASS_NAMES))

model = load_model()

if model is None:
    st.error(
        f"Modèle introuvable. Assurez-vous que `{MODEL_ARCHITECTURE}` et "
        f"`{MODEL_WEIGHTS}` existent (voir le README pour le lien de "
        "téléchargement de `best_model.h5`)."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Importer une image satellite à classer", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Veuillez importer une image pour lancer la classification.")
else:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Image importée", use_container_width=True)

    with st.spinner("Classification en cours..."):
        results = predict(model, image)

    best_class, best_score = results[0]

    with col2:
        st.subheader("Résultat")
        st.metric(label="Classe prédite", value=best_class)
        st.progress(best_score)
        st.write(f"Confiance : **{best_score * 100:.1f}%**")

        st.subheader("Top 5 des prédictions")
        for class_name, score in results:
            st.write(f"{class_name} — {score * 100:.1f}%")
            st.progress(score)
