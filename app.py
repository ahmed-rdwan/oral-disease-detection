"""
Oral Disease Classifier - Streamlit demo app.

"""

import json

import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model.keras"
METADATA_PATH = "metadata.json"

st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷")


@st.cache_resource
def load_model_and_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = keras.models.load_model(MODEL_PATH)

    preprocessing = metadata.get("preprocessing", "none")
    if preprocessing == "efficientnet":
        from keras.applications.efficientnet import preprocess_input
    elif preprocessing == "resnet50":
        from keras.applications.resnet50 import preprocess_input
    elif preprocessing == "mobilenet_v2":
        from keras.applications.mobilenet_v2 import preprocess_input
    else:
        def preprocess_input(x):
            return x

    return model, metadata, preprocess_input


model, metadata, preprocess_input = load_model_and_metadata()
IMAGE_SIZE = tuple(metadata["image_size"])
CLASS_NAMES = metadata["class_names"]

st.title("🦷 Oral Disease Classifier")
st.caption(
    f"CNN model ({metadata.get('model_name', 'custom')}) trained to classify "
    f"{len(CLASS_NAMES)} oral conditions: {', '.join(CLASS_NAMES)}. "
    f"Validation accuracy: {metadata.get('val_accuracy', 0):.2%} | "
    f"F1 (macro): {metadata.get('f1_macro', 0):.2f}"
)
st.warning(
    "⚠️ Educational / portfolio demo only. This model is NOT a medical "
    "device and must not be used for real diagnosis. Always consult a "
    "dentist or doctor for any oral health concern."
)

uploaded_file = st.file_uploader(
    "Upload a photo of the oral condition", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        img = tf.image.resize(np.array(image), IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img.numpy())
        img = tf.expand_dims(img, axis=0)

        probs = model.predict(img, verbose=0)[0]
        ranked = sorted(
            zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True
        )

    top_class, top_prob = ranked[0]
    st.subheader(f"Prediction: {top_class} ({top_prob:.1%} confidence)")

    st.write("All class probabilities:")
    for cls, prob in ranked:
        st.progress(float(prob), text=f"{cls}: {prob:.1%}")
