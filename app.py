"""
Oral Disease Classifier - Streamlit demo app.

"""

import json
import os
import subprocess

import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model.keras"
METADATA_PATH = "metadata.json"


# --- Git LFS: ensure the real model file is present, not just a pointer ---
def _ensure_lfs():
    """Pull Git-LFS files if the model is still a pointer."""
    if not os.path.exists(MODEL_PATH):
        return
    with open(MODEL_PATH, "rb") as f:
        header = f.read(50)
    if header.startswith(b"version https://git-lfs"):
        st.info("⏳ Downloading model from Git LFS …")
        subprocess.run(["git", "lfs", "install"], cwd=os.getcwd(), check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=os.getcwd(), check=True)
        st.info("✅ Model downloaded!")


_ensure_lfs()

st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="🦷",
    layout="centered",
)

# ── Custom CSS for premium look ──────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit branding ── */
#MainMenu, footer, header {visibility: hidden;}

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hero title ── */
.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d2ff 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    animation: fadeInDown 1s ease;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 2rem;
    animation: fadeInUp 1.2s ease;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin: 1.5rem 0;
    animation: fadeIn 1s ease;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(168, 85, 247, 0.15);
}

/* ── Result badge ── */
.result-badge {
    text-align: center;
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(168,85,247,0.20), rgba(236,72,153,0.20));
    border: 1px solid rgba(168,85,247,0.25);
    margin: 1rem 0;
    animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55);
}
.result-badge h2 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e0e0ff;
}
.result-badge .confidence {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d2ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    margin: 0.5rem 0;
    gap: 0.7rem;
}
.prob-label {
    color: rgba(255,255,255,0.80);
    font-size: 0.85rem;
    font-weight: 500;
    min-width: 140px;
}
.prob-bar-bg {
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #a855f7, #ec4899);
    transition: width 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.prob-value {
    color: rgba(255,255,255,0.60);
    font-size: 0.8rem;
    font-weight: 500;
    min-width: 50px;
    text-align: right;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(168,85,247,0.30);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(168,85,247,0.60);
}

/* ── Image styling ── */
[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* ── Warning at bottom ── */
.bottom-warning {
    text-align: center;
    color: rgba(255,255,255,0.35);
    font-size: 0.78rem;
    padding: 1.5rem 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 3rem;
    animation: fadeIn 2s ease;
}
.bottom-warning span {
    color: rgba(255, 200, 50, 0.50);
}

/* ── Divider ── */
.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.40), transparent);
    border: none;
    margin: 1.5rem 0;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: #a855f7 !important;
}

/* ── Animations ── */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn { 0% { opacity: 0; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1); } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Resolve the preprocessing function BEFORE loading the model.
    # The model contains a Lambda layer that wraps this function,
    # so Keras needs it registered as a custom object to deserialize.
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

    model = keras.models.load_model(
        MODEL_PATH,
        safe_mode=False,
        compile=False,
        custom_objects={"preprocess_input": preprocess_input},
    )

    return model, metadata, preprocess_input


model, metadata, preprocess_input = load_model_and_metadata()
IMAGE_SIZE = tuple(metadata["image_size"])
CLASS_NAMES = metadata["class_names"]

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🦷 Oral Disease Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered detection for common oral conditions · Upload a photo to get started</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── Upload Section ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📸  Upload a photo of the oral condition",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    # Show instructions when no file uploaded
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <p style="font-size:2.5rem; margin-bottom:0.5rem;">📤</p>
            <p style="color:rgba(255,255,255,0.7); font-size:1rem; font-weight:500; margin:0;">
                Drag & drop or click to upload an oral image
            </p>
            <p style="color:rgba(255,255,255,0.35); font-size:0.85rem; margin-top:0.4rem;">
                Supported: JPG, JPEG, PNG
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    image = Image.open(uploaded_file).convert("RGB")

    # Display image inside a glass card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("🔬 Analyzing image..."):
        img = tf.image.resize(np.array(image), IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img.numpy())
        img = tf.expand_dims(img, axis=0)

        probs = model.predict(img, verbose=0)[0]
        ranked = sorted(
            zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True
        )

    top_class, top_prob = ranked[0]

    # Result badge
    st.markdown(
        f"""
        <div class="result-badge">
            <p style="color:rgba(255,255,255,0.50); font-size:0.85rem; margin:0 0 0.3rem;">Detected Condition</p>
            <h2>{top_class.replace("_", " ")}</h2>
            <p class="confidence">{top_prob:.1%}</p>
            <p style="color:rgba(255,255,255,0.40); font-size:0.8rem; margin:0;">confidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # All probabilities
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:rgba(255,255,255,0.65); font-size:0.9rem; font-weight:600; margin-bottom:0.8rem;">📊 All Predictions</p>',
        unsafe_allow_html=True,
    )

    bars_html = ""
    for cls, prob in ranked:
        width = max(prob * 100, 1)
        bars_html += f"""
        <div class="prob-row">
            <span class="prob-label">{cls.replace("_", " ")}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: {width:.1f}%;"></div>
            </div>
            <span class="prob-value">{prob:.1%}</span>
        </div>
        """

    st.markdown(bars_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Warning at the very bottom ───────────────────────────────────────────────
st.markdown(
    """
    <div class="bottom-warning">
        <span>⚠️</span> Educational / portfolio demo only. This model is NOT a medical device
        and must not be used for real diagnosis. Always consult a dentist or doctor
        for any oral health concern.
    </div>
    """,
    unsafe_allow_html=True,
)
