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

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Animated background ── */
.stApp {
    background: #0a0a1a;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(120, 40, 200, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(0, 180, 255, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(236, 72, 153, 0.06) 0%, transparent 50%);
    animation: auroraMove 20s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
}
@keyframes auroraMove {
    0%   { transform: translate(0, 0) rotate(0deg); }
    25%  { transform: translate(-3%, 2%) rotate(1deg); }
    50%  { transform: translate(2%, -2%) rotate(-0.5deg); }
    75%  { transform: translate(-1%, 3%) rotate(0.5deg); }
    100% { transform: translate(3%, -1%) rotate(-1deg); }
}

/* ── Floating particles effect using pseudo-elements ── */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(2px 2px at 10% 20%, rgba(168,85,247,0.3) 50%, transparent 50%),
        radial-gradient(2px 2px at 30% 70%, rgba(0,210,255,0.2) 50%, transparent 50%),
        radial-gradient(1.5px 1.5px at 60% 30%, rgba(236,72,153,0.25) 50%, transparent 50%),
        radial-gradient(2px 2px at 80% 60%, rgba(168,85,247,0.2) 50%, transparent 50%),
        radial-gradient(1.5px 1.5px at 45% 90%, rgba(0,210,255,0.3) 50%, transparent 50%),
        radial-gradient(2px 2px at 90% 10%, rgba(236,72,153,0.15) 50%, transparent 50%);
    animation: particleFloat 30s linear infinite;
    z-index: 0;
    pointer-events: none;
}
@keyframes particleFloat {
    0%   { transform: translateY(0); }
    100% { transform: translateY(-100vh); }
}

/* ── Ensure content is above background ── */
.block-container {
    position: relative;
    z-index: 1;
}

/* ── Hero title ── */
.hero-container {
    text-align: center;
    padding: 2rem 0 0.5rem;
    animation: fadeInDown 0.8s ease;
}
.hero-icon {
    font-size: 3.5rem;
    display: inline-block;
    animation: toothBounce 2s ease-in-out infinite;
    filter: drop-shadow(0 0 20px rgba(168,85,247,0.4));
}
@keyframes toothBounce {
    0%, 100% { transform: translateY(0) scale(1); }
    50%      { transform: translateY(-8px) scale(1.05); }
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d2ff 0%, #a855f7 40%, #ec4899 70%, #f97316 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientText 4s ease infinite;
    margin: 0.5rem 0 0.3rem;
    letter-spacing: -1px;
}
@keyframes gradientText {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-sub {
    color: rgba(255,255,255,0.45);
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* ── Glowing divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.5), rgba(0,210,255,0.3), transparent);
    border: none;
    margin: 1.5rem 0 2rem;
    position: relative;
}
.glow-divider::after {
    content: '';
    position: absolute;
    top: -2px;
    left: 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.15), transparent);
    filter: blur(3px);
}




/* ── Style Streamlit uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] > label {
    display: none !important;
}
[data-testid="stFileUploader"] > div {
    padding-top: 0 !important;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(135deg, rgba(168,85,247,0.06), rgba(0,210,255,0.04)) !important;
    border: 2px dashed rgba(168,85,247,0.25) !important;
    border-radius: 24px !important;
    padding: 2.5rem 2rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(168,85,247,0.50) !important;
    background: rgba(168,85,247,0.10) !important;
    box-shadow: 0 8px 40px rgba(168,85,247,0.08) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] span {
    color: rgba(255,255,255,0.55) !important;
    font-weight: 500 !important;
}
/* Fix double-text button: hide original text, replace with clean label */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, rgba(168,85,247,0.25), rgba(0,210,255,0.20)) !important;
    border: 1px solid rgba(168,85,247,0.30) !important;
    color: transparent !important;
    border-radius: 12px !important;
    padding: 0.5rem 1.5rem !important;
    font-size: 0 !important;
    transition: all 0.3s ease !important;
    position: relative;
    min-width: 140px;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: '📁 Browse Files';
    color: rgba(255,255,255,0.85);
    font-size: 0.9rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.3px;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, rgba(168,85,247,0.40), rgba(0,210,255,0.30)) !important;
    box-shadow: 0 4px 20px rgba(168,85,247,0.20) !important;
}
/* Style "Drag and drop" text */
[data-testid="stFileUploaderDropzone"] small {
    color: rgba(255,255,255,0.35) !important;
}
/* file info after upload */
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
    color: rgba(255,255,255,0.50) !important;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 1.8rem;
    margin: 1.2rem 0;
    animation: fadeIn 0.8s ease;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.3), transparent);
}
.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168,85,247,0.15);
    box-shadow: 0 20px 60px rgba(168, 85, 247, 0.10), 0 0 40px rgba(0, 210, 255, 0.05);
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 15px 50px rgba(0,0,0,0.35);
    transition: transform 0.4s ease;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.01);
}

/* ── Result badge ── */
.result-container {
    animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55);
}
.result-badge {
    text-align: center;
    padding: 2rem 1.5rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(168,85,247,0.12), rgba(0,210,255,0.08));
    border: 1px solid rgba(168,85,247,0.15);
    position: relative;
    overflow: hidden;
}
.result-badge::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(transparent, rgba(168,85,247,0.05), transparent 30%);
    animation: rotateConic 8s linear infinite;
}
@keyframes rotateConic {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.result-label {
    position: relative;
    color: rgba(255,255,255,0.45);
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
.result-class {
    position: relative;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f0ff;
    margin: 0.3rem 0;
    text-shadow: 0 0 30px rgba(168,85,247,0.3);
}
.result-confidence {
    position: relative;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d2ff, #a855f7, #ec4899);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientText 3s ease infinite;
    line-height: 1.2;
}
.result-conf-label {
    position: relative;
    color: rgba(255,255,255,0.30);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 3px;
}

/* ── Probability bars ── */
.probs-title {
    color: rgba(255,255,255,0.55);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
}
.prob-row {
    display: flex;
    align-items: center;
    margin: 0.65rem 0;
    gap: 0.8rem;
}
.prob-label {
    color: rgba(255,255,255,0.75);
    font-size: 0.83rem;
    font-weight: 500;
    min-width: 135px;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #a855f7, #ec4899, #f97316);
    background-size: 200% 100%;
    animation: barGradient 3s ease infinite;
    transition: width 1.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    box-shadow: 0 0 10px rgba(168,85,247,0.3);
}
@keyframes barGradient {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
.prob-bar-fill.top {
    background: linear-gradient(90deg, #00d2ff, #a855f7);
    box-shadow: 0 0 15px rgba(0,210,255,0.3);
}
.prob-value {
    color: rgba(255,255,255,0.50);
    font-size: 0.8rem;
    font-weight: 600;
    min-width: 48px;
    text-align: right;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Bottom warning ── */
.bottom-warning {
    text-align: center;
    color: rgba(255,255,255,0.25);
    font-size: 0.72rem;
    padding: 2rem 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    letter-spacing: 0.3px;
    line-height: 1.6;
}

/* ── Animations ── */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInDown { from { opacity: 0; transform: translateY(-25px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn {
    0%   { opacity: 0; transform: scale(0.7) translateY(20px); }
    60%  { transform: scale(1.03) translateY(-3px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}
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
st.markdown(
    """
    <div class="hero-container">
        <span class="hero-icon">🦷</span>
        <div class="hero-title">Oral Disease Classifier</div>
        <div class="hero-sub">AI-powered detection for common oral conditions</div>
    </div>
    <div class="glow-divider"></div>
    """,
    unsafe_allow_html=True,
)

# ── Upload Section ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("🔬 Analyzing..."):
        img = tf.image.resize(np.array(image), IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img.numpy())
        img = tf.expand_dims(img, axis=0)

        probs = model.predict(img, verbose=0)[0]
        ranked = sorted(
            zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True
        )

    top_class, top_prob = ranked[0]

    # ── Result badge ──
    st.markdown(
        f"""
        <div class="result-container">
            <div class="result-badge">
                <div class="result-label">Detected Condition</div>
                <div class="result-class">{top_class.replace("_", " ")}</div>
                <div class="result-confidence">{top_prob:.1%}</div>
                <div class="result-conf-label">confidence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Probabilities ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="probs-title">📊 All Predictions</div>', unsafe_allow_html=True)

    bars_html = ""
    for i, (cls, prob) in enumerate(ranked):
        width = max(prob * 100, 0.5)
        fill_class = "prob-bar-fill top" if i == 0 else "prob-bar-fill"
        bars_html += f"""
        <div class="prob-row">
            <span class="prob-label">{cls.replace("_", " ")}</span>
            <div class="prob-bar-bg">
                <div class="{fill_class}" style="width: {width:.1f}%;"></div>
            </div>
            <span class="prob-value">{prob:.1%}</span>
        </div>
        """

    st.markdown(bars_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Bottom warning ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="bottom-warning">
        ⚠️ Educational / portfolio demo only · Not a medical device ·
        Always consult a dentist or doctor for any oral health concern
    </div>
    """,
    unsafe_allow_html=True,
)
