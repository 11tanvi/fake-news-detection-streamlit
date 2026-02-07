import streamlit as st
import joblib
import numpy as np
import pandas as pd
from src.utils import clean_text


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Smart Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# -------------------- ADVANCED CSS --------------------
st.markdown("""
<style>

/* ====== BACKGROUND ====== */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ====== GLASS CARD ====== */
.glass {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}

/* ====== TEXT ====== */
.title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #dcdcdc;
    margin-bottom: 40px;
}

.section {
    color: #ffffff;
}

.badge-fake {
    background: #ff4b4b;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
}

.badge-real {
    background: #00c896;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #cccccc;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    liar_model = joblib.load("models/liar_model.pkl")
    liar_vec = joblib.load("models/liar_vectorizer.pkl")
    fn_model = joblib.load("models/fakenewsnet_model.pkl")
    fn_vec = joblib.load("models/fakenewsnet_vectorizer.pkl")
    return liar_model, liar_vec, fn_model, fn_vec


liar_model, liar_vec, fn_model, fn_vec = load_models()


# -------------------- HERO --------------------
st.markdown("<div class='title'>📰 Smart Fake News Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>"
    "AI-powered misinformation risk analysis using trusted datasets"
    "</div>",
    unsafe_allow_html=True
)


# -------------------- MAIN LAYOUT --------------------
left, right = st.columns([2, 1])


# -------------------- LEFT: TEXT --------------------
with left:
    st.markdown("<div class='glass section'>", unsafe_allow_html=True)
    st.subheader("📝 News Content")

    news_text = st.text_area(
        "",
        height=240,
        placeholder="Paste the full news article, headline, or claim here..."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- RIGHT: CONTROLS --------------------
with right:
    st.markdown("<div class='glass section'>", unsafe_allow_html=True)
    st.subheader("⚙️ Analysis Settings")

    dataset_type = st.radio(
        "Select news category",
        [
            "Political / Public Statements (LIAR)",
            "Social Media / Viral News (FakeNewsNet)",
            "Analyze with Both Models"
        ]
    )

    analyze = st.button("🚀 Analyze News", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- ANALYSIS --------------------
if analyze:

    if not news_text.strip():
        st.warning("⚠️ Please enter news text.")
        st.stop()

    cleaned = clean_text(news_text)
    results = []

    with st.spinner("🔍 Analyzing credibility patterns..."):

        if dataset_type in ["Political / Public Statements (LIAR)", "Analyze with Both Models"]:
            X = liar_vec.transform([cleaned])
            probs = liar_model.predict_proba(X)[0]
            fake_prob = np.max(probs) * 100
            verdict = "FAKE" if fake_prob > 70 else "LIKELY REAL"
            results.append(("LIAR Dataset", verdict, round(fake_prob, 2)))

        if dataset_type in ["Social Media / Viral News (FakeNewsNet)", "Analyze with Both Models"]:
            X = fn_vec.transform([cleaned])
            probs = fn_model.predict_proba(X)[0]
            fake_prob = probs[0] * 100
            verdict = "FAKE" if fake_prob > 70 else "LIKELY REAL"
            results.append(("FakeNewsNet Dataset", verdict, round(fake_prob, 2)))


    st.markdown("## 📊 Model Outputs")

    for model, verdict, prob in results:
        badge = "badge-fake" if verdict == "FAKE" else "badge-real"

        st.markdown(f"""
        <div class="glass">
            <h4 style="color:white;">{model}</h4>
            <span class="{badge}">{verdict}</span>
            <p style="color:#eeeeee;margin-top:10px;">
            Fake Probability: <b>{prob}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(prob) / 100)


    # -------------------- FINAL VERDICT --------------------
    fake_scores = [r[2] for r in results if r[1] == "FAKE"]
    avg_risk = np.mean(fake_scores) if fake_scores else 0

    st.markdown("## 🧠 Final Risk Assessment")

    if avg_risk > 80:
        st.error("🚨 Very High Risk of Fake News")
    elif avg_risk > 60:
        st.warning("⚠️ Moderate Risk — Verification Recommended")
    else:
        st.success("✅ Low Risk — Likely Legitimate News")

    st.caption(
        "This tool estimates misinformation risk using learned patterns. "
        "It does not guarantee factual correctness."
    )


