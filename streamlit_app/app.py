import streamlit as st
import requests

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

st.markdown(
    """
    <h1 style='text-align: center;'>📰 Fake News Detector</h1>
    <p style='text-align: center; font-size:18px;'>
    Analyze news claims using a transformer-based AI model.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

news_text = st.text_area(
    "Paste a news claim or statement below",
    height=220,
    placeholder="Example: The government secretly controls the weather..."
)

analyze = st.button("Analyze News", use_container_width=True)

if analyze:
    if not news_text.strip():
        st.warning("Enter a news statement first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json={"text": news_text})
                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.divider()

                if prediction == "FAKE":
                    st.error("🚨 This statement looks FAKE")
                else:
                    st.success("✅ This statement looks REAL")

                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.progress(confidence)

            except Exception:
                st.error("API is not running. Start FastAPI first.")

st.divider()

st.caption("Built with RoBERTa, FastAPI, Streamlit, MLflow, and Docker.")