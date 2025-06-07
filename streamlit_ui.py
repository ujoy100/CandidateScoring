
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model and embedder
model = joblib.load("src/model.joblib")
embedder = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")

# Streamlit UI
st.set_page_config(page_title="Candidate Match Scoring", layout="centered")
st.title("🔍 Candidate Match Scoring")
st.markdown("Paste a candidate summary and job description to predict match strength.")

# Inputs
candidate_text = st.text_area("📄 Candidate Resume Summary", height=150)
job_text = st.text_area("📝 Job Description", height=150)
years_experience = st.slider("🔧 Years of Experience", 0, 20, 2)

if st.button("Predict Match Score"):
    if candidate_text.strip() == "" or job_text.strip() == "":
        st.warning("Please enter both candidate and job text.")
    else:
        # Feature computation
        cand_emb = embedder.encode(candidate_text)
        job_emb = embedder.encode(job_text)
        similarity = cosine_similarity([cand_emb], [job_emb])[0][0]
        features = [similarity, years_experience]

        # Get prediction probability for class 1
        proba = model.predict_proba([features])[0][1]
        score = round(proba, 2)

        # Label match category
        if score >= 0.75:
            verdict = "🟢 **Strong Match**"
        elif score >= 0.5:
            verdict = "🟡 **Medium Match**"
        else:
            verdict = "🔴 **Weak Match**"

        # Output
        st.success(f"Predicted Match Score: **{score}**")
        st.markdown(verdict)
