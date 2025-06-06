
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model and embedder
model = joblib.load("src/model.joblib")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load test cases
df = pd.read_csv("data/test_cases.csv")

# Prepare results
results = []

for _, row in df.iterrows():
    cand_text = row["candidate_text"]
    job_text = row["job_text"]
    years_exp = row["years_experience"]

    cand_emb = embedder.encode(cand_text)
    job_emb = embedder.encode(job_text)
    similarity = cosine_similarity([cand_emb], [job_emb])[0][0]
    features = [similarity, years_exp]

    proba = model.predict_proba([features])[0][1]
    score = round(proba, 2)

    if score >= 0.75:
        label = "🟢 Strong Match"
    elif score >= 0.5:
        label = "🟡 Medium Match"
    else:
        label = "🔴 Weak Match"

    results.append({
        "candidate_text": cand_text,
        "job_text": job_text,
        "years_experience": years_exp,
        "match_score": score,
        "match_label": label
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("data/batch_scoring_results.csv", index=False)
print("✅ Scoring complete. Results saved to data/batch_scoring_results.csv")
