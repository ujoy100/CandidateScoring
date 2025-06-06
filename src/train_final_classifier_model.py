import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("data/final_labeled_dataset.csv")

# Sentence transformer for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Feature engineering
def build_features(row):
    cand_emb = embedder.encode(row["candidate_text"])
    job_emb = embedder.encode(row["job_text"])
    similarity = cosine_similarity([cand_emb], [job_emb])[0][0]
    return [similarity, row["years_experience"]]

X = np.array([build_features(row) for _, row in df.iterrows()])
y = df["match_score"].astype(int).values

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, test_idx in split.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Train classifier
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save model
joblib.dump(model, "src/model.joblib")
print("✅ Classifier trained and saved to src/model.joblib")
