# 🎯 Candidate Scoring with Machine Learning

This project is an AI-powered candidate-job matching system. It evaluates how well a candidate's resume aligns with a job description using a trained ML model built on semantic embeddings and experience.

## 🚀 Features

- ✅ Streamlit UI for real-time scoring
- ✅ XGBoost classifier trained on real-world text pairs
- ✅ Embedding-based similarity using `sentence-transformers`
- ✅ Batch scoring support
- ✅ Ready for deployment (Streamlit Cloud or Hugging Face Spaces)

---

## 🧠 How It Works

- Text from resume and job description are embedded using `all-MiniLM-L6-v2`
- Cosine similarity and years of experience are used as model features
- A classifier predicts match probability and categorizes it into:
  - 🟢 Strong Match
  - 🟡 Medium Match
  - 🔴 Weak Match

---

## 🗂 Project Structure

```
CandidateScoring/
├── data/
│   ├── final_labeled_dataset.csv
│   ├── test_cases.csv
│   └── batch_scoring_results.csv
├── src/
│   ├── model.joblib
│   └── train_final_classifier_model.py
├── streamlit_ui.py
├── batch_score.py
├── requirements.txt
└── README.md
```

---

## 💻 Run Locally

### 1. Create & activate virtual environment (Mac/Linux)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run streamlit_ui.py
```

---

## 📊 Batch Scoring

Score multiple candidates in one go:

```bash
python batch_score.py
```

Output: `data/batch_scoring_results.csv`

---

## ☁️ Deployment

### 👉 Streamlit Cloud

1. Push repo to GitHub  
2. Go to https://streamlit.io/cloud  
3. Select repo and set:
   - Main file: `streamlit_ui.py`
   - Python version: 3.10
4. Deploy!

---

## 🔧 Requirements

```txt
streamlit
sentence-transformers
scikit-learn
xgboost
joblib
```

---

## 👩‍💻 Author

**Usha Joy Mangalan**  
M.Tech in Data Science & Engineering, BITS Pilani  
[LinkedIn](#) | [GitHub](#)

---

## 📜 License

This project is open-source and licensed under the MIT License.
