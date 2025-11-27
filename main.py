import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path

# Try Orange (optional)
try:
    import Orange
    ORANGE_AVAILABLE = True
except:
    ORANGE_AVAILABLE = False

MODEL_PATH = Path("models/fake_news_model.joblib")

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detector (with Explanation)")
st.write("Paste any headline or news article. The AI will classify it as **Real** or **Fake**, highlight key words, and give external fact-check links.")


# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found! Run train_model.py first.")
        return None, None

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    vectorizer = saved["vectorizer"]
    return model, vectorizer


model, vectorizer = load_model()
if model is None:
    st.stop()


# -------------------------
# Input From User
# -------------------------
mode = st.radio("Select Input Type:", ["Headline", "Full Article"])
text = st.text_area("Enter news text:", height=200)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    user_text = text.strip()

    # -------------------------
    # Clean
    # -------------------------
    def clean(txt):
        txt = re.sub(r"\s+", " ", txt)
        return txt

    cleaned = clean(user_text)

    # -------------------------
    # Predict
    # -------------------------
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        labels = model.classes_
        prob_dict = {labels[i]: float(proba[i]) for i in range(len(labels))}
    else:
        prob_dict = {pred: 1.0}

    st.markdown("## 🔮 Prediction")
    st.write(f"**Result:** {pred.upper()}")
    st.write("**Confidence:**", f"{prob_dict.get(pred, 0):.2f}")

    # -------------------------
    # Explain (Word Contributions)
    # -------------------------
    st.markdown("## 🧠 Why the model predicted this")

    try:
        coef = model.coef_
        classes = model.classes_
        idx = list(classes).index(pred)
        feature_names = vectorizer.get_feature_names_out()

        X_dense = X.tocoo()
        contributions = {}

        for i, val in zip(X_dense.col, X_dense.data):
            word = feature_names[i]
            weight = coef[idx, i]
            contributions[word] = float(weight * val)

        top = sorted(contributions.items(), key=lambda x: -abs(x[1]))[:15]

        df = pd.DataFrame(top, columns=["word", "contribution"])
        df["effect"] = df["contribution"].apply(lambda x: "supports prediction" if x > 0 else "opposes prediction")

        st.dataframe(df)

        # Highlight words in the original text
        highlighted = cleaned
        def underline(word):
            return f"<u><b>{word}</b></u>"

        for w, _ in top[:10]:
            highlighted = re.sub(fr'(?i){re.escape(w)}', underline(w), highlighted)

        st.markdown("### Highlighted Text (Important Words Underlined)")
        st.markdown(f"<div style='font-size:16px'>{highlighted}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not compute word contributions: {e}")

    # -------------------------
    # Optional Orange
    # -------------------------
    if ORANGE_AVAILABLE:
        st.markdown("## 🔶 Orange Text Mining Add-on Detected")
        st.write("Orange is installed. You may also process this text in Orange for tokenization, sentiment, etc.")

    # -------------------------
    # External Sources
    # -------------------------
    st.markdown("## 🔗 External Fact-Checking Sites")
    st.write("""
- **Alt News (India)** — https://www.altnews.in  
- **BoomLive** — https://www.boomlive.in  
- **FactCheck.org**  
- **AFP Fact Check**  
- **Snopes.com**  
- **Reuters Fact Check**  
    """)

    st.markdown("## ✅ Quick Human Checks")
    st.write("""
1. Check if reputable news outlets are reporting the same story.  
2. Look for date, author, & references in the article.  
3. Reverse image search any images.  
4. Search fact-check websites using keywords from the headline.  
5. If wording sounds extreme or emotional, be extra careful.  
    """)

    st.success("Analysis Completed ✔️")
