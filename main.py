import pandas as pd
import pickle
import re
import streamlit as st
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# -------------------
# Load Model, Vectorizer, Scaler
# -------------------
@st.cache_resource
def load_models():
    with open("Models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Models/countVectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    with open("Models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, cv, scaler

model, cv, scaler = load_models()

# -------------------
# Preprocess Function
# -------------------
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))  # remove non-letters
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return " ".join(text)

# -------------------
# Streamlit App
# -------------------
st.title("📊 Sentiment Analysis on Reviews")
st.write("Upload a CSV file (like Reviews.csv) to predict sentiments and see graphs.")

uploaded_file = st.file_uploader("Upload Reviews.csv", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")  # fallback

    # Detect text column automatically
    text_col = None
    for col in df.columns:
        if col.lower() in ["review", "text", "sentence", "feedback", "comment"]:
            text_col = col
            break
    if text_col is None:
        text_col = df.select_dtypes(include="object").columns[0]

    st.write(f"✅ Using column **{text_col}** for prediction.")

    # Preprocess & predict
    df['cleaned'] = df[text_col].apply(preprocess_text)
    X_new = cv.transform(df['cleaned']).toarray()
    X_new = scaler.transform(X_new)
    preds = model.predict(X_new)
    df['Predicted_Feedback'] = preds
    df['Predicted_Label'] = df['Predicted_Feedback'].map({1: "Positive", 0: "Negative"})

    # Show predictions
    st.subheader("🔮 Sample Predictions")
    st.write(df[[text_col, "Predicted_Label"]].head(10))

    # Pie chart
    st.subheader("📌 Sentiment Distribution")
    fig, ax = plt.subplots()
    df['Predicted_Label'].value_counts().plot(
        kind="pie", autopct="%1.1f%%", colors=["green", "red"], ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    # Bar chart
    st.subheader("📌 Sentiment Count")
    fig, ax = plt.subplots()
    df['Predicted_Label'].value_counts().plot(
        kind="bar", color=["green", "red"], edgecolor="black", ax=ax
    )
    plt.title("Sentiment Count")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot(fig)

    # Save results
    output_path = "predicted_reviews.csv"
    df.to_csv(output_path, index=False)
    st.success(f"✅ Predictions saved to {output_path}")
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="predicted_reviews.csv",
        mime="text/csv",
    )
