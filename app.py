import streamlit as st
import pickle
import os
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Text Sentiment Analysis",
    page_icon="🎯",
    layout="wide"
)

# -------------------- NLTK SETUP --------------------
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)

setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------- TEXT PREPROCESSING --------------------
def normalize_elongated_words(text):
    return re.sub(r"(.)\1{2,}", r"\1", text)

def preprocess_for_model(text):
    text = text.lower()
    text = normalize_elongated_words(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

def has_elongated_words(text):
    return bool(re.search(r'(.)\1{2,}',text.lower()))



# -------------------- LOAD MODEL --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model_and_vectorizer():
    model_path = os.path.join(BASE_DIR, "sentiment_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found")
        st.stop()

    if not os.path.exists(vectorizer_path):
        st.error("❌ Vectorizer file not found")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# -------------------- SESSION STATE --------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# -------------------- SIDEBAR --------------------
st.sidebar.title("Sentiment App")

st.sidebar.markdown("""
### 🔍 About
This app predicts **sentiment (Positive / Negative)** using  
**TF-IDF + Machine Learning**
""")

with st.sidebar.expander("How to use"):
    st.markdown("""
    1. Enter text  
    2. Click **Analyze Sentiment**  
    3. View prediction & confidence
    """)

st.sidebar.markdown("### 🧪 Try Examples")

if st.sidebar.button("😊 Positive Example"):
    st.session_state.user_text = "I absolutely loved this movie! The acting was fantastic."

if st.sidebar.button("😡 Negative Example"):
    st.session_state.user_text = "This was a terrible experience. Completely disappointing."

if st.sidebar.button("🧹 Clear Text"):
    st.session_state.user_text = ""

st.sidebar.markdown("---")
st.sidebar.markdown("""
### ⚡ Use Cases
- Product Reviews  
- Movie Reviews  
- Customer Feedback  
- Social Media Analysis  
""")

# -------------------- MAIN UI --------------------
st.title("💬 Text Sentiment Analysis")
st.write("Analyze sentiment of text using Machine Learning")

col1, col2 = st.columns([3, 2])

with col1:
    user_input = st.text_area(
        "Enter your text",
        placeholder="Example: I really enjoyed this product...",
        height=200,
        value=st.session_state.user_text
    )

with col2:
    st.info("""
    **Model Info**
    - TF-IDF Vectorizer
    - ML Classifier
    - Binary Sentiment Output
    """)

# -------------------- PREDICTION --------------------
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        processed_text = preprocess_for_model(user_input)
        vectorized_text = vectorizer.transform([processed_text])

        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]

        negative_prob = probabilities[0] * 100
        positive_prob = probabilities[1] * 100
        confidence = max(negative_prob, positive_prob)

        st.subheader("📊 Analysis Result")

        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")

        st.markdown("### 🔐 Confidence Score")

        if confidence > 80:
            st.success(f"High Confidence: {confidence:.2f}%")
        elif confidence >= 60:
            st.warning(f"Medium Confidence: {confidence:.2f}%")
        else:
            st.error(f"Low Confidence: {confidence:.2f}%")

        st.markdown("### 📈 Probability Breakdown")

        st.write(f"Positive: {positive_prob:.2f}%")
        st.progress(int(positive_prob))

        st.write(f"Negative: {negative_prob:.2f}%")
        st.progress(int(negative_prob))

    if has_elongated_words(user_input):
        st.info("💡 Tip: Consider reducing elongated words for better accuracy.")
        with st.expander("🔍 Why did I get this result?"):
            st.write("""
    - Slang, spelling mistakes, or elongated words reduce confidence
    - The model works best with normal English words
    - Try clearer sentences for better accuracy
    """)
        

