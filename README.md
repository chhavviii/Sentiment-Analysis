# 💬 Text Sentiment Analysis Web App

A Machine Learning–based **Text Sentiment Analysis** application that predicts whether a given text expresses **Positive or Negative sentiment**.  
The project combines **NLP preprocessing, TF-IDF feature extraction, classical ML models**, and a **Streamlit web interface** for real-time predictions.

---

## 🚀 Project Overview

This project focuses on building an **end-to-end sentiment analysis pipeline**, spanning data preprocessing, model training, and deployment as an interactive web application.

The app allows users to:
- Enter any text or review
- Get instant sentiment prediction
- View confidence scores and probability breakdown
- Understand model behavior through explanations

---

## 🧠 Key Features

### 📊 Sentiment Prediction
- Predicts **Positive 😊** or **Negative 😞** sentiment
- Color-coded output for clarity
- Displays confidence score (High / Medium / Low)

### 📈 Probability Breakdown
- Shows exact probability for each class
- Visual progress bars for Positive & Negative sentiment

### 🔬 NLP Preprocessing
- Lowercasing
- HTML tag removal
- Special character removal
- Tokenization
- Stopword removal
- Lemmatization
- Normalization of elongated words (e.g. *loveddddd → loved*)

### 🎨 User Interface
- Clean and professional Streamlit UI
- Sidebar with:
  - App description
  - How-to-use guide
  - Example inputs
- Responsive layout (desktop & mobile)

### 🛡️ Error Handling & Robustness
- Handles empty input gracefully
- Displays user-friendly warning messages
- Safe failure handling during preprocessing and prediction

---

## ⚙️ Technical Highlights

### 💾 Model & Resource Management
- Trained Machine Learning model loaded automatically
- TF-IDF vectorizer used for feature extraction
- File existence checks before loading
- Caching with `@st.cache_resource` for faster performance

### 📦 NLTK Data Handling
- Automatically downloads required NLTK resources
- One-time cached download
- Loading spinner during setup

---

## 🏗️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **NLTK**
- **Streamlit**
- **Pickle**

---

## 📁 Project Structure

text-sentiment-analysis-app/
│
├── app.py # Streamlit web application (main entry point)
├── sentiment_model.pkl # Trained sentiment classification model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer used during training
├── requirements.txt # Project dependencies
├── README.md # Project documentation
│
├── notebook/
│ └── sentiment_analysis.ipynb # EDA, preprocessing & model training notebook
│
├── screenshots/ # Application screenshots (optional)
│ └── app_demo.png


The **Jupyter notebook** is used for experimentation and model training,  
while the **Streamlit app (`app.py`)** represents the final deployed solution.

---

## ▶️ How to Run the Project

### Run Locally

Follow the steps below to run the app on your system.

#### 1️⃣ Clone the repository
```bash
git clone https://github.com/chhavviii/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
streamlit run app.py
http://localhost:8501


