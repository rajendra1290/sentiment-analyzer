# Sentiment Analyzer (Movie Review Classification)

A Machine Learning based Sentiment Analysis Web App built using:

- Python
- Scikit-learn
- NLP (TF-IDF + Lemmatization)
- Streamlit (Web Interface)

This application predicts whether a movie review is Positive or Negative using Natural Language Processing.

---

##  Features

Text preprocessing and cleaning  
Lemmatization using NLTK  
TF-IDF Vectorization  
Machine Learning Classification Model  
Interactive Web UI using Streamlit  
Real-time sentiment prediction  

---

## Machine Learning Pipeline

1. Data Cleaning:
   - Remove special characters
   - Remove URLs
   - Expand contractions

2. Tokenization:
   - Custom LemmaTokenizer using NLTK

3. Feature Engineering:
   - TF-IDF Vectorizer
   - n-grams (1,3)
   - max_features = 2000

4. Model Training:
   - Scikit-learn classification model

---

## 📂 Project Structure
sentiment_analyzer/
│
├── app.py
├── model_1.pkl
├── vectorizer.pkl
├── README.md
└── requirements.txt


---

##  Tech Stack

- Python
- Scikit-learn
- NLTK
- Streamlit
- Joblib



##  Future Improvements

- Deep Learning model (LSTM / Transformer)
- Better UI animations
- Deploy on Streamlit Cloud
- Add multi-language support



