import streamlit as st
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize




nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class LemmaTokenizer(object):

    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        tokens = word_tokenize(reviews)
        return [self.wordnetlemma.lemmatize(word) for word in tokens]


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# LOAD MODEL

@st.cache_resource
def load_files():
    model = joblib.load("model_1.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_files()


# CLEANING

def remove_special_character(content):
    return re.sub(r'\[[^&@#!]]*\]', ' ', content)

def remove_url(content):
    return re.sub(r'http\S+', '', content)

def contraction_expansion(content):
    content = re.sub(r"won\'t","would not",content)
    content = re.sub(r"can\'t","can not",content)
    content = re.sub(r"don\'t","do not",content)
    content = re.sub(r"n\'t"," not",content)
    content = re.sub(r"\'re"," are",content)
    content = re.sub(r"\'s"," is",content)
    return content

def data_cleaning(content):
    content = remove_special_character(content)
    content = remove_url(content)
    content = contraction_expansion(content)
    return content



st.markdown("""
<style>

.stApp{
background-color:#f4f6f9;
}

.main-card{
background:white;
padding:35px;
border-radius:12px;
box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

.main-title{
text-align:center;
font-size:36px;
font-weight:600;
color:#1f2933;
}

.subtitle{
text-align:center;
color:#6b7280;
margin-bottom:25px;
}

.result-box{
padding:20px;
border-radius:10px;
text-align:center;
font-size:20px;
font-weight:600;
margin-top:20px;
}

.positive{
background:#e6f4ea;
color:#1e7e34;
border:1px solid #c3e6cb;
}

.negative{
background:#fdecea;
color:#b02a37;
border:1px solid #f5c6cb;
}

</style>
""", unsafe_allow_html=True)



st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Movie Review Sentiment Detection</div>', unsafe_allow_html=True)

text = st.text_area("Enter Movie Review", height=150)


# PREDICTION


if st.button("Analyze Sentiment"):

    if text.strip():

        clean_text = data_cleaning(text)
        vec = vectorizer.transform([clean_text])

        prediction = int(model.predict(vec)[0])

        confidence = None
        if hasattr(model,"predict_proba"):
            prob = model.predict_proba(vec)[0]
            confidence = max(prob)*100

        if prediction == 1:

            st.markdown(
                f'<div class="result-box positive">POSITIVE SENTIMENT 😎</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'<div class="result-box negative">NEGATIVE SENTIMENT 😡</div>',
                unsafe_allow_html=True
            )

        if confidence:
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Enter text first")

st.markdown('</div>', unsafe_allow_html=True)
