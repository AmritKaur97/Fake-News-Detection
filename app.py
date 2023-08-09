import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page title and favicon
st.set_page_config(page_title="Fake News Classifier", page_icon="🕵️‍♂️")

# Set overall page style
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(#e66465, #9198e5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for specific components
st.markdown(
    """
    <style>
    .stTextInput>div>div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        background-color: #43BFC7;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #36929D;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize nltk components
ps = PorterStemmer()
voc_size = 10000
sent_length = 5000

def review_cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

stop_words = set(stopwords.words("english"))

# Load the model
model = load_model('fake_news_model.h5')

# Streamlit UI code
st.title("Fake News Classification")

st.write("Enter the news content:")
user_input = st.text_area("", height=150)

if st.button("Classify"):
    # Preprocess the user input
    cleaned_input = review_cleaning(user_input)
    stemmed_input = [ps.stem(word) for word in cleaned_input.split() if not word in stop_words]
    input_corpus = ' '.join(stemmed_input)
    input_onehot_repr = [one_hot(input_corpus, voc_size)]
    input_padded = pad_sequences(input_onehot_repr, padding='pre', maxlen=sent_length)

    # Predict using the trained model
    prediction = model.predict(input_padded)

    # Classify and show result
    prediction_threshold = 0.2
    predicted_class = "True" if prediction[0][0] >= prediction_threshold else "Fake"
    prediction_proba = prediction[0][0] if predicted_class == "True" else 1 - prediction[0][0]

    st.markdown("---")
    st.markdown(f"**Classification:** {predicted_class}")
    st.markdown(f"**Confidence:** {prediction_proba:.2%}")
