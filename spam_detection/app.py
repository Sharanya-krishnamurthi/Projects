import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def pre_processing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    processed = []
    for t in text:
        if (
            t.isalnum()
            and t not in stopwords.words("english")
            and t not in string.punctuation
        ):
            processed.append(ps.stem(t))
    return " ".join(processed)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    processed_text = pre_processing(input_sms)
    result = model.predict(tfidf.transform([processed_text]))[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
