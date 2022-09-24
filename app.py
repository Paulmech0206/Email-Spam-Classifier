import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

# creating a function to do all these preprocessing tasks.
def transform_text(text):
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # word tokenize

    y = []  # creating an empty list to store cleaned text.

    for i in text:
        if i.isalnum():  # if i is alphanumeric then append i into the list
            y.append(i)

    text = y[:]  # copying list y in variable --> text
    y.clear()  # cleared y to store new text

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # returning a joined string---> text



tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email / SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
if st.button('Predict'):
#Steps:

    #Text Preprocessing
    transformed_sms =transform_text(input_sms)


    # Text Vectorization
    vectorized_sms=tfidf.transform([transformed_sms])

    #predict
    result = model.predict(vectorized_sms)[0]
    # Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

