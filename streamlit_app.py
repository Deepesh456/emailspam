import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import stopwords
import string
import streamlit as st
import pickle
from PIL import Image


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    #     i.str.replace('[^a-zA0-9]', ' ', regex=True)
    #     y.append(i)
        # if i not in '[^a-zA0-9]':
        #     y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.get_stopwords('english',cache=True) and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model5.pkl','rb'))

st.title("Email/SMS Spam Classifier")
image=Image.open("email_spam.png")
st.image(image,width=500)

input_sms=st.text_input("Enter the Message")
if st.button('Predict'):


    # 1. preprocessing

    transform_sms=transform_text(input_sms)

    # 2. vectorize

    vector_input=tfidf.transform([transform_sms])

    # 3. predict

    result=model.predict(vector_input)[0]

    # 4. display

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
