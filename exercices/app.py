import streamlit as st
import joblib

import pandas as pd 
import numpy as np 
import seaborn as sns
import sklearn as sklearn

st.title("Algorithme de Nary")

st.header("Présentation du dataset")

df = pd.read_csv('../data/labeled_data.csv', sep=','); 
st.text(df.head())


st.header("Utilisation de l'algorithme")

clf = joblib.load("hatespeech.joblib.z")


tweet = st.text_input('Entrez votre texte')

if st.button("Appuyez pour valider"):
    metrics = clf.predict_proba([tweet])[0]
    st.info(metrics)



