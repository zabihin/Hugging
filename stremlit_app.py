import streamlit as st
from transformers import pipeline

# Load the model
classifier = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis", tokenizer="Zabihin/Symptom_to_Diagnosis")



@st.cache(allow_output_mutation=True)

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")



if user_input and button :
    result = classifier(user_input)
    predicted_label = result[0]['label']


    st.write("Prediction: ",predicted_label)