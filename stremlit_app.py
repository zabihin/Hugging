import streamlit as st
from transformers import pipeline

def main():
    # Load the model
    classifier = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis", tokenizer="Zabihin/Symptom_to_Diagnosis")
    
    # Set up the app layout
    st.title("Symptom to Diagnosis App")
    st.write("Enter your symptoms below:")
    
    # User input
    input_text = st.text_area("Symptoms")
    
    if st.button("Submit"):
        # Get the predicted label
        result = classifier(input_text)
        
        # Print the predicted label
        predicted_label = result[0]['label']
        st.write("Predicted Label:", predicted_label)

if __name__ == "__main__":
    main()
