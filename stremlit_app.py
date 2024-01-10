import streamlit as st
from transformers import pipeline
import pandas as pd 
import io

def main():
    # Load the model
    classifier = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis", tokenizer="Zabihin/Symptom_to_Diagnosis")
    
    symptoms_dict = {
            "General Symptoms (Body)": [
                ("Fever", "I have an elevated body temperature."),
                ("Nausea", "I feel the urge to vomit."),
                ("Vomiting", "I am expelling stomach contents through my mouth."),
                ("Dizziness", "I experience a sensation of lightheadedness or unsteadiness."),
                ("Weight loss", "I have lost a significant amount of body weight."),
                ("Fatigue", "I feel extreme tiredness and lack of energy."),
                ("Excessive sweating", "I am sweating profusely."),
                ("Anemia", "I have a deficiency of red blood cells, leading to fatigue."),
                ("Chills", "I experience shivering or feeling cold."),
                ("Bulging veins", "My veins appear swollen or protruded."),
                ("Body aches", "I feel discomfort or pain throughout my body.")
            ],
            "Head/Neck": [
                ("Head pain", "I have pain in my head."),
                ("Neck pain", "I experience pain in my neck."),
                ("Headache on one side", "I feel a headache concentrated on one side of my head."),
                ("Confusion", "I am disoriented or have difficulty understanding things."),
                ("Blurred vision", "My vision is unclear or fuzzy."),
                ("Distorted vision", "I see images in an altered or abnormal way."),
                ("Sensitivity to light and sound", "I am more sensitive to light and sound than usual."),
                ("Stiff neck", "My neck feels stiff and limited in movement.")
            ],
            "Eyes": [
                ("Itchy eyes", "My eyes are experiencing itching."),
                ("Watery eyes", "My eyes are producing excessive tears."),
                ("Red eyes", "The whites of my eyes appear red."),
                ("Eye pain", "I am experiencing pain or discomfort in my eyes."),
                ("Yellow eyes", "The whites of my eyes have a yellowish tint."),
                ("Blurred vision", "My vision is unclear or fuzzy."),
                ("Sensitivity to light", "I am more sensitive to light than usual.")
            ],
            "Digestive System": [
                ("Heartburn", "I feel a burning sensation in my chest or throat."),
                ("Upper abdominal or chest pain", "I experience pain in the upper abdomen or chest."),
                ("Difficulty swallowing", "I have trouble moving food from my mouth to my stomach."),
                ("Sensation of a lump in the throat", "I feel like there is something stuck in my throat."),
                ("Excessive thirst", "I am very thirsty."),
                ("Abdominal pain", "I have pain or discomfort in my abdomen."),
                ("Severe diarrhea", "I am experiencing frequent, watery bowel movements."),
                ("Vomiting", "I am expelling stomach contents through my mouth."),
                ("Nausea", "I feel the urge to vomit."),
                ("Bloating", "I have a feeling of fullness and tightness in the abdomen."),
                ("Belching", "I am expelling gas from the stomach through the mouth."),
                ("Decreased appetite", "I have a reduced desire to eat."),
                ("Indigestion", "I am experiencing discomfort or pain in the upper abdomen.")
            ],
            "Skin": [
                ("Changes in skin color", "There are alterations in the color of my skin."),
                ("Red/itchy sores", "I have red and itchy sores on my skin."),
                ("Yellow or honey-colored scabs", "Scabs on my skin have a yellow or honey-colored appearance."),
                ("Warm, red skin", "My skin feels warm and appears red."),
                ("Scaly skin", "My skin is dry and covered with scales."),
                ("Rash", "I have an outbreak of red, raised, and often itchy skin."),
                ("Scabs", "I have dried blood or pus over a healing wound."),
                ("Fluid-filled blisters", "Blisters on my skin contain clear fluid."),
                ("Itching", "I experience a sensation that prompts me to scratch my skin."),
                ("Dry skin", "My skin lacks moisture and feels rough or flaky."),
                ("Swelling", "There is an abnormal enlargement of body parts or areas.")
            ],
            "Urinary Tract": [
                ("Pain during urination", "I feel pain or discomfort while urinating."),
                ("Burning sensation during urination", "I experience a burning or stinging feeling during urination."),
                ("Frequent urination", "I need to urinate more often than usual."),
                ("Cloudy urine", "My urine appears cloudy or murky."),
                ("Blood in urine", "There is blood visible in my urine."),
                ("Difficulty controlling bladder", "I have trouble controlling my bladder, leading to leakage."),
                ("Difficulty controlling bowels", "I have trouble controlling my bowels, leading to leakage."),
                ("Dark urine", "My urine has a darker color than usual."),
                ("Pale or clay-colored stools", "My stools have a pale or clay-like color."),
                ("Urinary urgency", "I feel a strong and sudden need to urinate.")
            ],
            "Muscle/Skeletal System": [
                ("Joint pain", "I have pain or discomfort in the joints."),
                ("Restricted movement", "There is a limitation in the normal range of motion."),
                ("Weakness", "I feel a lack of strength or energy."),
                ("Muscle wasting", "My muscles are shrinking or losing mass."),
                ("Nighttime leg cramps", "I experience cramping in my legs during the night."),
                ("Swelling in joints", "There is an abnormal enlargement of joints."),
                ("Stiffness", "I feel difficulty in moving certain body parts."),
                ("Muscle spasms", "I experience involuntary contractions of muscles.")
            ],
            "Respiratory System (Lungs)": [
                ("Sneezing", "I forcefully expel air through my nose."),
                ("Nasal congestion", "My nasal passages are blocked or congested."),
                ("Coughing", "I am expelling air from the lungs with a sudden sharp sound."),
                ("Runny nose", "My nose is producing excess mucus."),
                ("Sore throat", "I have pain or irritation in the throat."),
                ("Wheezing", "I produce a whistling sound while breathing."),
                ("Coughing attacks", "I experience sudden and severe bouts of coughing."),
                ("Shortness of breath", "I find it difficult to breathe and feel breathless."),
                ("Chest tightness", "I feel a squeezing or pressure in my chest."),
                ("Rapid breathing", "I am breathing at a faster rate than normal.")
            ]
        }
    
    
    image_path = "symptom-checkers-do-not-replace-doctors.webp"
    st.image(image_path, width=300)
    # Set up the app layout with tabs
    tabs = ["Home", "About Us", "Model Details & Evaluations"]
    active_tab = st.sidebar.radio("Select Tab", tabs)

    if active_tab == "Home":
    
        st.title("Symptom Checker and Diagnosis App")
    
        # Create text area for symptoms and update input_text dynamically
        input_text = ""
        for category, symptoms in symptoms_dict.items():
            st.sidebar.write(f"### {category}")
            category_symptoms = st.sidebar.multiselect(f"Select Symptoms in {category}", [symptom[0] for symptom in symptoms])
            if category_symptoms:
                input_text += f"\n"
                for selected_symptom in category_symptoms:
                    symptom_description = next((symptom[1] for symptom in symptoms if symptom[0] == selected_symptom), "")
                    input_text += f"{symptom_description}\n"
                input_text += "\n"

        # Display the updated input_text
        st.text_area("Symptoms", value=input_text, height=200)

        # Button to submit and get the predicted label
        if st.button("Submit"):
            # Get the predicted label
            result = classifier(input_text)
            # Print the predicted label
            predicted_label = result[0]['label']
            st.write("Predicted Label:", predicted_label)

    elif active_tab == "About Us":
        st.title("Symptom Checker and Diagnosis App")
                
        st.markdown("**[Zahra ZABIHINPOUR](https://www.linkedin.com/in/zahra-zabihinpour/)**")
        st.markdown("**[Kevin GOUPIL](https://www.linkedin.com/in/kevin-goupil/)**")
        st.markdown(" We are a dynamic duo of data scientists collaborating to enhance our skills and stay at the forefront of the latest developments. With backgrounds in science and experience working with health data, we bring a unique blend of expertise to our data science projects. Our shared passion and commitment drive us to showcase and elevate our capabilities through innovative and impactful initiatives. Join us on this journey of continuous improvement and exploration in the world of data science. ")
        st.markdown(" ")

    elif active_tab == "Model Details & Evaluations":
        # Model Overview
        st.subheader("Model Overview:")
        st.write("This model is a fine-tuned adaptation of the bert-base-cased architecture, specifically designed for text classification tasks associated with diagnosing diseases based on symptoms. The primary goal is to scrutinize natural language symptom descriptions and accurately predict one of 22 potential diagnoses.")

        # Dataset Information
        st.subheader("Dataset Information:")
        st.write("The model was trained on the Gretel/symptom_to_diagnosis dataset, which consists of 1,065 symptom descriptions in English, each labeled with one of the 22 possible diagnoses. This dataset focuses on detailed, fine-grained, single-domain diagnosis, making it suitable for tasks requiring nuanced symptom classification. For those interested in utilizing the model, the Symptom Checker and Diagnosis App, or the Inference API, are accessible at [https://huggingface.co/Zabihin/Symptom_to_Diagnosis](https://huggingface.co/Zabihin/Symptom_to_Diagnosis).")

        # Model Performance Metrics
        st.subheader("Model Performance Metrics:")

        metrics_table = """
                 Disease                         | Precision | Recall | F1-Score  
                 Allergy                         | 0.91      | 1.00   | 0.95     
                 Arthritis                       | 1.00      | 1.00   | 1.00    
                 Bronchial Asthma                | 1.00      | 1.00   | 1.00      
                 Cervical Spondylosis            | 0.91      | 1.00   | 0.95     
                 Chicken Pox                     | 1.00      | 1.00   | 1.00     
                 Common Cold                     | 1.00      | 1.00   | 1.00     
                 Dengue                          | 1.00      | 0.90   | 0.95     
                 Diabetes                        | 1.00      | 0.80   | 0.89    
                 Drug Reaction                   | 0.80      | 1.00   | 0.89     
                 Fungal Infection                | 1.00      | 1.00   | 1.00      
                 Gastroesophageal Reflux Disease | 1.00      | 0.90   | 0.95      
                 Hypertension                    | 0.91      | 1.00   | 0.95     
                 Impetigo                        | 1.00      | 1.00   | 1.00     
                 Jaundice                        | 1.00      | 1.00   | 1.00      
                 Malaria                         | 1.00      | 1.00   | 1.00      
                 Migraine                        | 1.00      | 0.90   | 0.95     
                 Peptic Ulcer Disease            | 1.00      | 1.00   | 1.00      
                 Pneumonia                       | 1.00      | 1.00   | 1.00     
                 Psoriasis                       | 1.00      | 0.90   | 0.95     
                 Typhoid                         | 1.00      | 1.00   | 1.00     
                 Urinary Tract Infection         | 0.90      | 1.00   | 0.95     
                 Varicose Veins                  | 1.00      | 1.00   | 1.00     
            """
        metrics_data = pd.read_csv(io.StringIO(metrics_table), sep="|").dropna()
        st.table(metrics_data)



if __name__ == "__main__":
    main()
