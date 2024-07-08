import streamlit as st

import pandas as pd

# Load saved models and encoders
import pickle

# Load label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load preprocessing pipeline
with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# Load stacking classifier
with open('stacking_clf.pkl', 'rb') as f:
    stacking_clf = pickle.load(f)


# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    
    for column, le in label_encoders.items():
        df[column] = le.transform(df[column])
    
    X_transformed = preprocessing_pipeline.transform(df)
    return X_transformed

# Function to predict using the loaded classifier
def predict(data):
    try:
        X_transformed = preprocess_input(data)
        
        prediction = stacking_clf.predict(X_transformed)
        
        if prediction[0] == 1:
            return "Pass"
        else:
            return "Fail"
    
    except Exception as e:
        st.error(f'Error predicting: {e}')

# Streamlit app
def main():
    st.set_page_config(page_title="Cognitive Student Grade Prediction App")
    # Page title and description
    st.title('Cognitive Environmental Analysis Student Grade Prediction App')
    st.markdown('### Predict if you will Pass or Fail based on your inputs')
    
    # Sidebar with app description
    st.sidebar.markdown('This app predicts whether a student will pass or fail based on their demographics and educational background.')
    st.sidebar.markdown('Input your information on the left and click **Predict**.')
    
    # Input fields for user
    col1, col2 = st.columns([1, 2])
    with col1:
        gender = st.selectbox('Gender', ['Female', 'Male'])
        nationality = st.selectbox('Nationality', ['Saudi', 'Non-Saudi'])
        class_level = st.selectbox('Class Level', [1, 2, 3])
    with col2:
        age = st.slider('Age', 15, 30, 20)
        school_type = st.selectbox('School Type', list(label_encoders['School Type'].classes_))
        main_administration = st.selectbox('Main Administration', list(label_encoders['Main Administration'].classes_))
        candidacy_type = st.selectbox('Candidacy Type', ['Self-Candidacy', 'Talented-Candidacy'])
    
    # Predict button and result
    if st.button('Predict', key='predict_button'):
        user_input = {
            'Gender': gender.lower(),
            'Nationality': nationality,
            'Class Level': class_level,
            'Age': age,
            'School Type': school_type,
            'Main Administration': main_administration,
            'Candidacy type': candidacy_type
        }
        
        prediction = predict(user_input)
        if prediction == "Pass":
            st.success('Congratulations! You have passed.')
        else:
            st.error('Sorry, you have failed.')
    
    # Footer with team members and roll numbers
    st.markdown("---")
    st.markdown("#### Team Members:")
    st.markdown("- **Wahaj Raza** CS-21055")
    st.markdown("- **Farrukh Niaz** CS-21064")
    st.markdown("- **Huzaifa Naseer Khan** CS-21067")

if __name__ == '__main__':
    main()
