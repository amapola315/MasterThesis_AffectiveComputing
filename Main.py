import streamlit as st
import pandas as pd
import datetime
import os
from data_processing import process_files
#from emotion_detection import load_data, preprocess_data, train_and_compare_models

# File paths
participant_id_file = 'participant_id.txt'
data_file_path = 'c:/thesis/code/data/responses_test.csv'
arousal_image_path = 'c:/thesis/code/Self-Assessment-Manikin-SAM_Arousal.png'
valence_image_path = 'c:/thesis/code/Self-Assessment-Manikin-SAM_Valence.png'

# Default values for the session
default_values = {
    'session_id': "S00",
    'age': "18-24",
    'stimuli_id': "NS_00",
    'gender': "Female",
    'arousal': 5,
    'valence': 5,
    'emotion_classes': []
}
       
# Function to generate the next Participant ID
def get_next_participant_id():
    try:
        if os.path.exists(participant_id_file):
            with open(participant_id_file, 'r') as file:
                last_id = int(file.read().strip().replace('P', ''))  # Get the last number used
                next_id = last_id + 1
        else:
            next_id = 1  # Start from 1 if no file exists

        with open(participant_id_file, 'w') as file:
            file.write(f'P{next_id}')  # Write the new ID back to the file

        return f'P{next_id}'  # Return the new Participant ID
    except Exception as e:
        st.error(f"Error generating Participant ID: {e}")
        return None

# Ensure the data directory exists
# Ensure the data directory exists
def ensure_data_directory():
    folder_path = os.path.dirname(data_file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to save assessment data
def save_evaluation(data):
    try:
        ensure_data_directory()  # Ensure the directory exists
        df = pd.DataFrame(data)
        write_header = not os.path.exists(data_file_path)
        df.to_csv(data_file_path, mode='a', header=write_header, index=False)
        st.success("✅ Evaluation successfully saved!, you can proceed to Emotion Detection")
    except Exception as e:
        st.error(f"❌ Error saving evaluation: {e}")

# Function to validate the evaluation before saving
def validate_evaluation():
    if not st.session_state.participant_id:
        st.warning("Please generate a Participant ID first.")
        return False
    if st.session_state.arousal < 0 or st.session_state.arousal > 10:
        st.warning("Arousal must be between 0 and 10.")
        return False
    if st.session_state.valence < 0 or st.session_state.valence > 10:
        st.warning("Valence must be between 0 and 10.")
        return False
    return True

# Function to get the current Participant ID
def get_current_participant_id():
    
    
    if os.path.exists(participant_id_file):
        with open(participant_id_file, 'r') as file:
            return file.read().strip()
    else:
        return None
# Function to update the existing evaluation with processed ECG and GSR data
def update_evaluation_with_biometrics(participant_id, session_id, stimuli_id, merged_data):
    try:
        ensure_data_directory()  # Ensure the directory exists

        # Load the existing data
        if os.path.exists(data_file_path):
            df = pd.read_csv(data_file_path)

            # Convert the necessary columns to float to avoid FutureWarning
            for column in ['EDA_mean','EDA_std','EDA_min','EDA_max','EDA_range','EDA_median',
                           'BVP_mean','BVP_std','BVP_min','BVP_max','BVP_range','BVP_median',
                           'IBI_mean','IBI_std','IBI_min','IBI_max','IBI_range','IBI_rmssd',
                           'HR_mean','HR_std','HR_min','HR_max','HR_range','HR_median']:
                if column in df.columns:
                    df[column] = df[column].astype(float)

            # Find the row to update based on Participant_ID, Session_ID, and Stimuli_ID
            row_to_update = df[(df['Participant_ID'] == participant_id) & 
                               (df['Session_ID'] == session_id) & 
                               (df['Stimuli_ID'] == stimuli_id)]

            if not row_to_update.empty:     # Update the corresponding row with new biometrics data            
                df.loc[row_to_update.index, 'EDA_mean'] = merged_data['EDA_mean'].values[0]
                df.loc[row_to_update.index, 'EDA_std'] = merged_data['EDA_std'].values[0]
                df.loc[row_to_update.index, 'EDA_min'] = merged_data['EDA_min'].values[0]
                df.loc[row_to_update.index, 'EDA_max'] = merged_data['EDA_max'].values[0]
                df.loc[row_to_update.index, 'EDA_range'] = merged_data['EDA_range'].values[0]                
                df.loc[row_to_update.index, 'EDA_median'] = merged_data['EDA_median'].values[0]
                
                df.loc[row_to_update.index, 'BVP_mean'] = merged_data['BVP_mean'].values[0]
                df.loc[row_to_update.index, 'BVP_std'] = merged_data['BVP_std'].values[0]
                df.loc[row_to_update.index, 'BVP_min'] = merged_data['BVP_min'].values[0]
                df.loc[row_to_update.index, 'BVP_max'] = merged_data['BVP_max'].values[0]
                df.loc[row_to_update.index, 'BVP_range'] = merged_data['BVP_range'].values[0]
                df.loc[row_to_update.index, 'BVP_median'] = merged_data['BVP_median'].values[0]
                
                df.loc[row_to_update.index, 'IBI_mean'] = merged_data['IBI_mean'].values[0]
                df.loc[row_to_update.index, 'IBI_std'] = merged_data['IBI_std'].values[0]
                df.loc[row_to_update.index, 'IBI_min'] = merged_data['IBI_min'].values[0]
                df.loc[row_to_update.index, 'IBI_max'] = merged_data['IBI_max'].values[0]
                df.loc[row_to_update.index, 'IBI_range'] = merged_data['IBI_range'].values[0]
                df.loc[row_to_update.index, 'IBI_rmssd'] = merged_data['IBI_rmssd'].values[0]
                
                df.loc[row_to_update.index, 'HR_mean'] = merged_data['HR_mean'].values[0]
                df.loc[row_to_update.index, 'HR_std'] = merged_data['HR_std'].values[0]
                df.loc[row_to_update.index, 'HR_min'] = merged_data['HR_min'].values[0]
                df.loc[row_to_update.index, 'HR_max'] = merged_data['HR_max'].values[0]
                df.loc[row_to_update.index, 'HR_range'] = merged_data['HR_range'].values[0]
                df.loc[row_to_update.index, 'HR_median'] = merged_data['HR_median'].values[0]                

                # Save the updated data back to CSV
                df.to_csv(data_file_path, index=False)
                st.success(f"✅ Participant {participant_id} - Session {session_id} - Stimuli {stimuli_id} updated with biometric data!")
            else:
                st.error(f"❌ No record found for Participant {participant_id}, Session {session_id}, Stimuli {stimuli_id}.")
        else:
            st.error("❌ Data file does not exist.")
    except Exception as e:
        st.error(f"❌ Error updating evaluation with biometrics: {e}")


# Title of the application
st.title("Biofeedback System to Elicit Emotional Responses (prototype)")
st.write("Dataset: DREAMER")


# Form with multiple sections
with st.form("emotion_form"):
    # Create tabs for the different sections
    tabs = st.tabs(["Sensor Calibration", "Data Collection", "Emotion Detection", "Assessment"])

    # Section 1: Sensor Calibration
    with tabs[0]:
        st.subheader("1. Sensor Calibration")
        st.write("Calibrate your sensors before collecting data.")
        calibration_status = st.radio("Calibration Status", ("Not Calibrated", "Calibrated"))
        if calibration_status == "Not Calibrated":
            st.warning("Please calibrate the sensors before proceeding.")
        else:
            st.success("Calibration confirmed, you can proceed to Data Collection.")
        
        if st.form_submit_button('Proceed from Calibration'):
            st.session_state.calibration_done = True

    
    # Section 2: Data Collection
    with tabs[1]:
        st.subheader("2. Data Collection")
        st.write("Collecting emotion data using stimuli and biometric sensors.")
        
        if st.form_submit_button('New Session'):
            # Generate a new Participant ID and reset values
            new_participant_id = get_next_participant_id()
            if new_participant_id:
                st.session_state.participant_id = new_participant_id         

        # Initialize or retrieve Participant ID
        if 'participant_id' not in st.session_state:
            st.session_state.participant_id = get_current_participant_id() or get_next_participant_id()

        # Display current Participant ID
        st.write(f"**Participant ID:** {st.session_state.participant_id}")

        # Initialize session state with default values dynamically
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value

    
        col1, col2 = st.columns(2)
        
        # Place dropdowns in columns
        with col1:
            # Dropdown for Session ID
            session_id = st.selectbox("Session ID", options=["S00","S01", "S02"], index=["S00","S01", "S02"].index(st.session_state.session_id))
            
            # Dropdown for Age ranges
            age = st.selectbox("Age Range", options=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], index=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"].index(st.session_state.age))
        
        with col2:
            # Dropdown for Stimuli ID
            stimuli_id = st.selectbox("Stimuli ID", options=["NS_00", "SEUXIS_00", "SEUXIS_01", "ROCIO_00", "ROCIO_01", "JOSETH_00", "JOSETH_01"], index=["NS_00", "SEUXIS_00", "SEUXIS_01", "ROCIO_00", "ROCIO_01", "JOSETH_00", "JOSETH_01"].index(st.session_state.stimuli_id))
                                    
            # Dropdown for Gender
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.gender))
        
        # Emotional rating sliders with images (0 = Very Low, 10 = Very High)
        col3, col4 = st.columns(2)
        
        with col3:
            arousal = st.slider('**Arousal:** This measures how much energy or excitement you feel when viewing the stimuli.', 1, 5, st.session_state.arousal)
            st.image(arousal_image_path, caption="Arousal Scale")
        
        with col4:
            valence = st.slider('**Valence:** This describes how positive or negative your feelings are about the stimuli.', 1, 5, st.session_state.valence)
            st.image(valence_image_path, caption="Valence Scale")
        
        # Map Arousal and Valence to one of the 4 classes
        if arousal >= 3 and valence >= 3:
            emotion_4_classes = "HVHA"
        elif arousal < 3 and valence >= 3:
            emotion_4_classes = "HVLA"
        elif arousal >= 3 and valence < 3:
            emotion_4_classes = "LVHA"
        else:
            emotion_4_classes = "LVLA"
        
        # Add Self-annotation (8 classes) with Multi Choice Grid
        st.write("### Self-annotation")
        emotion_classes = st.multiselect("Select specific emotional states:", 
                                          options=["happiness", "sadness", "anger", "disgust", "amusement", "excitement", "calmness", "surprise", "fear"], default=st.session_state.emotion_classes)
    
    								

             # Save the assessment with validation
        if st.form_submit_button('Save Evaluation'):
            if validate_evaluation():
                timestamp = datetime.datetime.now()
                all_emotions = ["happiness", "sadness", "anger", "disgust", "amusement", "excitement", "calmness", "surprise", "fear"]
                emotion_data = {emotion: [1] if emotion in emotion_classes else [0] for emotion in all_emotions}
        
                data = {
                    'timestamp': [timestamp],
                    'Participant_ID': [st.session_state.participant_id],
                    'Session_ID': [session_id],
                    'Stimuli_ID': [stimuli_id],
                    'Age': [age],
                    'Gender': [gender],
                    'Arousal': [arousal],
                    'Valence': [valence],
                    'Emotion_4_Classes': [emotion_4_classes],
                    'EDA_mean':0,
                    'EDA_std':0,
                    'EDA_min':0,
                    'EDA_max':0,
                    'EDA_range':0,
                    'EDA_median':0,
                   	'BVP_mean':0,	
                    'BVP_std':0,	
                    'BVP_min':0,	
                    'BVP_max':0,	
                    'BVP_range':0,	
                    'BVP_median':0,	
                    'IBI_mean':0,	
                    'IBI_std':0,	
                    'IBI_min':0,	
                    'IBI_max':0,	
                    'IBI_range':0,	
                    'IBI_rmssd':0,	
                    'HR_mean':0,	
                    'HR_std':0,	
                    'HR_min':0,	
                    'HR_max':0,	
                    'HR_range':0,	
                    'HR_median':0	

                }
                data.update(emotion_data)
                
                with st.spinner('Saving evaluation...'):
                    save_evaluation(data)            
                
            
     # Section 3: Results of Detected Emotion
     # Formulario en la pestaña de Resultados
        with tabs[2]:
            st.subheader("3. Emotion Detection")
            st.write("Emotional response mapped from arousal and valence.")
            
            # Ask for the folder path where the Empatica E4 CSV files are located
            folder_path = st.text_input("Enter the folder path where the Empatica E4 CSV files are located:")
            
            # Verify if the folder exists
            if folder_path and os.path.exists(folder_path):
                # List all CSV files in the folder
                csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                
                if csv_files:
                    st.success(f"Found {len(csv_files)} CSV files in the folder:")
                    for file in csv_files:
                        st.write(f"- {file}")
                    
                    # Button to load and process the CSV files
                    if st.form_submit_button("Load and Process"):
                        try:
                            # Load all CSV files into a dictionary of DataFrames
                            selected_files = {}
                            for file in csv_files:
                                file_path = os.path.join(folder_path, file)
                                selected_files[file] = pd.read_csv(file_path)
                            
                            # Process the selected files
                            merged_data = process_files(selected_files)
                            
                            if not merged_data.empty:
                                st.success("Files processed successfully.")
                                st.write("Merged data:", merged_data.head())
                                
                                # Update the existing evaluation with the biometric data
                                update_evaluation_with_biometrics(st.session_state.participant_id, 
                                                                  session_id, 
                                                                  stimuli_id, 
                                                                  merged_data)
                            else:
                                st.error("Error processing files or data is empty.")
                        except Exception as e:
                            st.error(f"Error processing files: {e}")
                else:
                    st.warning("No CSV files were found in the specified folder.")
            else:
                if folder_path:
                    st.error("The provided folder path is not valid. Please verify that the folder exists.")
                
    # Section 4: Empathy Assessment
        with tabs[3]:
            st.subheader("4. Assessment")
            st.write("Assess your emotional alignment with the emotions of the artworks.")
            
            detect_button = st.form_submit_button("Assess")            
            
            if detect_button:        
                        
                empathy_level = st.slider("Empathy Level (0 = No empathy, 10 = High empathy)", 0, 10, 5)                
        
                # Mostrar el mensaje final al usuario
                st.write("The emotion detection model has been trained based on the input data.")
                st.write(f"Your empathy level is {empathy_level}/10.")
                
               
            
            # Submit button for the form
            end_button = st.form_submit_button("end test")
        
            # If the form is submitted
            if end_button:
                st.write("Thanks for participate.")
            
            
            
            
            