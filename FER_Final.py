import streamlit as st

import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the trained model
model = load_model('facial_expression.h5')  # Replace with your model path
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set the page configuration
st.set_page_config(
    page_title="Emotion Recognition and Enhancement",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for styling
def set_styles():
    light_mode = """
    body {
        background-color: #f0f5f5;
        color: #333333;
    }
    .accent {
        color: #4caf50;
    }
    /* Define other styles here */
    """
    
    dark_mode = """
    body {
        background-color: #1a1a1a;
        color: #f0f5f5;
    }
    .accent {
        color: #4caf50;
    }
    /* Define other styles here */
    """

    st.write(f'<style>{"body {" + light_mode + "}"}</style>', unsafe_allow_html=True)
    st.write(f'<style>{"@media (prefers-color-scheme: dark) {" + dark_mode + "}"}</style>', unsafe_allow_html=True)

# Add your Streamlit components and content here




st.title('Facial Emotion Recognition')

def main():
    st.sidebar.title("Activities")
    activities = ["Detection", "About","Analysis"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    uploaded_file = None

    if choice == 'Detection':
        st.subheader("Facial Emotion Recognition")

        # Upload image through Streamlit UI
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

            # Preprocess the image for the model
            img = Image.open(uploaded_file)
            img = img.resize((48, 48))
            img = img.convert('L')  # Convert to grayscale
            img_array = np.array(img)  # Convert PIL Image to numpy array
            img_array = img_array.reshape((1, 48, 48, 1))  # Reshape to (1, 48, 48, 1)
            img_array = img_array.astype('float32') / 255.0  # Normalize

            # Make prediction
            predictions = model.predict(img_array)
            predicted_label_index = np.argmax(predictions[0])
            predicted_emotion = emotion_labels[predicted_label_index]

            st.write('Predicted Emotion:', predicted_emotion)

            # Display the prediction probabilities
            # st.write('Prediction Probabilities:')
            # for i, label in enumerate(emotion_labels):
            #     st.write(f'{label}: {predictions[0][i]:.2f}')

            # # Display the prediction probabilities as a bar chart
            # st.subheader('Prediction Probabilities:')
            # fig, ax = plt.subplots()
            # ax.bar(emotion_labels, predictions[0], color='royalblue')
            # ax.set_xlabel('Emotion')
            # ax.set_ylabel('Probability')
            # ax.set_title('Prediction Probabilities')
            # st.pyplot(fig)


            

            
    elif choice =='Analysis':
            st.subheader("Facial Emotion Recognition")

        # Upload image through Streamlit UI
            uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])
            if uploaded_file is not None:
            # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

            # Preprocess the image for the model
            img = Image.open(uploaded_file)
            img = img.resize((48, 48))
            img = img.convert('L')  # Convert to grayscale
            img_array = np.array(img)  # Convert PIL Image to numpy array
            img_array = img_array.reshape((1, 48, 48, 1))  # Reshape to (1, 48, 48, 1)
            img_array = img_array.astype('float32') / 255.0  # Normalize

            # Make prediction
            predictions = model.predict(img_array)
            predicted_label_index = np.argmax(predictions[0])
            predicted_emotion = emotion_labels[predicted_label_index]

            st.write('Predicted Emotion:', predicted_emotion)
            # Display the prediction probabilities as a bar chart
            st.subheader('Prediction Probabilities:')
            fig, ax = plt.subplots()
            ax.bar(emotion_labels, predictions[0], color='royalblue')
            ax.set_xlabel('Emotion')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            st.pyplot(fig)
            

             # Emotion Heatmap
            st.subheader('Emotion Heatmap:')
            heatmap_data = np.random.random((7, 7))  # Replace with actual heatmap data
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig, use_container_width=True)



            # Emotion Explanation
            st.subheader('Emotion Explanation:')
            emotion_explanation = {
                'Angry': "Angry emotion is usually characterized by facial expressions such as furrowed brows and tightened lips.",
                'Happy': "Happy emotion involves a genuine smile, often with raised cheeks and eyes.",
                'Sad': "Sad emotion is marked by a downturned mouth, drooping eyelids, and sometimes tears.",
                'Surprise': "Surprise emotion can feature wide eyes and an open mouth, often accompanied by raised eyebrows.",
                'Neutral': "Neutral emotion indicates a calm and neutral facial expression.",
                'Fear': "Fear emotion may lead to wide eyes, a slightly open mouth, and raised eyebrows.",
                'Disgust': "Disgust emotion can involve a wrinkled nose, curled upper lip, and narrowed eyes."
            } 

            selected_emotion = st.selectbox('Select an emotion:', emotion_labels)
            st.write(emotion_explanation[selected_emotion]) 




    elif choice == 'About':
        st.subheader("About")
        st.write("The Facial Emotion Recognition System is a software application designed to analyse and interpret human facial expressions and accurately identify the underlying emotions.The system will utilize image processing and machine learning techniques to recognize emotions such as happiness, sadness, anger, surprise, fear, and disgust in real-time or from still images.")
    

    st.sidebar.subheader("Image Enhancement")
        
    enhance_type = st.sidebar.radio(
        "Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        
    if enhance_type == 'Original' and uploaded_file is not None:
        print(uploaded_file, "img")
        st.image(uploaded_file, caption='Original Image', use_column_width=True)

    elif enhance_type == 'Gray-Scale' and uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        st.image(img, caption='Gray-Scale Image')

    



    elif enhance_type == 'Contrast' and uploaded_file is not None:
        c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
        
        img = Image.open(uploaded_file)
        img = img.resize((48, 48))
        
        enhancer = ImageEnhance.Contrast(img)
        img_output = enhancer.enhance(c_rate)
        
        st.image(img_output, caption='Contrast Enhanced Image')

    elif enhance_type == 'Brightness' and uploaded_file is not None:
        c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
        
        img = Image.open(uploaded_file)
        img = img.resize((48, 48))
        
        img_array = np.array(img)  # Convert PIL Image to NumPy array
        img_array = img_array.astype(np.uint8)  # Ensure the correct data type
        
        enhancer = ImageEnhance.Brightness(img)
        img_output = enhancer.enhance(c_rate)
        
        st.image(img_output, caption='Brightness Enhanced Image')



    elif enhance_type == 'Blurring' and uploaded_file is not None:
        blur_rate = st.sidebar.slider("Blur Level", 0.5, 3.5)
        img = Image.open(uploaded_file)
        img = img.resize((48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        new_img = img_array[0]
        img_bgr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        blur_img = cv2.GaussianBlur(img_bgr, (11, 11), blur_rate)
        st.image(blur_img, caption='Blurred Image', channels="BGR")

        

if __name__ == '__main__':
    main()
