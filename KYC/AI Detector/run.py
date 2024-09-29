import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import tensorflow as tf
from PIL import Image
import numpy as np

# Define the model architecture
def build_model():
    model = Sequential()
    model.add(Conv2D(16, (4, 4), 1, activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (4, 4), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (4, 4), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

# Load model weights
model = build_model()
model.load_weights(r'D:\StudyMaterial\dbui\ai_imageclassifier.h5')  # Update with the path to your weights file

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to match model input shape
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to predict whether image is AI-generated or real
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]  # Get prediction score
    label = "AI-generated" if prediction >= 0.7 else "Real"
    probability = prediction if label == "AI-generated" else 1 - prediction
    return label, probability

# Streamlit app
def main():
    st.title("AI-Generated vs. Real Image Classifier")

    # Image uploader
    image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Display uploaded image
        img = Image.open(image_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Predict using the model
        label, probability = predict(img)

        # Display the prediction and probability
        st.write(f"Prediction: **{label}**")
        st.write(f"Probability: **{probability:.2f}**")

if __name__ == "__main__":
    main()
