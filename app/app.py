import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import base64

model = load_model("../models/model1_vgg16.h5")

classes=['Bean','Bitter_gourd','Bottle_Gourd','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Pumpkin','Radish','Tomato']

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)
    image = tf.image.resize(image, [255, 255])
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


st.markdown("""
    <h1 style='text-align: center; color: #006400; font-size: 52px; font-weight: bold;'>VEG-X</h1>
    <h3 style='text-align: center; color: #556B2F; font-size: 28px;'>VEGETABLE CLASSIFICATION SYSTEM</h3>
""", unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("Upload an image of a vegetable", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Vegetable Image", use_container_width=True)

    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]
        confidence = output[0][pred_idx] * 100
        pred_class = classes[pred_idx]

        st.write(f"### Predicted Class: {pred_class}")
