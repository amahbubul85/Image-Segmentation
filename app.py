import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained U-Net model
model = tf.keras.models.load_model('unet_model.h5', compile=False)

# Custom color map (adjust these colors based on your dataset)
COLOR_MAP = {
    0: (0, 0, 0),         # Class 0: black (e.g., background)
    1: (128, 0, 128),     # Class 1: purple (e.g., pet)
    2: (255, 255, 0),     # Class 2: yellow (e.g., outline)
}

# Convert a 2D class-index mask to a color RGB image
def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        color_mask[mask == class_idx] = color
    return Image.fromarray(color_mask)

# Prediction function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to model input
    image = np.array(image) / 255.0  # Normalize
    image = image.astype(np.float32)
    return np.expand_dims(image, axis=0)  # Add batch dim

def predict_mask(image):
    pred = model.predict(image)
    pred_mask = tf.argmax(pred, axis=-1)
    pred_mask = pred_mask[0]
    return tf.cast(pred_mask, tf.uint8).numpy()

# Streamlit UI
st.title("Semantic Segmentation App (U-Net) 🧠")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Predicting..."):
        input_tensor = preprocess_image(image)
        mask = predict_mask(input_tensor)

        # Colorize mask and display
        colored_mask = colorize_mask(mask)
        st.image(colored_mask, caption="Predicted Mask (Colored)", use_container_width=True)
