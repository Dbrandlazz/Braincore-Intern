import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import CRNN, get_vocabulary, encode, decode  # Assuming you have these in a separate file 'model.py'

# Configuration variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = [300, 100]
MODEL_PATH = '/content/crnn_model.pth'
INPUT_SIZE = 64
HIDDEN_SIZE = 128
OUTPUT_SIZE = 31  # VOCAB_SIZE + 1
NUM_LAYERS = 2

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = CRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) / 255.
    image = np.expand_dims(image, 0)
    image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image

# Function to perform inference
def infer_image(model, image):
    image = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        prediction = model(image)
        prediction = prediction.permute(1, 0, 2)
        _, max_index = torch.max(prediction, dim=2)

        # Decode the prediction
        prediction = max_index.squeeze(1).cpu()
        result = decode(prediction, get_vocabulary()[0])

    return result

# Streamlit app
def main():
    st.title("Handwritten Text Recognition with CRNN")
    st.write("Upload an image of handwritten text, and the model will predict the text.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # Load the model
        model = load_model()

        # Perform inference
        prediction = infer_image(model, image)

        # Display the prediction
        st.write(f"**Predicted Text:** {prediction}")

if __name__ == "__main__":
    main()