import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Configuration variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = [300, 100]
MODEL_PATH = 'crnn_model.pth'
INPUT_SIZE = 64
HIDDEN_SIZE = 128
OUTPUT_SIZE = 31  # VOCAB_SIZE + 1
NUM_LAYERS = 2
MAX_LENGTH = 20

# Model Definition (CRNN, FeatureExtractor, BiLSTM)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        self.fc = nn.Linear(768, 64)

    def forward(self, images):
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        outputs = self.maxpool(self.relu(self.bn2(self.conv2(outputs))))
        outputs = self.maxpool(self.relu(self.bn3(self.conv3(outputs))))

        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.reshape(outputs.shape[0], -1, outputs.shape[2] * outputs.shape[3])
        outputs = torch.stack([self.relu(self.fc(outputs[i])) for i in range(outputs.shape[0])])
        outputs = self.dropout(outputs)
        return outputs

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, labels):
        h0 = torch.zeros(self.num_layers * 2, labels.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, labels.size(0), self.hidden_size).to(DEVICE)

        outputs, _ = self.lstm(labels, (h0, c0))
        outputs = torch.stack([self.fc(outputs[i]) for i in range(outputs.shape[0])])
        outputs = self.softmax(outputs)
        return outputs

class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CRNN, self).__init__()
        self.RNN = BiLSTM(input_size, hidden_size, output_size, num_layers)
        self.CNN = FeatureExtractor()

    def forward(self, images):
        features = self.CNN(images)
        outputs = self.RNN(features)
        return outputs

# Vocabulary-related functions
def get_vocabulary():
    vocabulary = [' ', "'", '-', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    int2char = dict(enumerate(vocabulary))
    int2char = {k + 1: v for k, v in int2char.items()}
    char2int = {v: k for k, v in int2char.items()}
    return int2char, char2int

def encode(string):
    _, char2int = get_vocabulary()
    token = torch.tensor([char2int[i] for i in string])
    pad_token = F.pad(token, pad=(0, MAX_LENGTH - len(token)), mode='constant', value=0)
    return pad_token

def decode(token, vocabulary):
    int2char, _ = get_vocabulary()
    token = token[token != 0]
    string = [int2char[i.item()] for i in token]
    return "".join(string)

# Streamlit app
@st.cache_resource
def load_model():
    model = CRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) / 255.
    image = np.expand_dims(image, 0)
    image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image

def infer_image(model, image):
    image = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        prediction = model(image)
        prediction = prediction.permute(1, 0, 2)
        _, max_index = torch.max(prediction, dim=2)

        prediction = max_index.squeeze(1).cpu()
        result = decode(prediction, get_vocabulary()[0])
    return result

def main():
    st.title("Handwritten Text Recognition with CRNN")
    st.write("Upload an image of handwritten text, and the model will predict the text.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        model = load_model()

        prediction = infer_image(model, image)
        st.write(f"**Predicted Text:** {prediction}")

if __name__ == "__main__":
    main()
