import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import librosa
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from werkzeug.utils import secure_filename
import face_recognition

# --- Image Deepfake Detection Code ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def load_image_model():
    model_checkpoint_path = "C:/Users/91998/OneDrive/Desktop/saved_model/cnn5.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.eval()
    return model, device

# Image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# --- Audio Deepfake Detection Code ---
def extract_features_from_audio(audio_bytes, max_length=500, sr=16000, n_mfcc=40):
    try:
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def load_audio_model():
    try:
        model = tf.keras.models.load_model(r'C:\Users\91998\Downloads\DeepFake_Detection-main\DEEPFAKE_DETECTION_APP\MODELS\updated_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Video Deepfake Detection Code ---
IMG_SIZE = 224
SEQ_LENGTH = 20
PREDICTION_THRESHOLD = 0.88  # Adjust this threshold as needed
NUM_FEATURES = 2048
MODEL_PATH = r"C:\Users\91998\OneDrive\Desktop\saved_model\DeepFake_Detection-main\Deploy\models\inceptionNet_model.h5"

try:
    sequence_model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def sequence_prediction(file_path):
    class_vocab = ['FAKE', 'REAL']
    frames = load_video(file_path)

    if len(frames) == 0:
        st.error("Error: No valid face frames detected in the video.")
        return "No face frames found"

    frame_features, frame_mask = prepare_single_video(frames)

    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    if probabilities[0] > PREDICTION_THRESHOLD:
        prediction_label = class_vocab[0]  # FAKE
    else:
        prediction_label = class_vocab[1]  # REAL

    return prediction_label

def load_video(path, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        st.error(f"❌ Error: Cannot open video {path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"📽️ Total frames in video: {total_frames}")

    skip_frames_window = max(int(total_frames / SEQ_LENGTH), 1)
    frames = []

    for frame_cntr in range(SEQ_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr * skip_frames_window)
        ret, frame = cap.read()

        if not ret:
            st.warning(f"⚠️ Warning: Frame {frame_cntr} could not be read!")
            break

        frame = crop_face_center(frame)
        if frame is None:
            st.warning(f"⚠️ Warning: No face detected in frame {frame_cntr}")
            continue

        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
        frames.append(frame)

    cap.release()
    return np.array(frames)

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def crop_face_center(frame):
    if frame is None:
        st.error("❌ Error: Received an empty frame!")
        return None

    face_locations = face_recognition.face_locations(frame)
    
    if len(face_locations) == 0:
        st.warning("⚠️ Warning: No face detected in frame.")
        return None

    top, right, bottom, left = face_locations[0]
    face_image = frame[top:bottom, left:right]
    return face_image

# --- Streamlit App ---
def main():
    st.title("Deepfake Detection")
    st.write("Select a media type (image, audio, or video) for deepfake detection.")

    option = st.selectbox(
        "Select a media type for detection:",
        ("Image", "Audio", "Video")
    )

    if option == "Image":
        # Image Deepfake Detection
        st.header("Deepfake Image Detection")
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            model, device = load_image_model()
            img_tensor = transform_image(image).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                prediction = torch.round(output.squeeze()).item()
            result = "Fake" if prediction == 0 else "Real"
            st.write(f"**Prediction:** {result}")

    elif option == "Audio":
        # Audio Deepfake Detection
        st.header("Deepfake Audio Detection")
        uploaded_file = st.file_uploader("Upload an audio file...", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            model = load_audio_model()
            if model is not None:
                features = extract_features_from_audio(uploaded_file.getvalue())
                if features is not None:
                    prediction = model.predict(features)
                    confidence = prediction[0][0]
                    is_deepfake = confidence > 0.5
                    st.subheader('Detection Results')
                    if is_deepfake:
                        st.error(f'🚨 Deepfake Detected (Confidence: {confidence*100:.2f}%)')
                    else:
                        st.success(f'✅ Real Audio (Confidence: {(1-confidence)*100:.2f}%)')
                    st.progress(float(confidence))

    elif option == "Video":
        # Video Deepfake Detection
        st.header("Deepfake Video Detection")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            filename = secure_filename(uploaded_file.name)
            file_path = os.path.join("static/uploads/", filename)

            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("Video successfully uploaded!")
            st.video(file_path)

            # Perform prediction
            if st.button("Predict Deepfake"):
                prediction = sequence_prediction(file_path)
                st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
