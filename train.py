import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Sequential

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, target_size=(5, 5)):
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        processed_images.append(resized_img)
    return np.array(processed_images)

def generate_video(frames, output_file, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# Step 1: Prepare the Data
image_folder = 'output_frames'
images = load_images_from_folder(image_folder)
processed_images = preprocess_images(images)

# Step 2: Define the Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=processed_images.shape[1:]),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    TimeDistributed(LSTM(256, return_sequences=True)),
    TimeDistributed(Dense(3, activation='sigmoid'))
])

# Step 3: Train the Model
# (You may need to split your data into training and validation sets and define loss and optimizer)

# Step 4: Generate Videos
# (Use the trained model to generate videos from sequences of images)
# For demonstration, let's generate a video with the same frames repeatedly
generated_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in processed_images] * 120  # Repeat frames to create a longer video
generate_video(generated_frames, 'generated_video.mp4')
