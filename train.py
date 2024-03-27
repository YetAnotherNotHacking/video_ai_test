import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.layers import Conv2D, LSTM, TimeDistributed, Dense, Lambda, Reshape
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np


def load_images_from_folder(folder, target_size=(5, 5)):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return (np.array(images), np.array(sorted(os.listdir(folder)))[-target_size[1]:])

def preprocess_images(images, target_size=(5, 5), normalize=True):
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        if normalize:
            resized_img = (resized_img - 128.0) / 128.0
        processed_images.append(resized_img)
    return np.array(processed_images)

def generate_video(frames, output_file, fps=120):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame.astype(np.uint8))
    out.release()

def generate_images(frames, output_folder, fps=120):
    for i, frame in enumerate(frames):
        filename = f'{i}.png'
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame.astype(np.uint8))

def create_model(input_shape, nb_frames=5, nb_filters=32, kernel_size=3, final_activation='sigmoid'):
    model = Sequential()
    model.add(Lambda(lambda x: x[:, :, :, :nb_frames], input_shape=input_shape))
    model.add(Conv2D(nb_filters, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(Conv2D(nb_filters, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(Conv2D(nb_filters, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(Conv2D(nb_filters, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1] * input_shape[2], activation='relu')))
    model.add(Reshape(input_shape[1:]))
    model.add(Conv2D(1, kernel_size=1, activation=final_activation))
    return model

# Step 1: Prepare the Data
image_folder = 'ba_frames'
(images, filenames) = load_images_from_folder(image_folder, target_size=(5, 5))

# Step 2: Define the Model Architecture
input_shape = (None, 5, 5)



model = tf.keras.Sequential([
    tf.keras.layers.LSTM(8), # use return_sequences=True to solve the problem
    tf.keras.layers.LSTM(8),
    tf.keras.layers.Dense(1, activation='sigmoid')
])



# Step 3: Train the Model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

x_train = images[:-1]
x_test = images[-1]
y_train = images[1:]

x_train = preprocess_images(x_train, target_size=(5, 5))
y_train = preprocess_images(y_train, target_size=(5, 5))
x_test = preprocess_images(x_test, target_size=(5, 5))

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

model.fit(x_train, y_train, validation_data=(x_test, y_train), epochs=10, batch_size=16, callbacks=callbacks)

# Step 4: Generate Videos

generated_frames = []
for i in range(100):
    x_pred = np.expand_dims(x_test, axis=0)
    pred = model.predict(x_pred)
    generated_frames.append(pred.reshape(input_shape[1:]))

generate_images(generated_frames, 'ba_frames')
generate_video(generated_frames, 'generated_video.mp4')
