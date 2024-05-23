import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

import warnings
warnings.filterwarnings('ignore')


data_path = './musicData'
# print(list(os.listdir(f'{data_path}/genres_original/')))

## understand audio

# # y = sound, sequence of vibrations in varying pressure strengths
# # sr = sample rate, number of samples of audio carried per second
# y, sr = librosa.load(f'{data_path}/genres_original/reggae/reggae.00037.wav')
# y_shape = np.shape(y)

# print('y:', y, '\n')
# print('y shape:', y_shape, '\n')
# print('Sample rate:', sr, '\n')

# print('Check Len of Audio:', y_shape[0]/sr)

####################################################################################

# # understand spectrograms

# # Default FFT window size and hop length
# n_fft = 2048
# hop_length = 512

# y, sr = librosa.load(f'{data_path}/genres_original/metal/metal.00037.wav')
# y, _ = librosa.effects.trim(y) # trim leading and trailing silence from audio signal

# S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length) # computes the mel-scaled spectrogram of the audio signal y

# ## converts the amplitude of the mel spectrogram S to a decibel (dB) scale, which is a logarithmic scale more suitable for human perception. 
# ## ref=np.max sets the reference value to the maximum amplitude in the spectrogram, scaling the dB values accordingly.
# S_to_DB = librosa.amplitude_to_db(S, ref=np.max)

# plt.figure(figsize = (16, 6))
# librosa.display.specshow(S_to_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')
# plt.colorbar()
# plt.title("Metal Mel Spectrogram", fontsize=23)
# plt.show()

####################################################################################

## helper functions

n_fft = 2048
hop_length = 512

def extract_mel_spectrogram(file_path, n_fft=n_fft, hop_length=hop_length):
    y, sr = librosa.load(file_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return mel_spec

def preprocess_data(file_paths, labels, n_fft=n_fft, hop_length=hop_length, resize_shape=(128, 128)):
    X = []
    y = []

    for file_path, label in zip(file_paths, labels):
        print(file_path, '\n')
        mel_spec = extract_mel_spectrogram(file_path, n_fft=n_fft, hop_length=hop_length)

        # Resize the mel spectrogram using padding or truncation
        if mel_spec.shape[1] < resize_shape[1]:
            pad_width = resize_shape[1] - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec.shape[1] > resize_shape[1]:
            mel_spec = mel_spec[:, :resize_shape[1]]

        X.append(mel_spec)
        y.append(label)
    
    X = np.array(X)
    X = np.expand_dims(X, axis=-1) # adds a channel dimension
    y = np.array(y)
    
    return X, y 

## load

genres = os.listdir(f'{data_path}/genres_original/')

file_paths = []
labels = []

for genre in genres:
    genre_dir = os.path.join(f'{data_path}/genres_original/', genre)
    for root, _, files in os.walk(genre_dir):
        for file in files:
            if file.endswith('.wav'):
                file_paths.append(os.path.join(root, file))
                labels.append(genre)

X, y = preprocess_data(file_paths, labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

input_shape = X_train[0].shape
num_classes = len(np.unique(y_train))
y_train_one_hot = to_categorical(y_train_encoded, num_classes)
y_test_one_hot = to_categorical(y_test_encoded, num_classes)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=L2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train_one_hot, epochs=30, batch_size=16, validation_data=(X_test, y_test_one_hot))
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
print("Test accuracy:", test_acc)
