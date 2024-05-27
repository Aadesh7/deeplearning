import os

import matplotlib.pyplot as plt

import pandas as pd
import librosa
import librosa.display

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_path = './musicData'
# y = sound, sequence of vibrations in varying pressure strengths
# sr = sample rate, number of samples of audio carried per second
y, sr = librosa.load(f'{data_path}/genres_original/reggae/reggae.00037.wav')
y_shape = np.shape(y)

print('y:', y, '\n')
print('y shape:', y_shape, '\n')
print('Sample rate:', sr, '\n')

print('Check Len of Audio:', y_shape[0]/sr)

# understand spectrograms

# Default FFT window size and hop length
n_fft = 2048
hop_length = 512

y, sr = librosa.load(f'{data_path}/genres_original/metal/metal.00037.wav')
y, _ = librosa.effects.trim(y) # trim leading and trailing silence from audio signal

S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length) # computes the mel-scaled spectrogram of the audio signal y

## converts the amplitude of the mel spectrogram S to a decibel (dB) scale, which is a logarithmic scale more suitable for human perception. 
## ref=np.max sets the reference value to the maximum amplitude in the spectrogram, scaling the dB values accordingly.
S_to_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize = (16, 6))
librosa.display.specshow(S_to_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')
plt.colorbar()
plt.title("Metal Mel Spectrogram", fontsize=23)
plt.show()



