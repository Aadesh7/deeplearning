import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
import warnings
warnings.filterwarnings('ignore')

data_path = './musicData'

# Constants
audio_duration = 30
sample_rate = 22050
window_size = 2048
hop_size = 512
mel_bins = 128

frames_per_sec = sample_rate // hop_size
frames_num = frames_per_sec * audio_duration
audio_samples = int(sample_rate * audio_duration)

# Helper Functions

def normalize(audio):
    eps = 0.001
    if np.std(audio) != 0:
        audio = (audio - np.mean(audio)) / np.std(audio)
    else:
        audio = (audio - np.mean(audio)) / eps
    return audio

def read_audio(audio_path, target_fs=sample_rate):
    audio, fs = librosa.load(audio_path, sr=None)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

def compute_mel_spec(audio):
    stft_matrix = librosa.core.stft(y=audio, n_fft=window_size, hop_length=hop_size, window=np.hanning(window_size), center=True, dtype=np.complex64, pad_mode='reflect').T
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins).T
    mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, melW)
    logmel_spc = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
    logmel_spc = logmel_spc.astype(np.float32)
    logmel_spc = np.expand_dims(logmel_spc, axis=-1)  # Add channel dimension
    return logmel_spc

def extract_mel_spectrogram(file_path):
    y, sr = read_audio(file_path, target_fs=sample_rate)
    y = normalize(y)
    if len(y) < audio_samples:
        y = pad_truncate_sequence(y, audio_samples)
    elif len(y) > audio_samples:
        y = y[int((len(y) - audio_samples) / 2): int((len(y) + audio_samples) / 2)]
    mel_spec = compute_mel_spec(y)
    return mel_spec

def preprocess_data(file_paths, labels, resize_shape=(128, 128)):
    X = []
    y = []

    for file_path, label in zip(file_paths, labels):
        print(f"Processing file: {file_path}")
        mel_spec = extract_mel_spectrogram(file_path)

        if mel_spec.shape[1] < resize_shape[1]:
            pad_width = resize_shape[1] - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        elif mel_spec.shape[1] > resize_shape[1]:
            mel_spec = mel_spec[:, :resize_shape[1], :]

        X.append(mel_spec)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Shape of X before reshaping: {X.shape}")
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    print(f"Shape of X after reshaping: {X.shape}")
    
    return X, y 

# Load Data
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

# Build Model
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

# Train Model
model.fit(X_train, y_train_one_hot, epochs=15, batch_size=16, validation_data=(X_test, y_test_one_hot))
model.save_weights('music_genre_weights.weights.h5')
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
