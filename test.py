import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import L2
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Constants
audio_duration = 30
sample_rate = 22050
window_size = 2048
hop_size = 512
mel_bins = 128
data_path = './musicData'
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

# Load Label Encoder
label_encoder = LabelEncoder()
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
label_encoder.fit(genres)

# Build Model
input_shape = (128, 128, 1)
num_classes = len(genres)

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=L2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.3),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(128, activation='relu', kernel_regularizer=L2(0.001)),
    layers.Dropout(0.2),

    layers.Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
    layers.Dropout(0.1),

    layers.Dense(num_classes, activation='softmax')
])

# Load model weights
model.load_weights('./music_genre_weights.weights.h5')

def classify_audio(file_path):
    mel_spec = extract_mel_spectrogram(file_path)

    # Ensure the Mel spectrogram has the shape (128, 128, 1)
    if mel_spec.shape[0] < 128:
        pad_width = 128 - mel_spec.shape[0]
        mel_spec = np.pad(mel_spec, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
    elif mel_spec.shape[0] > 128:
        mel_spec = mel_spec[:128, :, :]

    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension
    predictions = model.predict(mel_spec)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_genre = label_encoder.inverse_transform(predicted_class)
    return predicted_genre[0]

# Example usage
new_audio_file_path = f'{data_path}/genres_original/reggae/reggae.00037.wav'
predicted_genre = classify_audio(new_audio_file_path)
print(f"The predicted genre is: {predicted_genre}")
