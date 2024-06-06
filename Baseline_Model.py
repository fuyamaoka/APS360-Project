from google.colab import drive
drive.mount('/content/drive')

import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def extract_mel_spectrogram(file_path):
    # Default spectrogram settings:
    sampling_rate = 22050
    n_mel_bands = 128
    fft_window_size = 2048
    hop_length = 512

    # Load the audio file
    y, sr = librosa.load(file_path, sr=sampling_rate)  # y is the audio loaded as an array to represent its time domain
    
    # Compute the Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel_bands, n_fft=fft_window_size, hop_length=hop_length)
    
    # Convert the power spectrogram to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

# cel -> cello
# cla -> clarinet
# flu -> flute
# gac -> acoustic guitar
# org -> organ
# pia -> piano
# sax -> saxophone
# tru -> trumpet
# vio -> violin
# pol -> polyphonic music

instrument_list = ['cel', 'cla', 'flu', 'gac', 'pia', 'sax', 'tru', 'vio', 'pol']
target_size = (128, 431) # Target size of the spectrograms - target_size[1] is set to roughly 10 seconds of time in each recording
inputs = []
ground_truth_labels = []
for instrument in instrument_list:
    folder_path = '/content/drive/MyDrive/Datasets/Combined_Data/' + instrument
    for file_name in os.listdir(folder_path):
            file_type = file_name[-4:]
            if file_type == '.txt':
                continue
            else:
                file_path = folder_path + '/' + file_name
                mel_spec = extract_mel_spectrogram(file_path)

                # Ensure spectrograms are all of the same shape
                if mel_spec.shape[1] < target_size[1]:
                    pad_width = target_size[1] - mel_spec.shape[1]
                    mel_spec_padded = np.pad(mel_spec, ((0, 0), (0, pad_width)))
                    inputs.append(mel_spec_padded.flatten())
                elif mel_spec.shape[1] > target_size[1]:
                    mel_spec_trimmed = mel_spec[:, :target_size[1]]
                    inputs.append(mel_spec_trimmed.flatten())
                else:
                    inputs.append(mel_spec.flatten())
                ground_truth_labels.append(instrument)
    np.array(inputs)

X_inputs, y_labels = np.array(inputs), np.array(ground_truth_labels)

X_train, X_test, y_train, y_test = train_test_split(X_inputs, y_labels, test_size=0.2)

# Initialize the model
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=instrument_list))
