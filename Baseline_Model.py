from google.colab import drive
drive.mount('/content/drive')

import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def extract_detailed_features(file_path, target_size=(128, 431)):
    # Default spectrogram settings
    sampling_rate = 22050
    n_mel_bands = 128
    fft_window_size = 2048
    hop_length = 512

    # Load the audio file
    y, sr = librosa.load(file_path, sr=sampling_rate)

    # Compute the Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel_bands, n_fft=fft_window_size, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Ensure Mel-spectrogram has consistent shape
    if mel_spec_db.shape[1] < target_size[1]:
        pad_width = target_size[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > target_size[1]:
        mel_spec_db = mel_spec_db[:, :target_size[1]]

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.shape[1] < target_size[1]:
        pad_width = target_size[1] - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > target_size[1]:
        mfccs = mfccs[:, :target_size[1]]

    # Extract delta and delta-delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Flatten the features and concatenate
    detailed_features = np.concatenate((mel_spec_db.flatten(), mfccs.flatten(), delta_mfccs.flatten(), delta2_mfccs.flatten()))

    return detailed_features

def extract_mfcc_means(file_path, sr=22050, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

instrument_list = ['cel', 'cla', 'flu', 'gac', 'pia', 'sax', 'tru', 'vio', 'pol']
inputs = []
ground_truth_labels = []

for instrument in instrument_list:
    folder_path = '/content/drive/MyDrive/Datasets/Combined_Data/' + instrument
    i = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):  # Assuming the files are .wav
            i += 1
            file_path = os.path.join(folder_path, file_name)

            detailed_features = extract_detailed_features(file_path)
            mfcc_mean = extract_mfcc_means(file_path)

            combined_features = np.concatenate((detailed_features, mfcc_mean))

            inputs.append(combined_features)
            ground_truth_labels.append(instrument)
        if i >= 750:
          break
    print(instrument, i)

X_inputs, y_labels = np.array(inputs), np.array(ground_truth_labels)

# Normalize the features
scaler = MinMaxScaler()
X_inputs_scaled = scaler.fit_transform(X_inputs)

# Handle imbalanced data with SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_inputs_scaled, y_labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled)

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
