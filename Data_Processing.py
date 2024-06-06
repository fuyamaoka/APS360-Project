from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

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

# Set up the folders - ensure that these paths/folders do not already exist
try:
    os.mkdir('/content/drive/MyDrive/Datasets/Combined_Data')
    for instrument in instrument_list:
        os.mkdir('/content/drive/MyDrive/Datasets/Combined_Data/' + instrument)
except:
    pass

# Put the training data into the customized folders
try:
    for instrument in instrument_list:
        if instrument != 'pol':
            folder_path = '/content/drive/MyDrive/Datasets/IRMAS-TrainingData/' + instrument
            for file_name in os.listdir(folder_path):
                src = '/content/drive/MyDrive/Datasets/IRMAS-TrainingData/' + instrument + '/' + file_name
                dst = '/content/drive/MyDrive/Datasets/Combined_Data/' + instrument + '/' + file_name
                os.replace(src, dst)

    shutil.rmtree('/content/drive/MyDrive/Datasets/IRMAS-TrainingData')
except:
    pass

# Put the testing data into the customized folders
folder_paths = ['/content/drive/MyDrive/Datasets/IRMAS-TestingData-Part1/Part1',
                '/content/drive/MyDrive/Datasets/IRMAS-TestingData-Part2/IRTestingData-Part2',
                '/content/drive/MyDrive/Datasets/IRMAS-TestingData-Part3/Part3']

for folder_path in folder_paths:
    for file_name in os.listdir(folder_path):
        file_name = file_name[:-4]
        try:
            with open(folder_path + '/' + file_name + '.txt') as file:
                instruments = file.read().split("\t\n")[:-1]
                in_instrument_list = True
                for instrument in instruments:
                    if instrument not in instrument_list:
                        in_instrument_list = False
                if in_instrument_list:
                    if len(instruments) == 1:
                        try:
                            src = folder_path + '/' + file_name + '.wav'
                            dst = '/content/drive/MyDrive/Datasets/Combined_Data/' + instruments[0] + '/' + file_name + '.wav'
                            os.replace(src, dst)
                            os.remove(folder_path + '/' + file_name + '.txt')
                        except:
                            os.remove(folder_path + '/' + file_name + '.txt')
                    else:
                        src = folder_path + '/' + file_name + '.wav'
                        dst = '/content/drive/MyDrive/Datasets/Combined_Data/pol/' + file_name + '.wav'
                        os.replace(src, dst)

                        src = folder_path + '/' + file_name + '.txt'
                        dst = '/content/drive/MyDrive/Datasets/Combined_Data/pol/' + file_name + '.txt'
                        os.replace(src, dst)
                else:
                    pass
        except:
            pass

    shutil.rmtree(folder_path)

remaining_folders = ['/content/drive/MyDrive/Datasets/IRMAS-TestingData-Part1',
                '/content/drive/MyDrive/Datasets/IRMAS-TestingData-Part2',
                '/content/drive/MyDrive/Datasets/IRMAS-TestingData-Part3']

for remaining_folder in remaining_folders:
    shutil.rmtree(remaining_folder)

import librosa
import numpy as np
import os

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

import torch

instrument_list = ['cel', 'cla', 'flu', 'gac', 'pia', 'sax', 'tru', 'vio', 'pol']
target_size = (128, 862) # Target size of the spectrograms - target_size[1] is set to roughly 20 seconds of time in each recording
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
                inputs.append(mel_spec_padded)
            elif mel_spec.shape[1] > target_size[1]:
                mel_spec_trimmed = mel_spec[:, :target_size[1]]
                inputs.append(mel_spec_trimmed)
            else:
                inputs.append(mel_spec)

        one_hot_label = []
        ground_truth_instruments = []
        if instrument != 'pol':
            ground_truth_instruments.append(instrument)
        else:
            with open(folder_path + '/' + file_name[:-4] + '.txt') as file:
                instruments = file.read().split("\t\n")[:-1]
                for inst in instruments:
                    ground_truth_instruments.append(inst)
        for i in range(len(instrument_list) - 1):
            if instrument_list[i] in ground_truth_instruments:
                one_hot_label.append(1)
            else:
                one_hot_label.append(0)

        ground_truth_labels.append(one_hot_label)

X_inputs, y_labels = torch.tensor(np.array(inputs)), torch.tensor(np.array(ground_truth_labels))
