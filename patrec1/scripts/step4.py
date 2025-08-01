from step3 import parse_audio_data_with_mfcc
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd


def extract_mfcc_histograms(mfcc_features, digits, n1, n2):
    mfcc1_n1 = []
    mfcc2_n1 = []
    mfcc1_n2 = []
    mfcc2_n2 = []

    for i in range(len(digits)):
        if digits[i] == n1:
            mfcc1_n1.append(mfcc_features[i][0])  # First MFCC
            mfcc2_n1.append(mfcc_features[i][1])  # Second MFCC
        elif digits[i] == n2:
            mfcc1_n2.append(mfcc_features[i][0])  # First MFCC
            mfcc2_n2.append(mfcc_features[i][1])  # Second MFCC
    
    return mfcc1_n1, mfcc2_n1, mfcc1_n2, mfcc2_n2

def plot_histograms(mfcc1_n1, mfcc2_n1, mfcc1_n2, mfcc2_n2, n1, n2):

    def flatten_list(nested_list):
        flat_list = []
        for sublist in nested_list:
            for item in sublist:
                flat_list.append(item)
        return flat_list
    
    mfcc1_n1_flat = flatten_list(mfcc1_n1)
    mfcc2_n1_flat = flatten_list(mfcc2_n1)
    mfcc1_n2_flat = flatten_list(mfcc1_n2)
    mfcc2_n2_flat = flatten_list(mfcc2_n2)



    # Plot histograms
    plt.figure(figsize=(12, 6))
   
    # First row, first column
    plt.subplot(2, 2, 1)
    plt.hist(mfcc1_n1_flat, bins=20, label=f'1st MFCC - {n1}')
    plt.xlabel('MFCC Value')
    plt.ylabel('Frequency')
    plt.legend()

    # First row, second column
    plt.subplot(2, 2, 2)
    plt.hist(mfcc2_n1_flat, bins=20, label=f'2nd MFCC - {n1}')
    plt.xlabel('MFCC Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Second row, first column
    plt.subplot(2, 2, 3)
    plt.hist(mfcc1_n2_flat, bins=20, label=f'1st MFCC - {n2}')
    plt.xlabel('MFCC Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Second row, second column
    plt.subplot(2, 2, 4)
    plt.hist(mfcc2_n2_flat, bins=20, label=f'2nd MFCC - "{n2}"')
    plt.xlabel('MFCC Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig("images/mfcc_hist.png")
    #plt.show()
   

def extract_mfscs(wav, sr):
    return librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=int(sr * 0.01), n_fft=int(sr * 0.025), n_mels=13)
    

def plot_mfscs_corr(mfscs_1_8, mfscs_2_8, mfscs_1_5, mfscs_2_5):
    fig = plt.figure(figsize=(15,12))

    ax1 = fig.add_subplot(2, 2, 1)
    mfsc_df_1_8 = pd.DataFrame.from_records(mfscs_1_8.T)
    ax1.title.set_text(f"MFSCs Speaker 1 - Digit {n1}")
    plt.imshow(mfsc_df_1_8.corr())

    ax2 = fig.add_subplot(2, 2, 2)
    mfsc_df_2_8 = pd.DataFrame.from_records(mfscs_2_8.T)
    ax2.title.set_text(f"MFSCs Speaker 2 - Digit {n1}")
    plt.imshow(mfsc_df_2_8.corr())

    ax3 = fig.add_subplot(2, 2, 3)
    mfcc_df_1_5 = pd.DataFrame.from_records(mfscs_1_5.T)
    ax3.title.set_text(f"MFSCs Speaker 1 - Digit {n2}")
    plt.imshow(mfcc_df_1_5.corr())

    ax4 = fig.add_subplot(2, 2, 4)
    mfsc_df_2_5 = pd.DataFrame.from_records(mfscs_2_5.T)
    ax4.title.set_text(f"MFSCs Speaker 2 - Digit {n2}")
    plt.imshow(mfsc_df_2_5.corr())

    plt.tight_layout()
    plt.savefig("images/mfsc_corr.png")

def plot_mfccs_corr(mfccs_1_8, mfccs_2_8, mfccs_1_5, mfccs_2_5):
    fig = plt.figure(figsize=(15,12))

    ax1 = fig.add_subplot(2, 2, 1)
    mfcc_df_1_8 = pd.DataFrame.from_records(mfccs_1_8.T)
    ax1.title.set_text(f"MFCCs Speaker 1 - Digit {n1}")
    plt.imshow(mfcc_df_1_8.corr())

    ax2 = fig.add_subplot(2, 2, 2)
    mfcc_df_2_8 = pd.DataFrame.from_records(mfccs_2_8.T)
    ax2.title.set_text(f"MFCCs Speaker 2 - Digit {n1}")
    plt.imshow(mfcc_df_2_8.corr())

    ax3 = fig.add_subplot(2, 2, 3)
    mfcc_df_1_5 = pd.DataFrame.from_records(mfccs_1_5.T)
    ax3.title.set_text(f"MFCCs Speaker 1 - Digit {n2}")
    plt.imshow(mfcc_df_1_5.corr())

    ax4 = fig.add_subplot(2, 2, 4)
    mfcc_df_2_5 = pd.DataFrame.from_records(mfccs_2_5.T)
    ax4.title.set_text(f"MFSCs Speaker 2 - Digit {n2}")
    plt.imshow(mfcc_df_2_5.corr())

    plt.tight_layout()
    plt.savefig("images/mfcc_corr.png")

# Example usage
n1 = 8
n2 = 5

wav_data, speakers, digits, mfcc_features, _, _ = parse_audio_data_with_mfcc()
mfcc1_n1, mfcc2_n1, mfcc1_n2, mfcc2_n2 = extract_mfcc_histograms(mfcc_features=mfcc_features, digits=digits, n1=n1, n2=n2)
plot_histograms(mfcc1_n1=mfcc1_n1, mfcc2_n1=mfcc2_n1, mfcc1_n2=mfcc1_n2, mfcc2_n2=mfcc2_n2, n1=n1, n2=n2)

index_1_8 = -1
index_2_8 = -1
index_1_5 = -1
index_2_5 = -1


for i in range(len(wav_data)):
    if speakers[i] == 1 and digits[i] == 8:
        index_1_8 = i
    elif speakers[i] == 2 and digits[i] == 8:
        index_2_8 = i
    elif speakers[i] == 1 and digits[i] == 5:
        index_1_5 = i
    elif speakers[i] == 2 and digits[i] == 5:
        index_2_5 = i

sr = 16000
mfscs_1_8 = extract_mfscs(wav_data[index_1_8], sr)
mfscs_2_8 = extract_mfscs(wav_data[index_2_8], sr)
mfscs_1_5 = extract_mfscs(wav_data[index_1_5], sr)
mfscs_2_5 = extract_mfscs(wav_data[index_2_5], sr)

mfccs_1_8 = mfcc_features[index_1_8]
mfccs_2_8 = mfcc_features[index_2_8]
mfccs_1_5 = mfcc_features[index_1_5]
mfccs_2_5 = mfcc_features[index_2_5]



plot_mfscs_corr(mfscs_1_8, mfscs_2_8, mfscs_1_5, mfscs_2_5)
plot_mfccs_corr(mfccs_1_8, mfccs_2_8, mfccs_1_5, mfccs_2_5)