import os
import librosa
import re
from step2 import data_parser

def parse_audio_data_with_mfcc(data_dir = '/home/arismarkog/patrec1/pr_lab2_data/digits'):
    
    wav_data, speakers, digits = data_parser(data_dir)

    mfcc_features = []
    delta_features = []
    delta_delta_features = []
        
    for wav in wav_data:
        
        sr = 16000  #sample rate
        
        # Calculate MFCCs with 13 coefficients, 25ms window length, and 10ms hop length
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13, 
                                    n_fft=int(sr * 0.025), hop_length=int(sr * 0.01))
        mfcc_features.append(mfcc)
        
        # Calculate delta and delta-delta features
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        delta_features.append(delta)
        delta_delta_features.append(delta_delta)
    
    return wav_data, speakers, digits, mfcc_features, delta_features, delta_delta_features