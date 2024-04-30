import librosa
import pandas as pd
import numpy as np

def extract_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    harmonic, percussive = librosa.effects.hpss(y)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr)

    feature_list = [np.mean(chroma_stft), np.var(chroma_stft),
                    np.mean(rms), np.var(rms),
                    np.mean(spectral_centroid), np.var(spectral_centroid),
                    np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
                    np.mean(rolloff), np.var(rolloff),
                    np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
                    np.mean(harmonic), np.var(harmonic),
                    np.mean(percussive), np.var(percussive),
                    tempo]

    for i in range(mfccs.shape[0]):
        feature_list.extend([np.mean(mfccs[i, :]), np.var(mfccs[i, :])])

    columns = ['chroma_stft_mean', 'chroma_stft_var',
               'rms_mean', 'rms_var',
               'spectral_centroid_mean', 'spectral_centroid_var',
               'spectral_bandwidth_mean', 'spectral_bandwidth_var',
               'rolloff_mean', 'rolloff_var',
               'zero_crossing_rate_mean', 'zero_crossing_rate_var',
               'harmony_mean', 'harmony_var',
               'perceptr_mean', 'perceptr_var',
               'tempo']

    for i in range(1, 21):
        columns.extend([f'mfcc{i}_mean', f'mfcc{i}_var'])

    df = pd.DataFrame([feature_list], columns=columns)
    return df


