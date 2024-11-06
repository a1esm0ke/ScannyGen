import numpy as np
import librosa

def load_song(song_path):
    x, sr = librosa.load(song_path)
    print("LOAD COMPLETED")
    return [x, sr]

def beat(song):
    tempo, beats = librosa.beat.beat_track(y=song[0], sr=song[1])
    return beats[-1] / tempo if tempo != 0 else 0  # Aggiunto controllo per evitare divisione per zero

def tempo(song):
    tempo, _ = librosa.beat.beat_track(y=song[0], sr=song[1])
    return tempo

def chroma_stft(song):
    y, sr = song[0], song[1]
    stft = librosa.feature.chroma_stft(y=y, sr=sr)
    out = np.sum(stft)
    return out / np.size(stft) if np.size(stft) > 0 else 0  # Uso di np.sum e np.size per efficienza e leggibilità

def rmse(song):
    y = song[0]
    rmse = librosa.feature.rms(y=y)
    return np.mean(rmse)

def spectral_centroid(song):
    y, sr = song[0], song[1]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)

def spectral_bandwidth(song):
    y, sr = song[0], song[1]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(bandwidth)

def spectral_rolloff(song):
    y, sr = song[0], song[1]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return np.mean(rolloff)

def zero_crossing_rate(song):
    y = song[0]
    rate = librosa.feature.zero_crossing_rate(y=y)
    return np.mean(rate)

def mfcc(song):
    y, sr = song[0], song[1]
    return librosa.feature.mfcc(y=y, sr=sr)

def get_song_feature(song):
    features = np.ndarray(28)
    print("Processing features, please wait a few seconds...")
    features[0] = tempo(song)
    features[1] = beat(song)
    features[2] = chroma_stft(song)
    features[3] = rmse(song)
    features[4] = spectral_centroid(song)
    features[5] = spectral_bandwidth(song)
    features[6] = spectral_rolloff(song)
    features[7] = zero_crossing_rate(song)
    mfcc_features = librosa.feature.mfcc(y=song[0], sr=song[1])
    for i, value in enumerate(mfcc_features, start=8):
        features[i] = np.mean(value)
    return features
