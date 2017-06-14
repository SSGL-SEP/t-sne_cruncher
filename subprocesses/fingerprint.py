import numpy as np
import librosa
from skimage.measure import block_reduce

window = np.hanning(1024)


def fingerprint_form_data(y: np.ndarray, sr: int, size: int):
    """
    Create fingerprint data from audio in numpy array.
    
    :param y: The numpy array containing audio data
    :type y: numpy.ndarray
    :return: Two-dimensional numpy array containing the fingerprint.
    :rtype: numpy.ndarray
    """
    s = librosa.stft(y, 1024, 4096, window=window)
    amp = np.abs(s)
    amp = block_reduce(amp, (10, 1), func=np.mean)
    if amp.shape[1] < 32:
        amp = np.pad(amp, ((0, 0), (0, 32 - amp.shape[1])), 'constant')
    amp = amp[:32, :32]
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()
    amp = np.flipud(amp)
    return amp


def ms_fingerprint(data: np.ndarray, sr: int, size: int):
    steps = 25
    hop = size//steps
    return librosa.feature.melspectrogram(y=data, sr=sr, hop_length=hop)


def tonnez_fingerprint(data: np.ndarray, sr: int, size: int):
    return librosa.feature.tonnetz(y=data.astype(np.float64), sr=sr)


def chroma_fingerprint(data: np.ndarray, sr: int, size: int):
    steps = 25
    hop = size//steps
    return librosa.feature.chroma_stft(y=data, sr=sr, hop_length=hop)


def mfcc_fingerprint(data: np.ndarray, sr: int, size: int):
    return librosa.feature.mfcc(y=data, sr=sr)
