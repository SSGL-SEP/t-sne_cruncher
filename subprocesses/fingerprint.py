import numpy as np
import librosa
from skimage.measure import block_reduce

window = np.hanning(1024)


def fft_fingerprint(y: np.ndarray, sr: int = None, size: int = None) -> np.ndarray:
    """
    Create fft fingerprint from input sample data
    :param y: Input samples
    :type y: numpy.ndarray
    :param sr: Sample rate. Not used.
    :type sr: int
    :param size: Samlpe lenth. Not used
    :type size: int
    :return: Fingerprint array
    :rtype: np.ndarray
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


def ms_fingerprint(data: np.ndarray, sr: int, size: int) -> np.ndarray:
    """
    Create mel spectrum fingerprint from input sample data
    :param data: Input samples
    :type data: numpy.ndarray
    :param sr: Sample rate
    :type sr: int
    :param size: Sample size
    :type size: int
    :return: Fingerprint array
    :rtype: numpy.ndarray
    """
    steps = 25
    hop = size//steps
    return librosa.feature.melspectrogram(y=data, sr=sr, hop_length=hop)


def tonnez_fingerprint(data: np.ndarray, sr: int, size: int) -> np.ndarray:
    """
    Generate tonal centroid features for input sample data.
    :param data: Input samples
    :type data: numpy.ndarray
    :param sr: Sample rate
    :type sr: int
    :param size: Sample size
    :type size: int
    :return: Fingerprint array
    :rtype: numpy.ndarray
    """
    return librosa.feature.tonnetz(y=data.astype(np.float64), sr=sr)


def chroma_fingerprint(data: np.ndarray, sr: int, size: int) -> np.ndarray:
    """
    Generate chromagram for input sample data.
    :param data: Input samples
    :type data: numpy.ndarray
    :param sr: Sample rate
    :type sr: int
    :param size: Sample size
    :type size: int
    :return: Fingerprint array
    :rtype: numpy.ndarray
    """
    steps = 25
    hop = size//steps
    return librosa.feature.chroma_stft(y=data, sr=sr, hop_length=hop)


def mfcc_fingerprint(data: np.ndarray, sr: int, size: int) -> np.ndarray:
    """
    Generate mel frequency chroma for input sample data.
    :param data: Input samples
    :type data: numpy.ndarray
    :param sr: Sample rate
    :type sr: int
    :param size: Sample size
    :type size: int
    :return: Fingerprint array
    :rtype: numpy.ndarray
    """
    steps = 25
    hop = size//steps
    return librosa.feature.mfcc(data, sr, hop_length=hop, n_mfcc=20)
