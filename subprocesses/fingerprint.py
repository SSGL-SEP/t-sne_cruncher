import os
from multiprocessing import Pool
import numpy as np
import librosa
from skimage.measure import block_reduce

window = np.hanning(1024)


def fingerprint_from_file_data(source_npy: str = (os.path.join(os.getcwd(), 'samples.npy')),
                               target_npy_file: str = None):
    """
    Read data from a specified .npy file and create a fingerprint.
    
    :param source_npy: Source .npy file to read from
    :type source_npy: str
    :param target_npy_file: Optional .npy target file to write results to 
    :type target_npy_file: str
    :return: Three-dimensional numpy array with audio fingerprints.
    :rtype: numpy.ndarray
    """
    samples = np.load(source_npy)
    with Pool() as p:
        fingerprints = p.map(fingerprint_form_data, samples)
    fingerprints = np.asarray(fingerprints).astype(np.float32)
    if target_npy_file:
        np.save(target_npy_file, fingerprints)
    return fingerprints


def fingerprint_form_data(y: np.ndarray):
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


def mfcc_fingerprint(tup: tuple):
    """

    :param tup: tuple containing audio data and sample rate
    :type tup: tuple(numpy.ndarray, int)
    :return: mfcc fingerprint
    :rtype: numpy.ndarray
    """
    return librosa.feature.mfcc(y=tup[0], sr=tup[1])

if __name__ == "__main__":
    fingerprint_from_file_data(target_npy_file=os.path.join(os.getcwd(), 'fingerprints.npy'))
