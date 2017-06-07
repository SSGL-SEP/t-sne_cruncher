"""This module provides functionality to create numpy data arrays form wave files."""
import os
from multiprocessing import Pool
import numpy as np
import scipy.io.wavfile as wf
from utils import *


def collect_data(max_duration: int = 500, source_folder: str = os.getcwd(),
                 extensions: list = None, target_file: str = None):
    """
    Collect audio data from all files in a given folder and returns a two-dimensional n*m numpy array,
    where n is the number of files found and m is the number of samples loaded per file.
    
    :param max_duration: Maximum allowed duration of samples in ms 
    :type max_duration: int
    :param source_folder: folder to load audio files from
    :type source_folder: str
    :param extensions: List of file extensions to load. Currently only .wav is supported
    :type extensions: List[str]
    :param target_file: Optional file to save npy data to.
    :type target_file: str
    :return: Two-dimensional numpy array
    :rtype: numpy.ndarray
    """
    if extensions is None:
        extensions = ['.wav']
    files = list(all_files(source_folder, extensions))
    with Pool() as p:
        results = p.map(load_sample, [(max_duration, f) for f in files])
    print("processed", len(results), "samples")
    if target_file:
        samples = [x[1] for x in filter(None, results)]
        samples = np.asarray(samples)
        np.save(target_file, samples)
    return results


def load_sample(tup: (int, str)):
    """
    Return a (possibly truncated) numpy array with the audio data of the specified .wav file. 
    Note that only the first channel of a multichannel .wav will be read.
    
    :param tup: Tuple containing the maximum length allowed and filepath to the .wav to load.
    :type tup: (int, str)
    :return: Tuple containing the filepath of the loaded .wav, the audio data and the length of the sample
    :rtype: (str, numpy.ndarray, int)
    """
    fn = tup[1]
    max_duration = tup[0]
    sr, audio = wf.read(fn)
    max_samples = (sr * max_duration) // 1000
    if len(audio.shape) > 1:
        audio = np.asarray([x[0] for x in audio])
    if len(audio) > max_samples:
        audio = np.resize(audio, max_samples)
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
    print(fn, " loaded.")
    return fn, audio, len(audio), sr


if __name__ == "__main__":
    # If run as a script: Saves sample data from all .wav files to 'samples.npy'
    collect_data(target_file=os.path.join(os.getcwd(), 'samples.npy'))
