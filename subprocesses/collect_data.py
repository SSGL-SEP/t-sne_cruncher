from typing import Tuple

import numpy as np
import scipy.io.wavfile as wf


def load_sample(file_path: str, max_duration: int = 0) -> Tuple[str, np.ndarray, int, int]:
    """
    Load wave file data into numpy array
    :param file_path: Path to .wav file.
    :type file_path: str
    :param max_duration: Maximum length of audio to load. 0 to load entire sample.
    :type max_duration: int
    :return: Tuple containing file path, numpy array, sample length and samlpe rate.
    :rtype: Tuple[str, numpy.ndarray, int, int]
    """
    sr, audio = wf.read(file_path)
    max_samples = (sr * max_duration) // 1000
    if len(audio.shape) > 1:
        audio = np.asarray([x[0] for x in audio])
    if max_duration and len(audio) > max_samples:
        audio = np.resize(audio, max_samples)
    return file_path, audio, len(audio), sr
