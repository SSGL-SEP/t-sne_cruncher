"""This module provides functionality to create numpy data arrays form wave files."""
import numpy as np
import scipy.io.wavfile as wf


def load_sample(file_path: str, max_duration: int):
    sr, audio = wf.read(file_path)
    max_samples = (sr * max_duration) // 1000
    if len(audio.shape) > 1:
        audio = np.asarray([x[0] for x in audio])
    if max_duration and len(audio) > max_samples:
        audio = np.resize(audio, max_samples)
    return file_path, audio, len(audio), sr
