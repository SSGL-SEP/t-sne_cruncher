from json import load

import numpy as np
import os
from multiprocessing import Pool
from utils import *
import scipy.io.wavfile as wf


def collect_data(source_folder=os.getcwd(), extensions=None, target_file=None):
    if extensions is None:
        extensions = ['.wav', '.mp3']
    files = list(all_files(source_folder, extensions))
    with Pool() as p:
        results = p.map(load_sample, files)
    print("processed", len(results), "samples")
    valid = filter(None, results)
    names = [x[0] for x in valid]
    samples = [x[1] for x in valid]
    samples = np.asarray(samples)
    for i in names:
        print(i)
    if target_file:
        np.save(target_file, samples)
    return results


def load_sample(fn):
    audio = wf.read(fn)[1]
    audio.resize(22050)
    return fn, audio, len(audio)


if __name__ == "__main__":
    collect_data(target_file=os.path.join(os.getcwd(), 'samples.npy'))