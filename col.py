from json import load

import numpy as np
import os
from multiprocessing import Pool
from utils import *
import scipy.io.wavfile as wf


def main():
    files = list(all_files(os.getcwd(), [".wav", ".mp3"]))
    with Pool() as p:
        results = p.map(load_sample, files)
    print("processed", len(results), "samples")
    valid = filter(None, results)
    names = [x[0] for x in valid]
    samples = [x[1] for x in valid]
    durations = [x[2] for x in valid]
    samples = np.asarray(samples)
    for i in names:
        print(i)
    np.save(os.path.join(os.getcwd(), "samples.npy"), samples)


def load_sample(fn):
    audio = wf.read(fn)[1]
    audio.resize(22050)
    return fn, audio, len(audio)


if __name__ == "__main__":
    main()
