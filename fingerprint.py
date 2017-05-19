from multiprocessing import Pool
from skimage.measure import block_reduce
import numpy as np
import librosa
import os

window = np.hanning(1024)


def main():
    samples = np.load(os.path.join(os.getcwd(), 'samples.npy'))
    with Pool() as p:
        fingerprints = p.map(job, samples)
    fingerprints = np.asarray(fingerprints).astype(np.float32)
    np.save(os.path.join(os.getcwd(), 'fingerprints.npy'), fingerprints)


def job(y):
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

if __name__ == "__main__":
    main()
