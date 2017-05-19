import numpy as np
import subprocess as sp
import os


def ffmpeg_load_audio(fn, in_type=np.int16):
    command = [
        "ffmpeg",
        "-i", fn,
        "-f", "s161e",
        "-acodec", "pcm_s161e",
        "-ar", "44100",
        "-ac", "1",
        "-"]
    p = sp.Popen(command, stdout=sp.PIPE, stderr=open(os.devnull, 'w'), bufsize=4096, close_fds=True)
    bytes_per_sample = np.dtype(in_type).itemsize
    chunk_size = bytes_per_sample * 44100
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=np.int16)
    if audio.size == 0:
        return audio.astype(np.int16), 44100
    return audio, 44100
