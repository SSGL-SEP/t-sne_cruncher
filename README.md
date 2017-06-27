
[![Code Climate](https://codeclimate.com/github/SSGL-SEP/t-sne_cruncher/badges/gpa.svg)](https://codeclimate.com/github/SSGL-SEP/t-sne_cruncher) [![Build Status](https://travis-ci.org/SSGL-SEP/t-sne_cruncher.svg?branch=master)](https://travis-ci.org/SSGL-SEP/t-sne_cruncher)


# Audio analyzer for [Speech explorer](https://github.com/SSGL-SEP/speech_explorer)

Typically reads data from .wav files and outputs json data used by [Speech explorer](https://github.com/SSGL-SEP/speech_explorer)

## Environment

For best results use python 3.6 or newer. Fingerprinting will work on any python 3 system but the tag order in the json file will be undefined and some features in the tests will not work.

### The following packages are required to run `cruncher.py`

* [numpy](https://pypi.python.org/pypi/numpy/)
* [scipy](https://pypi.python.org/pypi/scipy/)
* [matplotli](https://pypi.python.org/pypi/matplotlib/)
* [librosa](https://pypi.python.org/pypi/librosa/)
* [scikit-image](https://pypi.python.org/pypi/scikit-image/)
* [scikit-learn](https://pypi.python.org/pypi/scikit-learn/)

### The following are needed for running test coverage calculations

* [coverage](https://pypi.python.org/pypi/coverage/)
* [codecov](https://pypi.python.org/pypi/codecov/)

All of these are defined in the `requirements.txt` and can typically be installed with `pip3 install -r requirements.txt`

Additionally for generating the audio mp3 blob using `audio_concatenator.py` the ffmpeg program and python library `ffmpy` are needed. (Typically `pip3 install ffmpy`)

## `Cruncher.py` Usage

Run with `python3.6 cruncher.py -h` to see help on command line.

### Command line arguments:

#### -f / --input_folder

The folder to read audio files from. The default is to read from the directory where the script is executed. Note that the data will be read recursively.

#### -p / --perplexity

Perplexity / perplexities to use when running dimensionality reduction with t-SNE. List of space-delimited integers.
