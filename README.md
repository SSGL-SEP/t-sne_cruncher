
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

Perplexity / perplexities to use when running dimensionality reduction with t-SNE. List of space-delimited integers. The default value is 30.

#### -o / --output_file

The file to store json format output to for use with speech_explorer. The default is to store to `t_sne_##.json` to the folder the script is run from, where `##` is the perplexity used.

#### -r / --fingerprint_output

File to store output of feature extraction to. Output is handled by [numpy.save](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.save.html). The data is a list of tuples containing file name and a `numpy.ndarray` of values. By default fongerprint data will not be stored.

#### -m / --value_minimum

The minimum value to use when normalizing coordinate data for json output. The default minimum for speech_explorer coordinates is 0.

#### -x / --value_maximum

The maximum value to use when normalizing coordinate data for json output. The default maximum fo speech_explorer coordinates is 600.

#### -t / --plot_output

File to output pyplot of dimensionality reduction to. The perplexity will be appended to the file name to avoid overwriting. E.g. `plot_.png` will become `plot_30.png` if t-SNE is run with perplexity 30. By default no plot will be output.

#### -c / --collect_metadata

File to load sample metadata from. File format should be csv as defined in [formats.pdf](docs/formats.pdf). Ny default no metadata is collected.

#### -d / --duration

Maximum sample length to load in milliseconds. Longer samples will be truncated. A default value of 0 means that the entire sample will allways be loaded regardles of length.

#### -u / --unfilterables

Space separated list of tags that should be loaded but not added to filter lists.

#### -n / --data_set

Name of the dataset containing the samples.

#### -s / --sound_info

Name of or path to .json file that contains sample data. May or may not be used.

#### -a / --max_to_load

Maximum number of samples to load. Leave as 0 to load all found samples.

#### -b / --tags_to_ignore

Space delimited list of tags not to load while parsing metadata. By default the tags "waveform", "name", "filename" and "file name" are ignored in an attempt to not include the name of the sample as a tag.

#### -k / --reduction_method

Dimension reduction algorithm to use. At the time of writing supported values are "tsne" and "pca". By default dimensionality reduction is done using t-SNE.

#### -g / --fingerprint_method

Algorithm to use for feature extraction. At the time of writing supported values are "fft", "chroma", "ms", "mfcc" with "ms" or "mel spectrum" being the default.

#### -e / --fingerprint_input

Optional parameter for reading fingerprint data from file instead of generating the data at run time. The value should be the name of the file containing the data. Data format is defined by the **--format** parameter.

#### --format

Parameter to specify the file format for fingerprint data. Supported format are "npy", "tsv" and "csv" as defined in [formats.pdf](docs/formats.pdf)

#### --td

Flag to mark that diemnsionality reduction should be done to 2 dimensions instead of 3.

#### --colorby

Name of the tag that default coloration of the map should be based on.

#### --parallel

When running dimensionality reduction with t-SNE, this flag indicates that reductions with different perplexities can be run in parallel. This is not recommended unless the dataset is very limited or the system running the script has loads of idle memory.
