# Environment

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
