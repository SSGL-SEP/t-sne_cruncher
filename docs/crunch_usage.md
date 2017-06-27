# `Crunch.py` Usage

Run with `python3.6 cruncher.py -h` to see help on command line.

### Example

`python3.6 crunch.py -f phoneme/ -o mfcc_phoneme_.json -t mfcc_phoneme_.png -c phoneme/labels.csv -n phoneme -g mfcc --td --colorby phoneme`

* Read data from the folder `phoneme`
* Output json to `mfcc_phoneme_30.json` (The "30" is added automatically based on perplexity)
* Output png plot to `mfcc_phoneme_30.png`
* Load metadata from `phoneme/labels.csv`
* Set the dataset name to `phoneme`
* Run fingerprinting/feature extraction with mfcc
* Create 2d map
* Base default coloring on the "phoneme" tag

`python3.6 crunch.py -f nsynth/train/audio -o fft_nsynth_.json -t fft_nsynth_.png -p 50 70 90 200 -a 30000 -c nsynth/labels.csv -n nsynth -g fft --colorby pitch --parallel`

* Read data from the folder `nsynth/train/audio`
* Output json to `fft_nsynth_50.json`, `fft_nsynth_70.json`, `fft_nsynth_90.json` and `fft_nsynth_200.json`
* Output png plot to `fft_nsynth_50.png`, `fft_nsynth_70.png`, `fft_nsynth_90.png` and `fft_nsynth_200.png`
* Run t-SNE with perplexities 50, 70, 90 and 200
* Load a maximum of 30000 samples from the dataset
* Load metadata from `nsynth/labels.csv`
* Set the dataset name to `nsynth`
* Run fingerprinting/feature extraction with fft
* Create 3d map. (No `--td` flag present)
* Base default coloring on the "pitch" tag.
* Run t-SNE with all perplexities simultaneously. (Hopefully there is at least 32 gigs of idle memory)

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
