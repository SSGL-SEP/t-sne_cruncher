# Audio concatentor

The audio concatenator script is used to create a single binary file that contains all of the required sound samples used in [Speech explorer](https://github.com/SSGL-SEP/speech_explorer). The input files must have the same base name as specified in the json file that the cruncher produces. By default the script looks for mp3 files that are named according to the wav files in the input json file. The concatenator can optionally compress the input files before concatenation. 

The script is modeled after [ConcatSoundLoader](https://github.com/spite/ConcatSoundLoader)

### Usage
`audio_concatenator.py [-h] [-e EXT] [-i INPUT] [-o OUTPUT] [-c] json` 

### Example

The following command processes the json file located in `path/to/cruched/data.json`. It takes the mp3 files from `audio/file/path/` and produces an output file called `concatenated.file`

`python3 audio_concatenator -i audio/file/path/ -o concatenated.file -e mp3 path/to/crunched/data.json`

### Input options

* __-i / --input__ The folder that contains the audio files to concatenate. Defaults to the current working directory.
* __-o / --output__ The file name for the output. Defauls to concatenated_sounds.blob, which is what the Speech Explorer search for.
* __-e / --ext__ The file extension of the files to be concatenated. Defaults to mp3.
* __-c / --convert__ Converts the audio files to the format specified by `--ext` using the [ffmpeg](https://ffmpeg.org/) encoder. __Note:__ ffmpeg must be installed on the system in order to use this.

