#!/usr/bin/python3

import json
from os.path import splitext, getsize, join
import struct
from argparse import ArgumentParser
import ffmpy
from multiprocessing import Pool


def _parse_arguments():
    parser = ArgumentParser(description='Create concatenated binary file for SSGL Speech Explorer.')
    parser.add_argument("json", help="The json file containing sample information")
    parser.add_argument("-e", "--ext", default='mp3')
    parser.add_argument("-i", "--input", default='.', help='Folder path of sound files')
    parser.add_argument("-o", "--output", default='concatenated_sounds.blob', help='Name of output file')
    parser.add_argument("-c", "--convert", action='store_true', help='Convert to mp3 using ffmpeg')

    return parser


def _read_points(json_path):
    with open(json_path) as file:
        data = json.load(file)
    return data["points"]


def _write_to_file(output_file, file):
    b = getsize(file)
    output_file.write(struct.pack('I', b))
    with open(file, 'rb') as input_file:
        output_file.write(input_file.read())


def _convert(input_file_name, output_file_name):
    print('converting {} to {}'.format(input_file_name, output_file_name))
    ff = ffmpy.FFmpeg(
        inputs={input_file_name: None},
        outputs={output_file_name: None},
        global_options="-loglevel error -y"
    )
    ff.run()


def main(args):
    inputs_and_outputs = [(join(args.input, inpath[3]), join(args.input, '{}.{}'.format(splitext(inpath[3])[0], args.ext))) for inpath in _read_points(args.json)]
    if args.convert:
        with Pool() as pool:
            pool.starmap(_convert, inputs_and_outputs)
    with open(args.output, 'wb') as blob_output_file:
        for in_and_out in inputs_and_outputs:
            _write_to_file(blob_output_file, in_and_out[1])


if __name__ == '__main__':
    main(_parse_arguments().parse_args())
