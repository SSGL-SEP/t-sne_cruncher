#!/usr/bin/python3

import json
from os.path import splitext, getsize
import struct
from argparse import ArgumentParser


def _parse_arguments():
    parser = ArgumentParser(description='Create concatenated binary file for SSGL Speech Explorer.')
    parser.add_argument("json", help="The json file containing sample information")
    parser.add_argument("-e", "--ext", default='mp3')
    parser.add_argument("-i", "--input", default='.', help='Folder path of sound files')
    parser.add_argument("-o", "--output", default='concatenated_sounds.blob', help='Name of output file')

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


def main(args):
    with open(args.output, 'wb') as output_file:
        for point in _read_points(args.json):
            file = args.input + '/' + splitext(point[3])[0] + '.' + args.ext
            _write_to_file(output_file, file)


if __name__ is '__main__':
    main(_parse_arguments().parse_args())
