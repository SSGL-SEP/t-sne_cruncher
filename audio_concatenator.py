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

    return parser.parse_args()


def main(args):
    with open(args.json) as file:
        data = json.load(file)

    output_file = open(args.output, 'wb')

    for point in data['points']:
        file = args.input + '/' + splitext(point[3])[0] + '.' + args.ext
        b = getsize(file)
        output_file.write(struct.pack('I', b))
        f = open(file, 'rb')
        output_file.write(f.read())


if __name__ is '__main__':
    main(_parse_arguments())
