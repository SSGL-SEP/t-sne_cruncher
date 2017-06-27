#!/usr/bin/python3

import json
from os.path import splitext, getsize, join
import struct
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from typing import List, Union
from io import FileIO

import ffmpy


def _parse_arguments() -> ArgumentParser:
    """
    Provide an argument parser instance for the audio concatenator
    :return: Instance of ArgumentParser
    :rtype: argparse.ArgumentParser
    """
    parser = ArgumentParser(description='Create concatenated binary file for SSGL Speech Explorer.')
    parser.add_argument("json", help="The json file containing sample information")
    parser.add_argument("-e", "--ext", default='mp3')
    parser.add_argument("-i", "--input", default='.', help='Folder path of sound files')
    parser.add_argument("-o", "--output", default='concatenated_sounds.blob', help='Name of output file')
    parser.add_argument("-c", "--convert", action='store_true', help='Convert to mp3 using ffmpeg')

    return parser


def _read_points(json_path: str) -> List[List[Union[int, str]]]:
    """
    Read point data from json file
    
    :param json_path: Path to json file
    :type json_path: str
    :return: List of points (where points are lists containing coordinates and file name)
    :rtype: List[List[Union[int, str]]]
    """
    with open(json_path) as file:
        data = json.load(file)
    return data["points"]


def _write_to_file(output_file: FileIO, file: str) -> None:
    """
    Add data from file to output.
    :param output_file: binary file to append data to.
    :type output_file: FileIO
    :param file: file to read from
    :type file: str
    """
    b = getsize(file)
    output_file.write(struct.pack('I', b))
    with open(file, 'rb') as input_file:
        output_file.write(input_file.read())


def _convert(input_file_name: str, output_file_name: str) -> None:
    """
    Create mp3 file from wav.

    :param input_file_name: name of file to converti
    :type input_file_name: str
    :param output_file_name: name of file to write
    :type output_file_name: str
    """
    print('converting {} to {}'.format(input_file_name, output_file_name))
    ff = ffmpy.FFmpeg(
        inputs={input_file_name: None},
        outputs={output_file_name: None},
        global_options="-loglevel error -y"
    )
    ff.run()


def main(args: Namespace) -> None:
    """
    Create blob from wav or mp3 files
    :param args: command line parameters
    :type args: Namespace
    """
    inputs_and_outputs = [(join(args.input, inpath[3]),
                           join(args.input, '{}.{}'.format(splitext(inpath[3])[0], args.ext)))
                          for inpath in _read_points(args.json)]
    if args.convert:
        with Pool() as pool:
            pool.starmap(_convert, inputs_and_outputs)
    with open(args.output, 'wb') as blob_output_file:
        for in_and_out in inputs_and_outputs:
            _write_to_file(blob_output_file, in_and_out[1])


if __name__ == '__main__':
    main(_parse_arguments().parse_args())
