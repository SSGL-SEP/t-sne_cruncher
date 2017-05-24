import json
import os
import time
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np

from subprocesses import *
from utils import *


def _arg_parse():
    """
    Provides a command line argument parser object.
    
    :return: Argument parser for cruncher.py
    :rtype: argparse.ArgumentParser
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--input_folder", type=str, default=os.getcwd(),
                            help="Folder to read audio files from. Default: current directory.")
    arg_parser.add_argument("-p", "--perplexity", type=int, default=30,
                            help="Perplexity to use in t-sne crunching. Default: 30.")
    arg_parser.add_argument("-n", "--initial_dimensions", type=int, default=30,
                            help="Number of initial dimensions for t-sne crunching. Default: 30.")
    arg_parser.add_argument("-o", "--output_file", type=str, default=os.path.join(os.getcwd(), 't_sne.json'),
                            help="File to output 3d data to. Default: tsv.json in current directory.")
    arg_parser.add_argument("-s", "--sample_output", type=str,  default=None,
                            help="'.npy' file to output sample data to. Default: None.")
    arg_parser.add_argument("-r", "--fingerprint_output", type=str, default=None,
                            help="'.npy' file to output fingerprint data to. Default: None")
    arg_parser.add_argument("-m", "--value_minimum", type=int, default=0,
                            help="Minimum coordinate value to use when scaling. Default: 0")
    arg_parser.add_argument("-x", "--value_maximum", type=int, default=600,
                            help="Maximum coordinate value to use when scaling. Default: 600")
    arg_parser.add_argument("-t", "--plot_output", type=str, default=None,
                            help="'.png' file to write scatter plot to after t-sne crunching. Default: None")
    arg_parser.add_argument("-c", "--collect_metadata", type=str, default=None,
                            help="'.csv' file to read file metadata from. audio file names should be in first column. "
                                 "Default: None")
    arg_parser.add_argument("-d", "--max_duration", type=int, default=500,
                            help="Maximum duration of sound samples (in milliseconds). Longer samples will be "
                                 "truncated to the given length. Default: 500ms")
    return arg_parser


# noinspection PyTypeChecker
def main(args):
    """
    Run analysis and t-SNE reduction based on command line parameters

    :param args: parsed command line argument object
    :type args: argparse.Namespace
    """
    t = time.time()
    data = collect_data(max_duration=args.max_duration, source_folder=args.input_folder, target_file=args.sample_output)
    with Pool() as p:
        results = p.map(fingerprint_form_data, [x[1] for x in data])
    results = np.asarray(results).astype(np.float32)
    if args.fingerprint_output:
        np.save(args.fingerprint_output, results)
    results = results.astype(np.float64)
    results = results.reshape(len(results), -1)
    x_3d = t_sne(results, initial_dims=args.initial_dimensions, perplexity=args.perplexity)
    if args.plot_output:
        plot_t_sne(x_3d, args.plot_output)
    x_3d = normalize(np.asarray(x_3d), args.value_minimum, args.value_maximum)
    output(args.output_file, data, x_3d, args.collect_metadata)
    print("Crunching completed in ", int(time.time() - t), " seconds")


def output(file_path, data, x_3d, metadata_location):
    """
    Dump json containing calculated 3d projection and posibly metadata.
    
    :param file_path: .json file to write results to
    :type file_path: str
    :param data: Raw audio data as provided by the collect_data module.
    :type data: List[Tuple[str, numpy.ndarray, int]]
    :param x_3d: The numpy array containing the 3d projection.
    :type x_3d: numpy.ndarray
    :param metadata_location: .csv file to load metadata from.
    :type metadata_location: str
    """
    metadata = {}
    if metadata_location:
        metadata = parse_metadata(metadata_location)
    data_list = collect(data, x_3d, metadata)
    with open(file_path, 'w') as outfile:
        json.dump(data_list, outfile, indent=4, separators=(',', ':'))


def collect(data, x_3d, metadata):
    """
    Create a list of objects to write to json.

    :param data: raw data as provided by the collect_data module
    :type data: List[Tuple[str, numpy.ndarray, int]]
    :param x_3d: The numpy array containg the 3d projection
    :type x_3d: numpy.ndarray
    :param metadata: Dictionary containing metadata for the files.
    :type metadata: Dict
    :return: A list of lists of values to be written to json.
    :rtype: List[List[Union[int, str, List[Dict{str : str}]]]]
    """
    lst = []
    for i in range(len(data)):
        fn = data[i][0].split("/")[-1]
        lst.append([i, x_3d[i][0], x_3d[i][1], x_3d[i][2], fn, metadata[fn] if fn in metadata else []])
    return lst

if __name__ == "__main__":
    main(_arg_parse().parse_args())
