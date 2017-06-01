import json
import os
import time
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
from sklearn.decomposition import PCA

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
    arg_parser.add_argument("-p", "--perplexity", nargs='*', type=int, default=[30],
                            help="Perplexity to use in t-sne crunching. Default: [30].")
    # arg_parser.add_argument("-n", "--initial_dimensions", type=int, default=30,
    #                         help="Number of initial dimensions for t-sne crunching. Default: 30.")
    arg_parser.add_argument("-o", "--output_file", type=str, default=os.path.join(os.getcwd(), 't_sne.json'),
                            help="File to output 3d data to. Default: tsv.json in current directory.")
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
    arg_parser.add_argument("--td", help="Generates 2d data instead of the default 3d.", action="store_true")
    arg_parser.add_argument("--colorby", type=str, default=None,
                            help="Generate sample coloring based on metadata tag. "
                                 "Default: manhattan distance from origin")
    arg_parser.add_argument("--pca", action="store_true", help="Run dimensionality reduction with PCA instead of t-SNE")
    return arg_parser


# noinspection PyTypeChecker
def _finalize(x_3d, args, data, suffix=""):
    if args.plot_output:
        plot_t_sne(x_3d, insert_suffix(args.plot_output, suffix))
    x_3d = normalize(np.asarray(x_3d), args.value_minimum, args.value_maximum)
    output(insert_suffix(args.output_file, suffix), data, x_3d, args.collect_metadata, args.colorby)


def _read_data_to_fingerprints(max_duration, input_folder):
    file_list = list(all_files(input_folder, [".wav"]))
    results = []
    file_data = []
    count = 0
    with Pool() as p:
        while count < len(file_list):
            data = p.map(load_sample, [(max_duration, f) for f in file_list[count:(count + 1000)]])
            print("read to {}".format(min(len(file_list), count + 1000)))
            file_data += [(x[0], x[2]) for x in data]
            results += list(p.map(fingerprint_form_data, [x[1] for x in data]))
            count += 1000
    return results, file_data


def main(args):
    """
    Run analysis and t-SNE reduction based on command line parameters

    :param args: parsed command line argument object
    :type args: argparse.Namespace
    """
    t = time.time()
    output_dimensions = 2 if args.td else 3
    results, file_data = _read_data_to_fingerprints(args.max_duration, args.input_folder)
    results = np.asarray(results).astype(np.float32)
    if args.fingerprint_output:
        np.save(args.fingerprint_output, results)
    results = results.astype(np.float64)
    results = results.reshape(len(results), -1)
    # x_3d = t_sne(results, initial_dims=args.initial_dimensions, perplexity=args.perplexity, no_dims=output_dimensions)
    if args.pca:
        model = PCA(n_components=output_dimensions, svd_solver='full')
        x_3d = model.fit_transform(results)
        _finalize(x_3d, args, file_data)
    else:
        l_3d = t_sne(results, perplexity=args.perplexity, no_dims=output_dimensions)
        for x_3d in l_3d:
            _finalize(x_3d[0], args, file_data, x_3d[1])
    print("Crunching completed in ", int(time.time() - t), " seconds")


def output(file_path, data, x_3d, metadata_location, color_by):
    """
    Dump json containing calculated 2d or 3d projection and possibly metadata.
    
    :param color_by: tag to color samples by 
    :type color_by: Union(str, None)
    :param file_path: .json file to write results to
    :type file_path: str
    :param data: File name and length as provided by the _read_data_to_fingerprints.
    :type data: List[Tuple[str, int]]
    :param x_3d: The numpy array containing the 3d projection.
    :type x_3d: numpy.ndarray
    :param metadata_location: .csv file to load metadata from.
    :type metadata_location: Union(str, None)
    """
    metadata = {}
    if metadata_location:
        metadata = parse_metadata(metadata_location)
    data_list = collect(data, x_3d, metadata)
    add_color(data_list, color_by)
    with open(file_path, 'w') as outfile:
        json.dump(data_list, outfile, indent=4, separators=(',', ':'))


def collect(data, x_3d, metadata):
    """
    Create a list of objects to write to json.

    :param data: file name and length as provided by the _read_data_to_fingerprints
    :type data: List[Tuple[str, int]]
    :param x_3d: The numpy array containing the 3d projection
    :type x_3d: numpy.ndarray
    :param metadata: Dictionary containing metadata for the files.
    :type metadata: Dict
    :return: A list of lists of values to be written to json.
    :rtype: List[List[Union[int, str, List[Dict{str : str}]]]]
    """
    lst = []
    for i in range(len(data)):
        fn = data[i][0].split("/")[-1]
        lst.append([i, x_3d[i][0], x_3d[i][1], 0 if len(x_3d[i]) < 3 else x_3d[i][2], fn,
                    metadata[fn] if fn in metadata else []])
    return lst

if __name__ == "__main__":
    main(_arg_parse().parse_args())
