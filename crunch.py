import json
import os
import time
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from typing import Tuple, List, Dict, Any, Union

import numpy as np

from subprocesses import *
from utils import *


def _arg_parse() -> ArgumentParser:
    """
    Provides a command line argument parser object.
    
    :return: Argument parser for cruncher.py
    :rtype: ArgumentParser
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--input_folder", type=str, default=os.getcwd(),
                            help="Folder to read audio files from. Default: current directory.")
    arg_parser.add_argument("-p", "--perplexity", nargs='*', type=int, default=[30],
                            help="Perplexity to use in t-sne crunching. Default: [30].")
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
    arg_parser.add_argument("-d", "--duration", type=int, default=0,
                            help="Duration of sound samples (in milliseconds). Longer samples will be "
                                 "truncated and shorter samples may be padded. Default: 0 to load entire sample")
    arg_parser.add_argument("-u", "--unfilterables", type=str, nargs='*', default=[],
                            help="List of tags that should not be used for filtering.")
    arg_parser.add_argument("-n", "--data_set", type=str, default="dataset",
                            help="Name of data set the samples are associated with. Default: 'dataset'")
    arg_parser.add_argument("-s", "--sound_info", type=str, default=None,
                            help=".json file to read audio data from. Should provide a mapping between audio files"
                                 "and compressed versions to use when streaming. Default: None")
    arg_parser.add_argument("-a", "--max_to_load", type=lambda x: abs(int(x)), default=0,
                            help="Maximum number of samples to load. Default: 0 to load all samples")
    arg_parser.add_argument("-b", "--tags_to_ignore", type=str, nargs='*',
                            default=["waveform", "name", "filename", "file name"],
                            help="List of tags to completely ignore.")
    arg_parser.add_argument("-k", "--reduction_method", type=str,
                            choices=ProcessFunctions.dimensionality_reduction_dict.keys(), default="tsne",
                            help="Dimensionality reduction algorithm to use. Default: t-SNE")
    arg_parser.add_argument("-g", "--fingerprint_method", type=str, choices=ProcessFunctions.fingerprint_dict.keys(),
                            default="ms", help="Fingerprinting algorithm to use. Default: ms (mel spectrogram).")
    arg_parser.add_argument("-e", "--fingerprint_input", type=str, default=None,
                            help="Read fingerprints form file instead fo generating them. Default: None.")
    arg_parser.add_argument("--td", help="Generates 2d data instead of the default 3d.", action="store_true")
    arg_parser.add_argument("--colorby", type=str, default=None,
                            help="Tag to do default coloring by.")
    arg_parser.add_argument("--parallel", action="store_true", help="If t-SNE reduction is selected will run batch"
                                                                    " perplexities in parallel. Memory hog.")
    return arg_parser


def _finalize(x_nd: Tuple[np.ndarray, str], args: Namespace, data: List[str], metadata: Dict[str, dict]) -> None:
    """
    Add coloration to the metadata and output files.

    :param x_nd: Pairs of arrays of reduced 3d or 2d vectors and identifiers
    :type x_nd: Tuple[numpy.ndarray, str]
    :param args: command line argument namespace
    :type args: argparse.Namespace
    :param data: List of file paths.
    :type data: List[str]
    :param metadata: Metadata dictionary
    :type metadata: Dict[str, Dict[str, bool]]
    """
    add_color(metadata, x_nd[0])
    if args.plot_output:
        plot_results(x_nd[0], insert_suffix(args.plot_output, x_nd[1]), metadata, args.colorby)
    norm_data = normalize(np.asarray(x_nd[0]), args.value_minimum, args.value_maximum)
    output(insert_suffix(args.output_file, x_nd[1]), data, norm_data, args, metadata, x_nd[1])


def _read_and_fingerprint(file_path: str, args: Namespace) -> Tuple[np.ndarray, str, int]:
    """
    Read audio data from file and fingerprint it with specified fingerprint algorithm.

    :param file_path: Path to file to read.
    :type file_path: str
    :param args: Command line arguments.
    :type args: argparse.Namespace
    :return: Tuple containing fingerprint data, file path and audio length
    :rtype: Tuple[numpy.ndarray, str, int]
    """
    fingerprint_function = ProcessFunctions.fingerprint_dict[args.fingerprint_method]
    data = load_sample(file_path, args.duration)
    result = fingerprint_function(data[1], data[3], data[2])
    print("{} read and fingerprinted".format(file_path))
    return result, data[0], data[2]


def _read_data_to_fingerprints(args: Namespace) -> Tuple[List[np.ndarray], List[str]]:
    """
    Read audio files and generate fingerprint data asynchronously based on command line arguments.

    :param args: Command line arguments.
    :type args: argparse.Namespace
    :return: Tuple with list of fingerprint data and a list of file paths.
    :rtype: Tuple[List[numpy.ndarray], List[str]]]

    """
    file_list = list(all_files(args.input_folder, [".wav"]))
    max_to_read = min(len(file_list), args.max_to_load) if args.max_to_load else len(file_list)
    with Pool() as p:
        data = p.starmap(_read_and_fingerprint, [(f, args) for f in file_list[:max_to_read]])
    results = [x[0] for x in data]
    file_data = [x[1] for x in data]
    return results, file_data


def _run_dimensionality_reduction(data: List[np.ndarray], args: Namespace, file_data: List[str],
                                  metadata: Dict[str, dict]) -> List[Tuple[np.ndarray, str]]:
    """
    Run 3d or 2d dimensionality reduction on data

    :param data: data to run dimensionality reduction on.
    :type data: numpy.ndarray
    :param args: Command line arguments
    :type args: argparse.Namespace
    :param file_data: List of file paths.
    :type file_data: List[str]
    :param metadata: Metadata dictionary
    :type metadata: Dict[str, Dict[str, bool]]
    :return: List of tuples of reduced data and descriptor strings.
    :rtype: List[Tuple[numpy.ndarray, str]]
    """
    output_dimensions = 2 if args.td else 3
    reduction_function = ProcessFunctions.dimensionality_reduction_dict[args.reduction_method]
    return reduction_function(data, output_dimensions, args, a_func=_finalize, a_params=(args, file_data, metadata))


def load_fingerprints(args: Namespace) -> Tuple[List[np.ndarray], List[str]]:
    """
    Read previously calculated fingerprint data from file.

    :param args: Command line arguments
    :type args: argparse.Namespace
    :return: Tuple of a list of fingerprint data and list of paths.
    :rtype: Tuple[List[numpy.ndarray], List[str]]
    """
    data = np.load(args.fingerprint_input)
    results = [c[0] for c in data]
    file_data = [c[1] for c in data]
    return results, file_data


def generate_fingerprints(args: Namespace) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate fingerprints form wave files and optionally saves them to disk.

    :param args: Command line arguments.
    :type args: argparse.Namespace
    :return: Tuple of a list of fingerprint data and list of file paths.
    :rtype: Tuple[List[np.ndarray], List[str]]
    """
    results, file_data = _read_data_to_fingerprints(args)
    results = np.asarray(results).astype(np.float64)
    print("Read and fingerprinted {} files.".format(len(file_data)))
    if len(results.shape) > 2:
        results = results.reshape(len(results), -1)
    if args.fingerprint_output:
        np.save(args.fingerprint_output, [(results[i], file_data[i]) for i in range(len(results))])
        print("Wrote fingerprint data to {}.".format(args.fingerprint_output))
    return results, file_data


def main(args: Namespace) -> None:
    """
    Run analysis and reduction based on command line parameters

    :param args: parsed command line argument object
    :type args: argparse.Namespace
    """
    t = time.time()
    results, file_data = load_fingerprints(args) if args.fingerprint_input else generate_fingerprints(args)
    metadata = {}
    if args.collect_metadata:
        metadata = parse_metadata(args, {file_data[i].split('/')[-1]: i for i in range(len(file_data))})
    _run_dimensionality_reduction(results, args, file_data, metadata)
    print("Crunching completed in ", int(time.time() - t), " seconds")


def output(file_path: str, data: List[str], x_nd: np.ndarray, args: Namespace,
           metadata: Dict[str, dict], suffix: str) -> None:
    """
    Output results of dimensionality reduction.

    :param file_path: File to write to
    :type file_path: str
    :param data: List of file paths.
    :type data: List[str]
    :param x_nd: collection of 3d or 2d data to output.
    :type x_nd: numpy.ndarray
    :param args: Command line arguments
    :type args: argparse.Namespace
    :param metadata: metadata dictionaty
    :type metadata: Dict[str, Dict[str, bool]]
    :param suffix: reduction suffix to append to the processing method field.
    :type suffix: str
    """
    data_list = collect(data, x_nd, metadata, args, suffix)
    with open(file_path, 'w') as outfile:
        json.dump(data_list, outfile, indent=4, separators=(',', ':'))
    print("Wrote data to {}.".format(file_path))


def _make_header(args: Namespace, point_count: int, suffix: str) -> Dict[str, Any]:
    """
    Generate header data for output
    :param args: Command line arguments
    :type args: argparse.Namespace
    :param point_count: Number of total points in sample set
    :type point_count: int
    :param suffix: reduction suffix to append to the processing method field.
    :type suffix: str
    :return: Dictionary containing header data to be written to file.
    :rtype: Dict[str, NoneType]
    """
    return {"soundInfo": args.sound_info, "dataSet": args.data_set,
            "processingMethod": "{} - {}, {}".format(args.fingerprint_method, args.reduction_method, suffix),
            "colorBy": args.colorby, "totalPoints": point_count}


def collect(data: List[str], x_nd: np.ndarray, metadata: Dict[str, dict], args: Namespace,
            suffix: str) -> Dict[str, Union[dict, str]]:
    """
    Generate data dictionary to be written to json.

    :param data: List of file paths.
    :type data: List[str]
    :param x_nd: Numpy array containing 2d or 3d data.
    :type x_nd: numpy.ndarray
    :param metadata: Dictionary of tag data associated with the samples
    :type metadata: Dict[str, Dict[str, bool]]
    :param args: Command line arguments
    :type args: argparse.Namespace
    :param suffix: reduction specific suffix
    :type suffix: str
    :return: Dictionary ready to be written to json.
    :rtype: Dict[str, NoneType]
    """
    out_data = _make_header(args, len(data), suffix)
    out_data["tags"] = metadata
    lst = []
    for i in range(len(data)):
        fn = data[i].split("/")[-1]
        lst.append([x_nd[i][0], x_nd[i][1], 0 if len(x_nd[i]) < 3 else x_nd[i][2], fn])
    out_data["points"] = lst
    return out_data


class ProcessFunctions:
    fingerprint_dict = {"fft": fft_fingerprint, "chroma": chroma_fingerprint,
                        "ms": ms_fingerprint, "mfcc": mfcc_fingerprint}

    dimensionality_reduction_dict = {"pca": pca, "tsne": t_sne}

if __name__ == "__main__":
    main(_arg_parse().parse_args())
