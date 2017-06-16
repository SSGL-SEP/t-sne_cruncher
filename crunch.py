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
    arg_parser.add_argument("--td", help="Generates 2d data instead of the default 3d.", action="store_true")
    arg_parser.add_argument("--colorby", type=str, default=None,
                            help="Tag to do default coloring by.")
    arg_parser.add_argument("--parallel", action="store_true", help="If t-SNE reduction is selected will run batch"
                                                                    " perplexities in parallel. Memory hog.")
    return arg_parser


def _finalize(x_nd: tuple, args, data: list, metadata: dict) -> None:
    add_color(metadata, x_nd[0])
    if args.plot_output:
        plot_results(x_nd[0], insert_suffix(args.plot_output, x_nd[1]), metadata, args.colorby)
    norm_data = normalize(np.asarray(x_nd[0]), args.value_minimum, args.value_maximum)
    output(insert_suffix(args.output_file, x_nd[1]), data, norm_data, args, metadata, x_nd[1])


def _read_and_fingerprint(tup: tuple):
    file_path = tup[0]
    args = tup[1]
    fingerprint_function = ProcessFunctions.fingerprint_dict[args.fingerprint_method]
    data = load_sample(file_path, args.duration)
    result = fingerprint_function(data[1], data[3], data[2])
    print("{} read and fingerprinted".format(file_path))
    return result, data[0], data[2]


def _read_data_to_fingerprints(args) -> tuple:
    """
    Read audio files and generate fingerprint data

    """
    file_list = list(all_files(args.input_folder, [".wav"]))
    max_to_read = min(len(file_list), args.max_to_load) if args.max_to_load else len(file_list)
    with Pool() as p:
        data = p.map(_read_and_fingerprint, [(f, args) for f in file_list[:max_to_read]])
    results = [x[0] for x in data]
    file_data = [(x[1], x[2]) for x in data]
    return results, file_data


def _run_dimensionality_reduction(data, args, file_data, metadata):
    output_dimensions = 2 if args.td else 3
    reduction_function = ProcessFunctions.dimensionality_reduction_dict[args.reduction_method]
    return reduction_function(data, output_dimensions, args, a_func=_finalize, a_params=(args, file_data, metadata))


def main(args):
    """
    Run analysis and reduction based on command line parameters

    :param args: parsed command line argument object
    :type args: argparse.Namespace
    """
    t = time.time()
    results, file_data = _read_data_to_fingerprints(args)
    results = np.asarray(results).astype(np.float64)
    print("Read and fingerprinted {} files.".format(len(file_data)))
    if len(results.shape) > 2:
        results = results.reshape(len(results), -1)
    if args.fingerprint_output:
        np.save(args.fingerprint_output, results)
        print("Wrote fingerprint data to {}.".format(args.fingerprint_output))
    metadata = {}
    if args.collect_metadata:
        metadata = parse_metadata(args, {file_data[i][0].split('/')[-1]: i for i in range(len(file_data))})
    _run_dimensionality_reduction(results, args, file_data, metadata)
    print("Crunching completed in ", int(time.time() - t), " seconds")


def output(file_path: str, data: list, x_nd: np.ndarray, args, metadata, suffix: str) -> None:
    data_list = collect(data, x_nd, metadata, args, suffix)
    with open(file_path, 'w') as outfile:
        json.dump(data_list, outfile, indent=4, separators=(',', ':'))
    print("Wrote data to {}.".format(file_path))


def _make_header(args, point_count, suffix: str):
    return {"soundInfo": args.sound_info, "dataSet": args.data_set,
            "processingMethod": "{} - {}, {}".format(args.fingerprint_method, args.reduction_method, suffix),
            "colorBy": args.colorby, "totalPoints": point_count}


def collect(data: list, x_nd: np.ndarray, metadata: dict, args, suffix: str) -> dict:
    out_data = _make_header(args, len(data), suffix)
    out_data["tags"] = metadata
    lst = []
    for i in range(len(data)):
        fn = data[i][0].split("/")[-1]
        lst.append([x_nd[i][0], x_nd[i][1], 0 if len(x_nd[i]) < 3 else x_nd[i][2], fn])
    out_data["points"] = lst
    return out_data


class ProcessFunctions:
    fingerprint_dict = {"fft": mfcc_fingerprint, "chroma": chroma_fingerprint,
                        "ms": ms_fingerprint}

    dimensionality_reduction_dict = {"pca": pca, "tsne": t_sne}

if __name__ == "__main__":
    main(_arg_parse().parse_args())
