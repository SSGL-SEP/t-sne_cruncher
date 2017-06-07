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
    arg_parser.add_argument("--fft", action="store_true", help="Fingerprint using simple fft instead of mfcc")
    arg_parser.add_argument("-s", "--sound_info", type=str, default=None,
                            help=".json file to read audio data from. Should provide a mapping between audio files"
                                 "and compressed versions to use when streaming. Default: None")
    arg_parser.add_argument("-n", "--data_set", type=str, default="dataset",
                            help="Name of data set the samples are associated with. Default: 'dataset'")
    arg_parser.add_argument("-e", "--processing_method", type=str, default="t-SNE",
                            help="Descriptive name for the fingerprinting/mapping functions used. Default: 't_SNE'")
    arg_parser.add_argument("-u", "--unfilterables", type=str, nargs='*', default=["name", "filename"],
                            help="List of tags that should not be used for filtering.")
    return arg_parser


def _finalize(x_3d: np.ndarray, args, data: list, metadata: dict, suffix: str = "") -> None:
    if args.plot_output:
        plot_t_sne(x_3d, insert_suffix(args.plot_output, suffix))
    x_3d = normalize(np.asarray(x_3d), args.value_minimum, args.value_maximum)
    output(insert_suffix(args.output_file, suffix), data, x_3d, args, metadata)


def _read_data_to_fingerprints(max_duration: int, input_folder: str, fft: bool = False) -> tuple:
    """
    Read audio files and generate fingerprint data

    :param max_duration: Maximum allowed duration of audio samples in ms.
    :type max_duration: int
    :param input_folder: Folder path to read files from
    :type input_folder: str
    :return: A list with fingerprint data and a list of file data
    :rtype: Tuple[List[numpy.ndarray], List[Tuple[str, int]]]
    """
    file_list = list(all_files(input_folder, [".wav"]))
    results = []
    file_data = []
    count = 0
    with Pool() as p:
        while count < len(file_list):
            data = p.map(load_sample, [(max_duration, f) for f in file_list[count:(count + 1000)]])
            print("read to {}".format(min(len(file_list), count + 1000)))
            file_data += [(x[0], x[2]) for x in data]
            if fft:
                results += list(p.map(fingerprint_form_data, [x[1] for x in data]))
            else:
                results += list(p.map(mfcc_fingerprint, [(x[1], x[3]) for x in data]))
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
    results, file_data = _read_data_to_fingerprints(args.max_duration, args.input_folder, args.fft)
    results = np.asarray(results).astype(np.float32)
    if args.fingerprint_output:
        np.save(args.fingerprint_output, results)
    results = results.astype(np.float64)
    results = results.reshape(len(results), -1)
    metadata = {}
    if args.collect_metadata:
        metadata = parse_metadata(args.collect_metadata,
                                  {file_data[i][0].split('/')[-1]: i for i in range(len(file_data))},
                                  args.unfilterables)
    if args.pca:
        model = PCA(n_components=output_dimensions, svd_solver='full')
        x_3d = model.fit_transform(results)
        _finalize(x_3d, args, file_data, metadata)
    else:
        l_3d = t_sne(results, perplexity=args.perplexity, no_dims=output_dimensions)
        for x_3d in l_3d:
            _finalize(x_3d[0], args, file_data, metadata, x_3d[1])
    print("Crunching completed in ", int(time.time() - t), " seconds")


def output(file_path: str, data: list, x_3d: np.ndarray, args, metadata) -> None:
    data_list = collect(data, x_3d, metadata, args)
    with open(file_path, 'w') as outfile:
        json.dump(data_list, outfile, indent=4, separators=(',', ':'))


def _make_header(args, point_count):
    return {"soundInfo": args.sound_info, "dataSet": args.data_set,
            "processingMethod": args.processing_method, "colorBy": args.colorby,
            "totalPoints": point_count}


def collect(data: list, x_3d: np.ndarray, metadata: dict, args) -> dict:
    out_data = _make_header(args, len(data))
    add_color(metadata, x_3d)
    out_data["tags"] = metadata
    lst = []
    for i in range(len(data)):
        fn = data[i][0].split("/")[-1]
        lst.append([x_3d[i][0], x_3d[i][1], 0 if len(x_3d[i]) < 3 else x_3d[i][2], fn])
    out_data["points"] = lst
    return out_data

if __name__ == "__main__":
    main(_arg_parse().parse_args())
