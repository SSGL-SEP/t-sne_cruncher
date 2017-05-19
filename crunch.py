from utils import *
from calculate_tsne import t_sne, plot_t_sne
from collect_data import collect_data
from fingerprint import fingerprint_form_data
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import os
import json


def _arg_parse():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--input_folder", type=str, default=os.getcwd(),
                            help="Folder to read audio files from. Default: current directory.")
    arg_parser.add_argument("-p", "--perplexity", type=int, default=30,
                            help="Perplexity to use in t-sne crunching. Default: 30.")
    arg_parser.add_argument("-n", "--initial_dimensions", type=int, default=30,
                            help="Number of initial dimensions for t-sne crunching. Default: 30.")
    arg_parser.add_argument("-o", "--output_File", type=str, default=os.path.join(os.getcwd(), 't_sne.json'),
                            help="File to output 3d data to. Default: tsv.json in current directory.")
    arg_parser.add_argument("-s", "--sample_output", type=str,  default=None,
                            help="'.npy' file to output sample data to. Default: None.")
    arg_parser.add_argument("-r", "--fingerprint_output", type=str, default=None,
                            help="'.npy' file to output fingerprint data to. Default: None")
    arg_parser.add_argument("-m", "--value_minimum", type=int, default=0,
                            help="Minimum coordinate value to use when scaling. Default: 0")
    arg_parser.add_argument("-x", "--value_maximum", type=int, default=100,
                            help="Maximum coordinate value to use when scaling. Default: 100")
    arg_parser.add_argument("-t", "--plot_output", type=str, default=None,
                            help="'.png' file to wrote scatter plot to after t-sne crunching. Default: None")
    return arg_parser


def main(args):
    data = collect_data(args.input_folder, target_file=args.sample_output)
    with Pool() as p:
        results = p.map(fingerprint_form_data, [x[1] for x in data])
    results = np.asarray(results).astype(np.float32)
    if args.fingerprint_output:
        np.save(args.fingerprint_output, results)
    results = results.reshape(len(results), -1)
    results = results.astype(np.float64)
    x_3d = t_sne(results, initial_dims=args.initial_dimensions, perplexity=args.perplexity)
    if args.plot_output:
        plot_t_sne(x_3d, args.plot_output)
    x_3d = normalize(np.asarray(x_3d), args.value_minimum, args.value_maximum)
    output(args.output_File, data, x_3d)


def output(file_path, data, x_3d):
    data_list = collect(data, x_3d)
    with open(file_path, 'w') as outfile:
        json.dump(data_list, outfile, indent=4, separators=(',', ':'))


def collect(data, x_3d):
    lst = []
    for i in range(len(data)):
        lst.append([i, x_3d[i][0], x_3d[i][1], x_3d[i][2], data[i][0], []])
    return lst

if __name__ == "__main__":
    main(_arg_parse().parse_args())
