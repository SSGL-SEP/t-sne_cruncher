import os
from unittest import mock, TestCase

import numpy

import crunch


class TestMain(TestCase):
    def _file_data_comp(self, data, index, x_coord, y_coord, z_coord, file_name, tags):
        self.assertEqual(data[0], index, "Unexpected index. Expected {}, Got{}".format(index, data[0]))
        self.assertEqual(data[1], x_coord, "Unexpected x coordinate. Expected {}, Got{}".format(x_coord, data[1]))
        self.assertEqual(data[2], y_coord, "Unexpected y coordinate. Expected {}, Got{}".format(y_coord, data[2]))
        self.assertEqual(data[3], z_coord, "Unexpected z coordinate. Expected {}, Got{}".format(z_coord, data[3]))
        self.assertEqual(data[4], file_name, "Unexpected file name. Expected {}, Got{}".format(file_name, data[4]))
        self.assertEqual(data[5], tags, "Unexpected tag list. Expected {}, Got{}".format(tags, data[5]))

    def test_arg_parse_defaults(self):
        ap = crunch._arg_parse()
        args = ap.parse_args(args=[])
        self.assertEqual(args.input_folder, os.getcwd(), "Unexpected input folder")
        self.assertSequenceEqual(args.perplexity, [30], "Unexpected perplexity list")
        self.assertEqual(args.output_file, os.path.join(os.getcwd(), "t_sne_.json"), "Unexpected output directory")
        self.assertIsNone(args.fingerprint_output, "Unexpected fingerprint output")
        self.assertEqual(args.value_minimum, 0, "Unexpected minimum value")
        self.assertEqual(args.value_maximum, 600, "Unexpected maximum value")
        self.assertIsNone(args.plot_output, "Unexpected plot output")
        self.assertIsNone(args.collect_metadata, "Unexpected metadata path")
        self.assertEqual(args.duration, 0, "Unexpected maximum duration")
        self.assertSequenceEqual(args.unfilterables, [], "Unexpected unfilterable argument")
        self.assertEqual(args.data_set, "dataset", "Unexpected dataset value")
        self.assertIsNone(args.sound_info, "Unexpected sound_info value")
        self.assertEqual(args.max_to_load, 0, "Unexpected max_to_load value")
        self.assertSequenceEqual(args.tags_to_ignore, ["waveform", "name", "filename", "file name"],
                                 "Unexpected values in tags_to_ignore.")
        self.assertEqual(args.reduction_method, "tsne", "Unexpected value for reduction_method.")
        self.assertEqual(args.fingerprint_method, "ms", "Unexpected value for fingerprint_mathod.")
        self.assertIsNone(args.fingerprint_input, "Unexpected value for fingerprint input.")
        self.assertFalse(args.td, "Unexpected value for --td")
        self.assertIsNone(args.colorby, "Unexpected colorby value")
        self.assertFalse(args.parallel, "Unexpected value for parallel.")

    def test_arg_parse_values(self):
        ap = crunch._arg_parse()
        args = ap.parse_args(args=["-f", "samp/", "-p", "20", "50", "60", "-o", "pca.json", "-r", "fing.npy",
                                   "-m", "-100", "-x", "100", "-t", "pca.png", "-c", "samp/data.csv", "-d", "4000",
                                   "-u", "unfilterable", "-n", "test_data_set", "-s", "sound_info.json", "-a", "1000",
                                   "-b", "pronunciation", "-k", "pca", "-g", "fft", "-e", "input.npy", "--td",
                                   "--colorby", "phoneme", "--parallel"])
        self.assertEqual(args.input_folder, "samp/", "Unexpected input folder")
        self.assertSequenceEqual(args.perplexity, [20, 50, 60], "Unexpected perplexity list")
        self.assertEqual(args.output_file, "pca.json", "Unexpected output directory")
        self.assertEqual(args.fingerprint_output, "fing.npy", "Unexpected fingerprint output")
        self.assertEqual(args.value_minimum, -100, "Unexpected minimum value")
        self.assertEqual(args.value_maximum, 100, "Unexpected maximum value")
        self.assertEqual(args.plot_output, "pca.png", "Unexpected plot output")
        self.assertEqual(args.collect_metadata, "samp/data.csv", "Unexpected metadata path")
        self.assertEqual(args.duration, 4000, "Unexpected maximum duration")
        self.assertSequenceEqual(args.unfilterables, ["unfilterable"], "Unexpected value for unfilterable.")
        self.assertEqual(args.data_set, "test_data_set", "Unexpected value for data_set.")
        self.assertEqual(args.sound_info, "sound_info.json", "Unexpected value for sound_info.")
        self.assertEqual(args.max_to_load, 1000, "Unexpected value for max_to_load")
        self.assertSequenceEqual(args.tags_to_ignore, ["pronunciation"], "Unexpected tags_to_ignore value.")
        self.assertEqual(args.reduction_method, "pca", "Unexpected reduction method value.")
        self.assertEqual(args.fingerprint_method, "fft", "Unexpected fingerprint_method.")
        self.assertEqual(args.fingerprint_input, "input.npy", "Unexpected value for fingerprint input.")
        self.assertTrue(args.td, "Unexpected value for --td")
        self.assertEqual(args.colorby, "phoneme", "Unexpected colorby value")
        self.assertTrue(args.parallel, "Unexpected value for parallel.")
