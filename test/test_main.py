import os
from unittest import mock, TestCase

import numpy

import crunch


def mock_read(path, args):
    return numpy.asarray([1, 3, 3, 7]), path, 10


class TestMain(TestCase):
    def setUp(self):
        self.ap = crunch._arg_parse()
        self.def_ap = self.ap.parse_args(args=[])
        self.val_ap = self.ap.parse_args(args=["-f", "samp/", "-p", "20", "50", "60", "-o", "pca.json", "-r",
                                               "fing.npy", "-m", "-100", "-x", "100", "-t", "pca.png", "-c",
                                               "samp/data.csv", "-d", "4000", "-u", "unfilterable", "-n",
                                               "test_data_set", "-s", "sound_info.json", "-a", "1000", "-b",
                                               "pronunciation", "-k", "pca", "-g", "fft", "-e", "input.npy", "--td",
                                               "--colorby", "phoneme", "--parallel"])

    def _file_data_comp(self, data, index, x_coord, y_coord, z_coord, file_name, tags):
        self.assertEqual(data[0], index, "Unexpected index. Expected {}, Got{}".format(index, data[0]))
        self.assertEqual(data[1], x_coord, "Unexpected x coordinate. Expected {}, Got{}".format(x_coord, data[1]))
        self.assertEqual(data[2], y_coord, "Unexpected y coordinate. Expected {}, Got{}".format(y_coord, data[2]))
        self.assertEqual(data[3], z_coord, "Unexpected z coordinate. Expected {}, Got{}".format(z_coord, data[3]))
        self.assertEqual(data[4], file_name, "Unexpected file name. Expected {}, Got{}".format(file_name, data[4]))
        self.assertEqual(data[5], tags, "Unexpected tag list. Expected {}, Got{}".format(tags, data[5]))

    def test_arg_parse_defaults(self):
        args = self.def_ap
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
        args = self.val_ap
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

    @mock.patch("crunch.add_color")
    @mock.patch("crunch.plot_results")
    @mock.patch("crunch.normalize")
    @mock.patch("crunch.output")
    def test_finalize(self, mock_output, mock_normalize, mock_plot, mock_color):
        norm_data = mock.MagicMock
        mock_normalize.return_value = norm_data
        arr = numpy.asarray([[1, 2, 3], [2, 3, 4]]), "30"
        data = mock.MagicMock()
        metadata = mock.MagicMock()
        crunch._finalize(arr, self.def_ap, data, metadata)
        mock_color.assert_called_with(metadata, arr[0])
        self.assertFalse(mock_plot.called)
        mock_normalize.assert_called_with(arr[0], self.def_ap.value_minimum, self.def_ap.value_maximum)
        file_path = os.path.join(os.getcwd(), "t_sne_30.json")
        mock_output.assert_called_with(file_path, data, norm_data, self.def_ap, metadata, arr[1])

    @mock.patch("crunch.add_color")
    @mock.patch("crunch.plot_results")
    @mock.patch("crunch.normalize")
    @mock.patch("crunch.output")
    def test_finalize_output(self, mock_output, mock_normalize, mock_plot, mock_color):
        norm_data = mock.MagicMock
        mock_normalize.return_value = norm_data
        arr = numpy.asarray([[1, 2, 3], [2, 3, 4]]), "30"
        data = mock.MagicMock()
        metadata = mock.MagicMock()
        crunch._finalize(arr, self.val_ap, data, metadata)
        mock_color.assert_called_with(metadata, arr[0])
        mock_plot.assert_called_with(arr[0], "pca30.png", metadata, "phoneme")
        mock_normalize.assert_called_with(arr[0], self.val_ap.value_minimum, self.val_ap.value_maximum)
        file_path = "pca30.json"
        mock_output.assert_called_with(file_path, data, norm_data, self.val_ap, metadata, arr[1])

    @mock.patch("crunch.load_sample")
    def test_read_and_fingerprint(self, mock_load):
        mock_fingerprint = mock.MagicMock()
        crunch.ProcessFunctions.fingerprint_dict["ms"] = mock_fingerprint
        file_path = "mock_path"
        file_data = mock.MagicMock()
        arr = numpy.asarray([1, 3, 3, 7])
        mock_load.return_value = (file_path, file_data, 10, 16000)
        mock_fingerprint.return_value = arr
        ret = crunch._read_and_fingerprint(file_path, self.def_ap)
        mock_load.assert_called_with(file_path, 0)
        mock_fingerprint.assert_called_with(file_data, 16000, 10)
        numpy.testing.assert_array_equal(ret[0], arr)
        self.assertEqual(ret[1], file_path)
        self.assertEqual(ret[2], 10)

    @mock.patch("crunch.all_files")
    @mock.patch("crunch._read_and_fingerprint", mock_read)
    def test_read_data_to_fingerprints(self, mock_all_files):
        file_name = "mock_file.wav"
        mock_all_files.return_value = [file_name]
        arr = numpy.asarray([1, 3, 3, 7])
        res = crunch._read_data_to_fingerprints(self.def_ap)
        mock_all_files.assert_called_with(os.getcwd(), [".wav"])
        numpy.testing.assert_array_equal(res[0][0], arr)
        self.assertEqual(res[1][0], file_name)

    def test_run_dimensionality_reduction(self):
        rf = mock.MagicMock()
        crunch.ProcessFunctions.dimensionality_reduction_dict["tsne"] = rf
        rf.return_value = "mock_result"
        arr = numpy.asarray([1, 3, 3, 7])
        res = crunch._run_dimensionality_reduction([arr], self.def_ap, "mock_data", "mock_dict")
        self.assertEqual(res, "mock_result")
        numpy.testing.assert_array_equal(rf.call_args[0][0][0], arr)
        self.assertEqual(rf.call_args[0][1], 3)
        self.assertEqual(rf.call_args[0][2], self.def_ap)
        self.assertEqual(rf.call_args[1]["a_func"], crunch._finalize)
        self.assertSequenceEqual(rf.call_args[1]["a_params"], (self.def_ap, "mock_data", "mock_dict"))

    @mock.patch("crunch._read_data_to_fingerprints")
    @mock.patch("numpy.save")
    def test_generate_fingerprints(self, mock_save, mock_read_fingerprints):
        mock_read_fingerprints.return_value = [numpy.asarray([[1, 3], [3, 7]]), numpy.asarray([[7, 3], [3, 1]])], \
                                              ["file1", "file2"]
        res = crunch.generate_fingerprints(self.def_ap)
        mock_read_fingerprints.assert_called_with(self.def_ap)
        self.assertFalse(mock_save.called)
        numpy.testing.assert_array_equal(res[0], numpy.asarray([[1, 3, 3, 7], [7, 3, 3, 1]]).astype(numpy.float64))
        self.assertSequenceEqual(res[1], ["file1", "file2"])

    @mock.patch("crunch._read_data_to_fingerprints")
    @mock.patch("numpy.save")
    def test_generate_fingerprints_save(self, mock_save, mock_read_fingerprints):
        mock_read_fingerprints.return_value = [numpy.asarray([[1, 3], [3, 7]]), numpy.asarray([[7, 3], [3, 1]])], \
                                              ["file1", "file2"]
        res = crunch.generate_fingerprints(self.val_ap)
        mock_read_fingerprints.assert_called_with(self.val_ap)
        self.assertEqual(mock_save.call_args[0][0], "fing.npy")
        self.assertEqual(mock_save.call_args[0][1][0][0], "file1")
        numpy.testing.assert_array_equal(mock_save.call_args[0][1][0][1],
                                         numpy.asarray([1, 3, 3, 7]).astype(numpy.float64))
        self.assertEqual(mock_save.call_args[0][1][1][0], "file2")
        numpy.testing.assert_array_equal(mock_save.call_args[0][1][1][1],
                                         numpy.asarray([7, 3, 3, 1]).astype(numpy.float64))
        numpy.testing.assert_array_equal(res[0], numpy.asarray([[1, 3, 3, 7], [7, 3, 3, 1]]).astype(numpy.float64))
        self.assertSequenceEqual(res[1], ["file1", "file2"])
