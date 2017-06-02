import os
from unittest import mock, TestCase

import numpy as np

import crunch


class PickableMock(mock.MagicMock):
    def __reduce__(self):
        return (mock.MagicMock, ())


class TestMain(TestCase):
    def _file_data_comp(self, data, index, x_coord, y_coord, z_coord, file_name, tags):
        self.assertEqual(data[0], index, "Unexpected index. Expected {}, Got{}".format(index, data[0]))
        self.assertEqual(data[1], x_coord, "Unexpected x coordinate. Expected {}, Got{}".format(x_coord, data[1]))
        self.assertEqual(data[2], y_coord, "Unexpected y coordinate. Expected {}, Got{}".format(y_coord, data[2]))
        self.assertEqual(data[3], z_coord, "Unexpected z coordinate. Expected {}, Got{}".format(z_coord, data[3]))
        self.assertEqual(data[4], file_name, "Unexpected file name. Expected {}, Got{}".format(file_name, data[4]))
        self.assertEqual(data[5], tags, "Unexpected tag list. Expected {}, Got{}".format(tags, data[5]))

    def test_collect(self):
        data = [("root/fna.wav", 16000), ("root/fnb.wav", 16000), ("root/fnc.wav", 16000)]
        x_d = np.asarray([[1, 1], [2, 2, 2], [3, 3]])
        s = {"fna.wav": [{"key": "phonem", "val": "a"}]}
        lst = crunch.collect(data, x_d, s)
        self.assertEqual(len(lst), 3, "Unexpected element count. 3 expected. Was {}".format(len(lst)))
        self._file_data_comp(lst[0], 0, 1, 1, 0, "fna.wav", [{"key": "phonem", "val": "a"}])
        self._file_data_comp(lst[1], 1, 2, 2, 2, "fnb.wav", [])
        self._file_data_comp(lst[2], 2, 3, 3, 0, "fnc.wav", [])

    @mock.patch("crunch.parse_metadata")
    @mock.patch("crunch.add_color")
    def test_output(self, mock_add_color, mock_parse_metadata):
        mock_parse_metadata.return_value = {"bla.wav":
                                            [{"key": "phonem", "val": "a"}, {"key": "voice", "val": "voiced"}]}
        with mock.patch("builtins.open", mock.mock_open()) as mopen:
            with mock.patch("json.dump") as mock_json_dump:
                crunch.output("root/st.json", [("bla.wav", 1)], [[1, 2, 3]], "bla.csv", None)
                self.assertTrue(mock_add_color.called)
                mock_parse_metadata.assert_called_with("bla.csv")
                mopen.assert_called_with("root/st.json", "w")
                self.assertTrue(mock_json_dump.called, "json dump not called")

    def test_main(self):
        pass
        # TODO: write test

    @mock.patch("scipy.io.wavfile.read")
    @mock.patch("crunch.all_files")
    def test_read_data_to_fingerprints(self, mock_all_files, mock_wav_load):
        mock_wav_load.return_value = (16000, np.asarray(range(16000)))
        mock_all_files.return_value = ["mock.wav"]
        res, fd = crunch._read_data_to_fingerprints(1000, "mock/")
        self.assertEqual(fd[0][0], "mock.wav", "Unexpected file name")
        self.assertEqual(fd[0][1], 16000, "Unexpected sample length")
        self.assertEqual(res[0].shape, (32, 32), "Invalid result array")

    @mock.patch("crunch.output")
    @mock.patch("crunch.normalize")
    @mock.patch("crunch.plot_t_sne")
    def test_finalize(self, mock_plot, mock_normalize, mock_output):
        ndarr = np.asarray([[1, 2, 3]])
        norarr = np.asarray([[0, 300, 600]])
        mock_normalize.return_value = norarr
        args = mock.MagicMock()
        args.plot_output = "mock.png"
        args.value_minimum = 0
        args.value_maximum = 600
        args.output_file = "mock.json"
        args.collect_metadata = None
        args.colorby = None
        crunch._finalize(ndarr, args, "mock data")
        mock_plot.assert_called_with(ndarr, "mock.png")
        mock_normalize.assert_called_with(ndarr, 0, 600)
        mock_output.assert_called_with("mock.json", "mock data", norarr, None, None)

    def test_arg_parse_defaults(self):
        ap = crunch._arg_parse()
        args = ap.parse_args(args=[])
        self.assertEqual(args.input_folder, os.getcwd(), "Unexpected input folder")
        self.assertEqual(args.perplexity, [30], "Unexpected perplexity list")
        self.assertEqual(args.output_file, os.path.join(os.getcwd(), "t_sne.json"), "Unexpected output directory")
        self.assertEqual(args.fingerprint_output, None, "Unexpected fingerprint output")
        self.assertEqual(args.value_minimum, 0, "Unexpected minimum value")
        self.assertEqual(args.value_maximum, 600, "Unexpected maximum value")
        self.assertEqual(args.plot_output, None, "Unexpected plot output")
        self.assertEqual(args.collect_metadata, None, "Unexpected metadata path")
        self.assertEqual(args.max_duration, 500, "Unexpected maximum duration")
        self.assertEqual(args.td, False, "Unexpected value for --td")
        self.assertEqual(args.colorby, None, "Unexpected colorby value")
        self.assertEqual(args.pca, False, "Unexpected value for --pca")

    def test_arg_parse_values(self):
        ap = crunch._arg_parse()
        args = ap.parse_args(args=["-f", "samp/", "-p", "20", "50", "60", "-o", "pca.json", "-r", "fing.npy",
                                   "-m", "-100", "-x", "100", "-t", "pca.png", "-c", "samp/data.csv", "-d", "4000",
                                   "--td", "--colorby", "phonem", "--pca"])
        self.assertEqual(args.input_folder, "samp/", "Unexpected input folder")
        self.assertEqual(args.perplexity, [20, 50, 60], "Unexpected perplexity list")
        self.assertEqual(args.output_file, "pca.json", "Unexpected output directory")
        self.assertEqual(args.fingerprint_output, "fing.npy", "Unexpected fingerprint output")
        self.assertEqual(args.value_minimum, -100, "Unexpected minimum value")
        self.assertEqual(args.value_maximum, 100, "Unexpected maximum value")
        self.assertEqual(args.plot_output, "pca.png", "Unexpected plot output")
        self.assertEqual(args.collect_metadata, "samp/data.csv", "Unexpected metadata path")
        self.assertEqual(args.max_duration, 4000, "Unexpected maximum duration")
        self.assertEqual(args.td, True, "Unexpected value for --td")
        self.assertEqual(args.colorby, "phonem", "Unexpected colorby value")
        self.assertEqual(args.pca, True, "Unexpected value for --pca")
