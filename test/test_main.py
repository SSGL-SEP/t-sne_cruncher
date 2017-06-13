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

    @mock.patch("crunch.add_color")
    def test_collect(self, mock_add_color):
        ap = crunch._arg_parse()
        args = ap.parse_args(args=["--colorby", "phoneme", "-s", "mock_sound_info.json", "-n",
                                   "mock_nsynth", "-e", "mock - pca"])
        data = [("root/fna.wav", 16000), ("root/fnb.wav", 16000), ("root/fnc.wav", 16000)]
        x_d = numpy.asarray([[1, 1], [2, 2, 2], [3, 3]])
        s = {"phoneme": {"a": {"color": "#ffffff", "points": [0, 1, 2]}, "__filterable":  True, }}
        d = crunch.collect(data, x_d, s, args)
        mock_add_color.assert_called()
        self.assertEqual(len(d.keys()), 7, "Unexpected number of keys")
        self.assertEqual(d["totalPoints"], 3, "unexpected number in totalPoints")
        self.assertEqual(d["soundInfo"], "mock_sound_info.json")
        self.assertEqual(d["dataSet"], "mock_nsynth")
        self.assertEqual(d["processingMethod"], "mock - pca")
        self.assertEqual(d["colorBy"], "phoneme")
        self.assertEqual(len(d["tags"].keys()), 1)
        self.assertEqual(len(d["points"]), 3)

    @mock.patch("crunch.add_color")
    def test_output(self, mock_add_color):
        s = {"phoneme": {"a": {"color": "#ffffff", "points": [0, 1, 2]}, "__filterable": True, }}
        with mock.patch("builtins.open", mock.mock_open()) as mopen:
            with mock.patch("json.dump") as mock_json_dump:
                crunch.output("root/st.json", [("bla.wav", 1)], numpy.asarray([[1, 2, 3]]),
                              crunch._arg_parse().parse_args(), s)
                self.assertTrue(mock_add_color.called)
                mopen.assert_called_with("root/st.json", "w")
                self.assertTrue(mock_json_dump.called, "json dump not called")

    @mock.patch("crunch.parse_metadata")
    @mock.patch("crunch._read_data_to_fingerprints")
    @mock.patch("crunch.PCA")
    @mock.patch("crunch._finalize")
    def test_main_pca(self, mock_finalize, mock_PCA, mock_read, mock_parse_metadata):
        s = {"phoneme": {"a": {"color": "#ffffff", "points": [0, 1]}, "__filterable": True, }}
        mock_parse_metadata.return_value = s
        numpy.save = mock.MagicMock()
        model = mock.MagicMock()
        out = "mock_results"
        model.fit_transform.return_value = numpy.asarray(out)
        mock_PCA.return_value = model
        res = [numpy.asarray([[1, 2], [3, 4]]), numpy.asarray([[2, 3], [4, 5]])]
        fd = [("1.wav", 4), ("2.wav", 4)]
        mock_read.return_value = (res, fd)
        ap = crunch._arg_parse()
        args = ap.parse_args(args=["-f", "mock/", "-o", "mock.json", "-r", "mock_fing.npy", "-m", "-100", "-x",
                                   "100", "-t", "mock.png", "-c", "mock/mock.csv", "-d", "4000",
                                   "--td", "--colorby", "phonem", "--pca"])
        crunch.main(args)
        mock_parse_metadata.assert_called()
        mock_read.assert_called_with(4000, "mock/", 0, False)
        self.assertTrue(numpy.save.called, "Save not called")
        mock_PCA.assert_called_with(n_components=2, svd_solver="full")
        self.assertTrue(model.fit_transform.called, "PCA fit transform not called")
        mock_finalize.assert_called_with(out, args, fd, s)

    @mock.patch("crunch.parse_metadata")
    @mock.patch("crunch._read_data_to_fingerprints")
    @mock.patch("crunch.t_sne")
    @mock.patch("crunch._finalize")
    def test_main_t_sne(self, mock_finalize, mock_t_sne, mock_read, mock_parse_metadata):
        s = {"phoneme": {"a": {"color": "#ffffff", "points": [0, 1]}, "__filterable": True, }}
        mock_parse_metadata.return_value = s
        res = [numpy.asarray([[1, 2], [3, 4]]), numpy.asarray([[2, 3], [4, 5]])]
        fd = [("1.wav", 4), ("2.wav", 4)]
        mock_read.return_value = (res, fd)
        mock_t_sne.return_value = [("mock1val1", "mock1val2"), ("mock2val1", "mock2val2")]
        ap = crunch._arg_parse()
        args = ap.parse_args(args=["-f", "mock/", "-o", "mock.json", "-m", "-100", "-x",
                                   "100", "-t", "mock.png", "-c", "mock/mock.csv", "-d", "4000",
                                   "--colorby", "phonem"])
        crunch.main(args)
        mock_parse_metadata.assert_called()
        mock_read.assert_called_with(4000, "mock/", 0, False)
        self.assertTrue(mock_t_sne.called, "t_SNE not called")
        mock_finalize.assert_called_with("mock2val1", args, fd, s, "mock2val2")

    @mock.patch("scipy.io.wavfile.read")
    @mock.patch("crunch.all_files")
    def test_read_data_to_fingerprints(self, mock_all_files, mock_wav_load):
        mock_wav_load.return_value = (16000, numpy.asarray(range(16000)))
        mock_all_files.return_value = ["mock.wav"]
        res, fd = crunch._read_data_to_fingerprints(1000, "mock/", True)
        self.assertEqual(fd[0][0], "mock.wav", "Unexpected file name")
        self.assertEqual(fd[0][1], 16000, "Unexpected sample length")
        self.assertEqual(res[0].shape, (20, 32), "Invalid result array")

    @mock.patch("crunch.output")
    @mock.patch("crunch.normalize")
    @mock.patch("crunch.plot_t_sne")
    def test_finalize(self, mock_plot, mock_normalize, mock_output):
        ndarr = numpy.asarray([[1, 2, 3]])
        norarr = numpy.asarray([[0, 300, 600]])
        mock_normalize.return_value = norarr
        args = mock.MagicMock()
        args.plot_output = "mock.png"
        args.value_minimum = 0
        args.value_maximum = 600
        args.output_file = "mock.json"
        args.collect_metadata = None
        args.colorby = None
        data = list([numpy.asarray([1, 2, 3]),
                     numpy.asarray([2, 3, 4]),
                     numpy.asarray([4, 5, 6])])
        s = {"phoneme": {"a": {"color": "#ffffff", "points": [0, 1, 2]}, "__filterable": True, }}
        crunch._finalize(ndarr, args, data, s)
        mock_plot.assert_called_with(ndarr, "mock.png")
        mock_normalize.assert_called_with(ndarr, 0, 600)
        mock_output.assert_called_with("mock.json", data, norarr, args, s)

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
        self.assertEqual(args.duration, 500, "Unexpected maximum duration")
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
        self.assertEqual(args.duration, 4000, "Unexpected maximum duration")
        self.assertEqual(args.td, True, "Unexpected value for --td")
        self.assertEqual(args.colorby, "phonem", "Unexpected colorby value")
        self.assertEqual(args.pca, True, "Unexpected value for --pca")
