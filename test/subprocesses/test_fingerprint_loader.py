from unittest import TestCase, mock

import numpy as np
from numpy.testing import assert_array_equal

from crunch import _arg_parse, load_fingerprints


class TestFingerprintLoader(TestCase):
    def test_load_csv_fingerprint(self):
        with mock.patch("builtins.open", mock.mock_open()) as m:
            with mock.patch("csv.reader") as mock_csv_reader:
                args = _arg_parse().parse_args(["-e", "mock.csv", "--format", "csv"])
                mock_csv_reader.return_value = [["name_1", "1", "2", "3"],
                                                ["name_2", "4", "5", "6"]]
                results, file_data = load_fingerprints(args)
                m.assert_called_with("mock.csv")
                self.assertSequenceEqual(file_data, ["name_1", "name_2"])
                self.assertEqual(len(results), 2)
                assert_array_equal(results[0], [1, 2, 3])
                assert_array_equal(results[1], [4, 5, 6])

    def test_load_tsv_fingerprint(self):
        with mock.patch("builtins.open", mock.mock_open()) as m:
            with mock.patch("csv.reader") as mock_csv_reader:
                args = _arg_parse().parse_args(["-e", "mock.tsv", "--format", "tsv"])
                mock_csv_reader.return_value = [["name_1", "1", "2", "3"],
                                                ["name_2", "4", "5", "6"]]
                results, file_data = load_fingerprints(args)
                m.assert_called_with("mock.tsv")
                self.assertSequenceEqual(file_data, ["name_1", "name_2"])
                self.assertEqual(len(results), 2)
                assert_array_equal(results[0], [1, 2, 3])
                assert_array_equal(results[1], [4, 5, 6])

    def test_load_numpy_fingerprint(self):
        with mock.patch("numpy.load", mock.mock_open()) as mock_load:
            args = _arg_parse().parse_args(["-e", "mock.npy", "--format", "npy"])
            mock_load.return_value = [("name_1", np.asarray([1, 2, 3])), ("name_2", np.asarray([4, 5, 6]))]
            results, file_data = load_fingerprints(args)
            mock_load.assert_called_with("mock.npy")
            self.assertSequenceEqual(file_data, ["name_1", "name_2"])
            self.assertEqual(len(results), 2)
            assert_array_equal(results[0], [1, 2, 3])
            assert_array_equal(results[1], [4, 5, 6])
