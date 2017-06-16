from unittest import mock, TestCase

from utils import *
from crunch import _arg_parse


class TestParseMetadata(TestCase):

    def test_parse_metadata(self):
        with mock.patch("builtins.open", mock.mock_open()) as m:
            with mock.patch("csv.reader") as mock_csv_reader:
                args = _arg_parse().parse_args(["-c", "mock_location.csv", "-u", "c"])
                mock_csv_reader.return_value = [["filename", "b", "c"], ["f.wav", "v", "y"]]
                d = parse_metadata(args, {"f.wav": 0})
                m.assert_called_with("mock_location.csv", "r")
                self.assertFalse("filename" in d, "'filename' was not properly ignored")

                self.assertTrue("b" in d)
                self.assertTrue("__filterable" in d["b"], "'__filterable' not in 'a'")
                self.assertTrue(d["b"]["__filterable"], "'__filterable' in 'b' is not True")
                self.assertTrue("v" in d["b"], "'v' not in 'b'")
                self.assertTrue("points" in d["b"]["v"], "'points' not in 'b'.'v'")
                self.assertEqual(d["b"]["v"]["points"], [0], "Invalid point list for 'b'")

                self.assertTrue("c" in d, "'c' not in dictionary")
                self.assertTrue("__filterable" in d["c"], "'__filterable' not in 'c'")
                self.assertFalse(d["c"]["__filterable"], "'__filterable' in 'c' is not False")
                self.assertTrue("y" in d["c"], "'y' not in 'c'")
                self.assertEqual(d["c"]["y"]["points"], [0], "Invalid point list for 'c'")
                self.assertTrue("points" in d["c"]["y"], "'points' not in 'c'.'y'")
