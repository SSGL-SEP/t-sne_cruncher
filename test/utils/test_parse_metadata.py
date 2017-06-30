from unittest import mock, TestCase

from utils.utils import parse_metadata, _parse_row
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

    def test_parse_row(self):
        i_d = {"file1": 1, "file2": 2}
        h = ["name", "phoneme"]
        d = {"phoneme": {"__filterable": True}}
        ig = ["name"]
        _parse_row(d, h, ["file3", "a"], i_d, ig)
        self.assertTrue("a" not in d["phoneme"])
        _parse_row(d, h, ["file1", 'a'], i_d, ig)
        self.assertTrue("a" in d["phoneme"])
        self.assertTrue("points" in d["phoneme"]['a'])
        self.assertSequenceEqual(d["phoneme"]['a']["points"], [1])
        _parse_row(d, h, ["file2", 'a'], i_d, ig)
        self.assertTrue("a" in d["phoneme"])
        self.assertTrue("points" in d["phoneme"]['a'])
        self.assertSequenceEqual(d["phoneme"]['a']["points"], [1, 2])
        self.assertEqual(len(d.keys()), 1)
        self.assertEqual(len(d["phoneme"].keys()), 2)
