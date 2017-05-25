from unittest import mock, TestCase
from utils import *


class TestParseMetadata(TestCase):

    def test_parse_metadata(self):
        with mock.patch("builtins.open", mock.mock_open()) as m:
            with mock.patch("csv.reader") as mock_csv_reader:
                mock_csv_reader.return_value = [["a", "b"], ["f.wav", "v"]]
                d = parse_metadata("path")
                m.assert_called_with("path", "r")
                self.assertTrue("f.wav" in d)
                self.assertTrue(d["f.wav"][0]["key"] == "a")
                self.assertTrue(d["f.wav"][0]["val"] == "f.wav")
                self.assertTrue(d["f.wav"][1]["key"] == "b")
                self.assertTrue(d["f.wav"][1]["val"] == "v")
