from unittest import mock, TestCase
import io
import struct

import audio_concatenator


class TestMain(TestCase):
    @mock.patch("audio_concatenator._read_points")
    @mock.patch("audio_concatenator._write_to_file")
    def test_main(self, mock_write, mock_read_points):
        mock_read_points.return_value = [[1, 2, 3, "file_1.wav"]]
        with mock.patch("builtins.open", mock.mock_open(), create=True) as mopen:
            mock_file = mock.MagicMock()
            mopen.return_value = mock_file
            audio_concatenator.main(audio_concatenator._parse_arguments().parse_args(["mock.json"]))
            mopen.assert_called_with("concatenated_sounds.blob", 'wb')
        mock_read_points.assert_called_with("mock.json")
        self.assertTrue(mock_write.called)
        self.assertEqual(mock_write.call_args[0][1], "./file_1.mp3")

    @mock.patch("audio_concatenator.getsize")
    def test_write_to_file(self, mock_getsize):
        mock_getsize.return_value = 117
        mock_output = mock.MagicMock()
        with mock.patch("builtins.open", mock.mock_open(), create=True) as mopen:
            audio_concatenator._write_to_file(mock_output, "mock.mp3")
            mopen.assert_called_with("mock.mp3", 'rb')
        mock_getsize.asset_called_with("mock.mp3")
        self.assertTrue(mock_output.write.called)

    @mock.patch("json.load")
    def test_read_points(self, mock_load):
        mock_load.return_value = {"points": "mock"}
        with mock.patch("builtins.open", mock.mock_open(), create=True) as mopen:
            res = audio_concatenator._read_points("mock_path")
            mopen.assert_called()
        self.assertTrue(mock_load.called)
        self.assertEqual(res, "mock")


class TestParseArguments(TestCase):
    def test_parse_arguments_default(self):
        args = audio_concatenator._parse_arguments().parse_args(["mock.json"])
        self.assertEqual(args.json, "mock.json")
        self.assertEqual(args.ext, "mp3")
        self.assertEqual(args.input, ".")
        self.assertEqual(args.output, "concatenated_sounds.blob")

    def test_parse_arguments_values(self):
        args = audio_concatenator._parse_arguments().parse_args(
            ["mock.json", "-e", "wav", "-i", "subfolder/", "-o", "my_blob.blob"])
        self.assertEqual(args.json, "mock.json")
        self.assertEqual(args.ext, "wav")
        self.assertEqual(args.input, "subfolder/")
        self.assertEqual(args.output, "my_blob.blob")
