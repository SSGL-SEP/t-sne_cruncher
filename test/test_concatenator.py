from unittest import mock, TestCase
import io
import audio_concatenator


class TestMain(TestCase):
    def test_main(self):
        pass
        # TODO: jotain


class TestParseArguments(TestCase):
    def test_parse_arguments(self):
        with mock.patch("builtins.open", mock.mock_open()) as mopen:
            with mock.patch("json.load") as mock_json_load:
                # mopen.return_value = mock.MagicMock(spec=io.IOBase)
                mock_json_load.return_value = {'points': [[0, 0, 0, 'xyz.wav']]}

                mock_binary = mock.MagicMock()
                mock_binary.read.return_value = map(bin, bytearray('jotain', 'utf8'))

                args = audio_concatenator._parse_arguments().parse_args(['whatever.json'])

                audio_concatenator.main(args)
                mopen.assert_called_with('xyz.mp3', 'rb')
