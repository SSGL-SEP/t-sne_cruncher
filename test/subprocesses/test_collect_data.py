from unittest import mock, TestCase
import numpy
from subprocesses.collect_data import collect_data, load_sample


class TestCollectData(TestCase):

    @mock.patch("scipy.io.wavfile.read")
    @mock.patch("os.walk")
    @mock.patch("builtins.print", autospec=True)
    def test_collect_data(self, mock_print, mock_os_walk, mock_wavfile_read):
        f_data = (16000, numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        mock_wavfile_read.return_value = f_data
        numpy.save = mock.MagicMock()
        mock_os_walk.return_value = iter([("root/", [], ['bla.wav'])])
        res = collect_data()
        self.assertTrue(mock_os_walk.called, "mock_os_walk not called")
        self.assertTrue(mock_print.called, "mock_print not called")
        self.assertFalse(numpy.save.called, "numpy save called")
        self.assertEqual(res[0][2], 8000, "Incorrect audio length")

    @mock.patch("scipy.io.wavfile.read")
    @mock.patch("os.walk")
    @mock.patch("builtins.print", autospec=True)
    def test_save_called_when_file_name_given(self, mock_print, mock_os_walk, mock_wavfile_read):
        f_data = (16000, numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        mock_wavfile_read.return_value = f_data
        numpy.save = mock.MagicMock()
        mock_os_walk.return_value = iter([("root/", [], ['bla.wav'])])
        collect_data(target_file="bla.npy")
        self.assertTrue(numpy.save.called, "numpy save not called")

    @mock.patch("scipy.io.wavfile.read")
    @mock.patch("builtins.print", autospec=True)
    def test_load_sample_reshape(self, mock_print, mock_wavfile_read):
        f_data = (10, numpy.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [10, 9], [8, 7], [6, 5], [4, 3], [2, 1]]))
        mock_wavfile_read.return_value = f_data
        res = load_sample((500, "bla.wav"))
        self.assertTrue(mock_print.called, "print not called")
        self.assertTrue(mock_wavfile_read.called, "wavfile read not called")
        self.assertTrue("bla" in res[0], "filename not in results")
        self.assertEqual(len(res[1]), 5, "Unexpected size of audio")
        self.assertEqual(res[2], 5, "Incorrect audio length in return")

