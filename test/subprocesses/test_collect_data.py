from unittest import mock, TestCase
import numpy
from subprocesses.collect_data import load_sample


class TestCollectData(TestCase):

    @mock.patch("scipy.io.wavfile.read")
    def test_load_sample_reshape(self, mock_wavfile_read):
        f_data = (10, numpy.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [10, 9], [8, 7], [6, 5], [4, 3], [2, 1]]))
        mock_wavfile_read.return_value = f_data
        res = load_sample("bla.wav", 500)
        self.assertTrue(mock_wavfile_read.called, "wavfile read not called")
        self.assertTrue("bla" in res[0], "filename not in results")
        self.assertEqual(len(res[1]), 5, "Unexpected size of audio")
        self.assertEqual(res[2], 5, "Incorrect audio length in return")

    @mock.patch("scipy.io.wavfile.read")
    def test_load_sample_long(self, mock_wavfile_read):
        f_data = (100, numpy.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [10, 9], [8, 7], [6, 5], [4, 3], [2, 1]]))
        mock_wavfile_read.return_value = f_data
        res = load_sample("bla.wav", 500)
        self.assertTrue(mock_wavfile_read.called, "wavfile read not called")
        self.assertTrue("bla" in res[0], "filename not in results")
        self.assertEqual(len(res[1]), 10, "Unexpected size of audio")
        self.assertEqual(res[2], 10, "Incorrect audio length in return")

    @mock.patch("scipy.io.wavfile.read")
    def test_load_sample_all(self, mock_wavfile_read):
        f_data = (100, numpy.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [10, 9], [8, 7], [6, 5], [4, 3], [2, 1]]))
        mock_wavfile_read.return_value = f_data
        res = load_sample("bla.wav")
        self.assertTrue(mock_wavfile_read.called, "wavfile read not called")
        self.assertTrue("bla" in res[0], "filename not in results")
        self.assertEqual(len(res[1]), 10, "Unexpected size of audio")
        self.assertEqual(res[2], 10, "Incorrect audio length in return")
