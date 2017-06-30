from unittest import mock, TestCase
import numpy
import subprocesses


class TestFingerprintFormData(TestCase):
    def test_fingerprint_form_data(self):
        r = subprocesses.fft_fingerprint(
            numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]), 16000, 500)
        self.assertEqual(r.shape, (32, 32), "unexpected result shape")

    @mock.patch("librosa.feature.mfcc")
    def test_mfcc_fingerprint(self, mock_mfcc):
        data = "mock_data"
        mock_mfcc.return_value = "mock_return"
        res = subprocesses.mfcc_fingerprint(data, 16000, 16000)
        self.assertEqual(res, "mock_return")
        mock_mfcc.assert_called_with(data, 16000, hop_length=640, n_mfcc=20)

    @mock.patch("librosa.feature.chroma_stft")
    def test_chroma_fingerprint(self, mock_chroma):
        data = "mock_data"
        mock_chroma.return_value = "mock_return"
        res = subprocesses.chroma_fingerprint(data, 16000, 16000)
        self.assertEqual(res, "mock_return")
        mock_chroma.assert_called_with(y=data, sr=16000, hop_length=640)

    @mock.patch("librosa.feature.tonnetz")
    def test_tonnez_fingerprint(self, mock_tonnez):
        data = numpy.asarray([1, 2, 3])
        mock_tonnez.return_value = "mock_return"
        res = subprocesses.tonnez_fingerprint(data, 16000, 16000)
        self.assertEqual(res, "mock_return")
        mock_tonnez.assert_called()
        numpy.testing.assert_array_equal(mock_tonnez.call_args[1]["y"], data.astype(numpy.float64))
        self.assertEqual(mock_tonnez.call_args[1]["sr"], 16000)

    @mock.patch("librosa.feature.melspectrogram")
    def test_ms_fingerprint(self, mock_ms):
        data = "mock_data"
        mock_ms.return_value = "mock_return"
        res = subprocesses.ms_fingerprint(data, 16000, 16000)
        self.assertEqual(res, "mock_return")
        mock_ms.assert_called_with(y=data, sr=16000, hop_length=640)
