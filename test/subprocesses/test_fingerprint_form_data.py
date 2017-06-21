from unittest import mock, TestCase
import numpy
import subprocesses


class TestFingerprintFormData(TestCase):
    def test_fingerprint_form_data(self):
        r = subprocesses.fft_fingerprint(
            numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]), 16000, 500)
        self.assertEqual(r.shape, (32, 32), "unexpected result shape")
