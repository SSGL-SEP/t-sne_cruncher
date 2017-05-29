from unittest import mock, TestCase
import numpy
import subprocesses


class TestFingerprintFormData(TestCase):
    def test_fingerprint_form_data(self):
        r = subprocesses.fingerprint_form_data(
            numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))
        self.assertEqual(r.shape, (32, 32), "unexpected result shape")

    def test_fingerprint_from_file(self):
        arr = numpy.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        numpy.load = mock.MagicMock(return_value=arr)
        numpy.save = mock.MagicMock()
        r = subprocesses.fingerprint_from_file_data(source_npy="samples.npy", target_npy_file="trg.npy")
        self.assertEqual(r.shape, (2, 32, 32), "unexpected result shape")
        self.assertTrue(numpy.load.called, "numpy load not called")
        self.assertTrue(numpy.save.called, "numpy save not called")
