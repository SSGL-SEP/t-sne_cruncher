from unittest import mock, TestCase
from unittest.mock import call

import numpy

from subprocesses.dimensionality_reduction import t_sne, _t_sne_job
from crunch import _arg_parse


def mock_tsne_job(arr, perp, dims, a, b):
    return numpy.asarray([3, 2, 1]), "mock"


class TestCalculateTSNE(TestCase):

    def test_t_sne_job(self):
        ret = _t_sne_job(numpy.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 30, 2)
        self.assertEqual(ret[0].shape, (3, 2), "Invalid shape")

    @mock.patch("subprocesses.dimensionality_reduction._t_sne_job")
    def test_t_sne_serial(self, mock_t_sne_job):
        args = _arg_parse().parse_args(["-p", "30", "50"])
        rarr = numpy.asarray([1, 2])
        mock_t_sne_job.return_value = (rarr, "30")
        arr = numpy.asarray([1, 2, 3])
        res = t_sne(arr, 2, args)
        self.assertEqual(len(res), 2)
        mock_t_sne_job.assert_has_calls([call(arr, 30, 2, None, None),
                                         call(arr, 50, 2, None, None)])
        self.assertEqual(res[0], (rarr, "30"))
        self.assertEqual(res[1], (rarr, "30"))

    @mock.patch("subprocesses.dimensionality_reduction._t_sne_job", mock_tsne_job)
    def test_t_sne_parallel(self):
        args = _arg_parse().parse_args(["-p", "30", "50", "--parallel"])
        arr = numpy.asarray([1, 2, 3])
        res = t_sne(arr, 2, args)
        rarr = numpy.asarray([3, 2, 1])
        self.assertEqual(len(res), 2)
        self.assertTrue(numpy.array_equal(res[0][0], rarr))
        self.assertEqual(res[0][1], "mock")
        self.assertTrue(numpy.array_equal(res[1][0], rarr))
        self.assertEqual(res[1][1], "mock")

    @mock.patch("subprocesses.dimensionality_reduction.TSNE")
    def test_t_sne_job_func_call(self, mock_sci_tsne):
        mock_func = mock.MagicMock()
        arr = numpy.asarray([1, 2, 3])
        params = (4, 5)
        res = _t_sne_job(arr, 30, 2, mock_func, params)
        mock_sci_tsne.assert_called_with(method='exact', n_components=2, perplexity=30, verbose=2)
        self.assertEqual(mock_func.call_args[0][0][1], "30")
        self.assertEqual(mock_func.call_args[0][1], 4)
        self.assertEqual(mock_func.call_args[0][2], 5)
        self.assertEqual(res[1], "30")
