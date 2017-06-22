from unittest import mock, TestCase

import numpy

from subprocesses.dimensionality_reduction import pca
from crunch import _arg_parse


class TestCalculatePCA(TestCase):
    def test_pca_job(self):
        args = _arg_parse().parse_args()
        arr = numpy.asarray([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        ret = pca(arr, 2, args)
        self.assertEqual(ret[0][0].shape, (3, 2), "invalid shape")
        self.assertEqual(ret[0][1], "pca")
        self.assertEqual(len(ret), 1)

    @mock.patch("subprocesses.dimensionality_reduction.PCA")
    def test_pca_func_call(self, mock_sci_pca):
        mock_func = mock.MagicMock()
        args = _arg_parse().parse_args()
        arr = numpy.asarray([1, 2, 3])
        params = (4, 5)
        res = pca(arr, 2, args, mock_func, params)
        mock_sci_pca.assert_called_with(n_components=2, svd_solver='full')
        self.assertEqual(mock_func.call_args[0][0][1], "pca")
        self.assertEqual(mock_func.call_args[0][1], 4)
        self.assertEqual(mock_func.call_args[0][2], 5)
        self.assertEqual(res[0][1], "pca")
