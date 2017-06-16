from unittest import mock, TestCase
import numpy
import matplotlib.pyplot
import subprocesses
from subprocesses.dimensionality_reduction import t_sne, t_sne_job


class TestDimensionalityReduction(TestCase):

    def test_t_sne_job(self):
        params = (numpy.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 30, 2)
        ret = t_sne_job(params)
        self.assertEqual(ret[0].shape, (3, 2), "Invalid shape")
