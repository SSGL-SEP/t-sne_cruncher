from unittest import TestCase
import numpy as np
from utils import normalize


class TestNormalize(TestCase):
    def test_normalize(self):
        arr = np.asarray([
            np.asarray([1, 2]),
            np.asarray([3, 4])]).astype(np.float64)
        arr = normalize(arr, 10, 100)
        self.assertAlmostEqual(np.amax(arr), 100)
        self.assertAlmostEqual(np.amin(arr), 10)
