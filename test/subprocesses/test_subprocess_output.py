from unittest import TestCase, mock
import os

import numpy

from subprocesses.dimensionality_reduction import _get_colors, plot_results


class TestColorMapping(TestCase):
    def setUp(self):
        self.manhattan = [6, 11, -1, 21, 14]
        self.arr = numpy.asarray([[1, 2, 3], [2, 5, 4], [3, 4, -8], [4, 11, 6], [5, 9]])
        self.metadata = {"test_tag": {"__filterable": True,
                                      "v1":
                                          {"points": [0, 2, 3],
                                           "color": "#ffffff"},
                                      "v2":
                                          {"points": [1, 4],
                                           "color": "#ff0000"},
                                      }}

    def test_color_by_manhattan(self):
        res = _get_colors(self.arr)
        self.assertSequenceEqual(res, self.manhattan)

    def test_color_by_metadata(self):
        res = _get_colors(self.arr, self.metadata, "test_tag")
        self.assertSequenceEqual(res, [(1.0, 1.0, 1.0),
                                       (1.0, 0.0, 0.0),
                                       (1.0, 1.0, 1.0),
                                       (1.0, 1.0, 1.0),
                                       (1.0, 0.0, 0.0)])

    def test_color_missing_metadata(self):
        res = _get_colors(self.arr, None, "test_tag")
        self.assertSequenceEqual(res, self.manhattan)

    def test_color_missing_colorby(self):
        res = _get_colors(self.arr, self.metadata)
        self.assertSequenceEqual(res, self.manhattan)


class TestPlotOutput(TestCase):
    def setUp(self):
        self.arr = numpy.asarray([[1, 2, 3], [2, 5, 4], [3, 4, -8], [4, 11, 6], [5, 9]])

    @mock.patch("subprocesses.dimensionality_reduction.plt")
    def test_plot_results(self, mock_plot):
        plot_results(self.arr)
        mock_plot.figure.assert_called_with(figsize=(16, 16))
        unnamed, named = mock_plot.scatter.call_args
        self.assertSequenceEqual(unnamed[0], [1, 2, 3, 4, 5])
        self.assertSequenceEqual(unnamed[1], [2, 5, 4, 11, 9])
        self.assertSequenceEqual(named["c"], [6, 11, -1, 21, 14])
        self.assertEqual(named["s"], 20)
        mock_plot.tight_layout.assert_called()
        mock_plot.savefig.assert_called_with(os.path.join(os.getcwd(), "prints.png"))
        mock_plot.close.assert_called()

