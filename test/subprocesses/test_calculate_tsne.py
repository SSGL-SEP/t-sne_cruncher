from unittest import mock, TestCase
import numpy
import matplotlib.pyplot
import subprocesses
from subprocesses.calculate_tsne import save_tsv


class TestCalculateTsne(TestCase):

    @mock.patch("builtins.print", autospec=True)
    def test_calculate_tsne(self, mock_print):
        npf = numpy.asarray([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                             [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                             [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]).astype(numpy.float32)
        numpy.load = mock.MagicMock(return_value=npf)
        matplotlib.pyplot.figure = mock.MagicMock()
        matplotlib.pyplot.scatter = mock.MagicMock()
        matplotlib.pyplot.tight_layout = mock.MagicMock()
        matplotlib.pyplot.close = mock.MagicMock()
        matplotlib.pyplot.savefig = mock.MagicMock()
        res = subprocesses.calculate_tsne()
        self.assertTrue(mock_print.called, "mock_print not called")
        self.assertTrue(res.shape[1] == 3, "Invalid output shape: {}".format(res.shape))

    def test_save_tsv(self):
        ar = numpy.asarray([1, 2, 3])
        with mock.patch("numpy.savetxt") as mock_savetxt:
            save_tsv(ar, "bla.tsv")
            mock_savetxt.assert_called_once_with("bla.tsv", ar, delimiter="\t", fmt="%.5f")
