from unittest import mock, TestCase
import numpy
import matplotlib.pyplot
import subprocesses
from subprocesses.calculate_tsne import t_sne


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
        t = type(res)
        a = res[0]
        s = type(a)
        b = a[0]
        v = type(b)
        print(t, s, v)
        self.assertEqual(res[0][0].shape[1], 3, "Invalid output shape: {}".format(res[0][0].shape))

    def test_save_tsv(self):
        ar = numpy.asarray([[1, 2, 3, 1, 2, 3, 1, 2, 3],
                            [1, 2, 3, 1, 2, 3, 1, 2, 3],
                            [1, 2, 3, 1, 2, 3, 1, 2, 3]]).astype(numpy.float32)
        with mock.patch("numpy.savetxt") as mock_savetxt:
            t_sne(ar, perplexity=[32], output_file="bla.tsv")
            self.assertEqual(mock_savetxt.call_args[0][0], "bla32.tsv", "Save not called or called with bad arguments.")
