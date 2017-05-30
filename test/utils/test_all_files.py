from unittest import mock, TestCase
from utils import all_files


class TestAllFiles(TestCase):

    @mock.patch("os.walk")
    def test_all_files(self, mock_os_walk):
        pa = "root/"
        mock_os_walk.return_value = iter([(pa, [], ['TP_VerbKick-F#3.wav', 'TP_RoomaKick-C4.wav', "data.csv"])])
        res = list(all_files(pa, [".wav"]))
        self.assertEqual(len(res), 2, "Unexpected number of elements returned: {}.".format(len(res)))
        self.assertTrue("root/TP_VerbKick-F#3.wav" in res)
        self.assertTrue("root/TP_RoomaKick-C4.wav" in res)
        self.assertFalse("root/data.csv" in res)
        mock_os_walk.assert_called_with(pa)
