import errno
from unittest import mock, TestCase
from utils.mkdir_p import mkdir_p


class TestMkdirP(TestCase):

    @mock.patch("os.makedirs")
    def test_mkdir_p_default(self, mock_os_makedirs):
        mkdir_p("bla")
        mock_os_makedirs.assert_called_with("bla")

    @mock.patch("os.path.isdir")
    @mock.patch("os.makedirs")
    def test_mkdir_p_exists(self, mock_os_makedirs, mock_os_path_isdir):
        mock_os_path_isdir.return_value = True
        mock_os_makedirs.side_effect = OSError(errno.EEXIST, "Directory exists")
        mkdir_p("bla")
        mock_os_makedirs.assert_called_with("bla")

    @mock.patch("os.makedirs")
    def test_mkdir_p_error(self, mock_os_makedirs):
        e = OSError(errno.EPERM, "Access denied")
        mock_os_makedirs.side_effect = e
        self.assertRaises(OSError, mkdir_p, "bla")
