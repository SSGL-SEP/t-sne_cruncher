from unittest import TestCase

from utils import insert_suffix


class TestInsertSuffix(TestCase):
    def test_insert_suffix(self):
        self.assertEqual(insert_suffix("mock_.wav", "bla"), "mock_bla.wav")
