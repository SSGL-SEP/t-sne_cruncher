from unittest import TestCase

from utils import *


def generate_data():
    return [
        [1, 0, 0, 0, "fn1", [{"key": "phonem", "val": "a"}, {"key": "voice", "val": "voiced"}]],
        [2, 1, 1, 1, "fn2", [{"key": "phonem", "val": "c"}, {"key": "voice", "val": "voiced"}]],
        [3, 2, 2, 2, "fn3", [{"key": "phonem", "val": "b"}, {"key": "voice", "val": "voiced"}]]
    ]


def generate_missing_data():
    return [
        [1, 0, 0, 0, "fn1", [{"key": "phonem", "val": "a"}, {"key": "voice", "val": "voiced"}]],
        [2, 1, 1, 1, "fn2", [{"key": "phonem", "val": "c"}, {"key": "voice", "val": "voiced"}]],
        [3, 2, 2, 2, "fn3", [{"key": "phonem", "val": "b"}, {"key": "voice", "val": "voiced"}]],
        [4, 5, 5, 5, "fn4", [{"key": "voice", "val": "voiced"}]]
    ]


class TestAddColor(TestCase):
    def test_add_color(self):
        data = generate_data()
        add_color(data)
        print(data)
        self.assertEqual(data[0][6], "#ff0000", "{} has wrong color. #ff0000 expected. Got {}".format(
            data[0][4], data[0][6]
        ))
        self.assertEqual(data[1][6], "#00ffff", "{} has wrong color. #00ffff expected. Got {}".format(
            data[1][4], data[1][6]
        ))
        self.assertEqual(data[2][6], "#ff0000", "{} has wrong color. #ff0000 expected. Got {}".format(
            data[2][4], data[2][6]
        ))

    def test_add_color_by_tag(self):
        data = generate_missing_data()
        add_color(data, "phonem")
        print(data)
        self.assertEqual(data[0][6], "#ff0000", "{} has wrong color. #ff0000 expected. Got {}".format(
            data[0][4], data[0][6]
        ))
        self.assertEqual(data[1][6], "#0000ff", "{} has wrong color. #0000ff expected. Got {}".format(
            data[1][4], data[1][6]
        ))
        self.assertEqual(data[2][6], "#00ff00", "{} has wrong color. #00ff00 expected. Got {}".format(
            data[2][4], data[2][6]
        ))
        self.assertEqual(data[3][6], "#ffffff", "{} has wrong color. #ffffff expected. Got {}".format(
            data[3][4], data[3][6]
        ))
