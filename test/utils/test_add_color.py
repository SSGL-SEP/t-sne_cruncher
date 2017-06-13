from unittest import TestCase

import numpy

from utils import *


def generate_data():
    return {"phoneme": {"a": {"points": [0]},
                        "b": {"points": [1]},
                        "c": {"points": [2]}},
            "voice": {"voiced": {"points": [0, 1]},
                      "unvoiced": {"points": [2]}}}


def get_array():
    return numpy.asarray([
        [1, 1, 1],
        [1, 1, 2],
        [5, 5, 5]
    ])


class TestAddColor(TestCase):
    def test_add_color_by_tag(self):
        metadata = generate_data()
        arr = get_array()
        add_color(metadata, arr)
        print(metadata)
        self.assertEqual(metadata["phoneme"]["a"]["color"],
                         "#00ff00",
                         "{} has wrong color. #00ff00 expected. Got {}".format(
                             "a", metadata["phoneme"]["a"]["color"]))
        self.assertEqual(metadata["phoneme"]["b"]["color"],
                         "#ff0000",
                         "{} has wrong color. #ff0000 expected. Got {}".format(
                             "b", metadata["phoneme"]["b"]["color"]))
        self.assertEqual(metadata["phoneme"]["c"]["color"],
                         "#0000ff",
                         "{} has wrong color. #0000ff expected. Got {}".format(
                             "c", metadata["phoneme"]["c"]["color"]))
        self.assertEqual(metadata["voice"]["voiced"]["color"],
                         "#00ffff",
                         "{} has wrong color. #00ffff expected. Got {}".format(
                             "voiced", metadata["voice"]["voiced"]["color"]))
        self.assertEqual(metadata["voice"]["unvoiced"]["color"],
                         "#ff0000",
                         "{} has wrong color. #ff0000 expected. Got {}".format(
                             "unvoiced", metadata["voice"]["unvoiced"]["color"]))
