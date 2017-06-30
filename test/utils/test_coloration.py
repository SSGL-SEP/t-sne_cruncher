from math import sqrt
from unittest import TestCase

import numpy

from utils.coloration import add_color, ColorData, Edge, select_colors


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

    def test_color_data_init(self):
        cd = ColorData(100)
        self.assertEqual(cd.start_index, 0, "Unexpected start index {}".format(cd.start_index))
        self.assertEqual(len(cd.colors), 100, "Unexpexted number of colors generated {}".format(len(cd.colors)))
        self.assertEqual(len(cd.color_usages.keys()), 100, "Unexpected color usage dictionary length {}".format(
                                                           len(cd.color_usages.keys())))
        self.assertEqual(sum(cd.color_usages.values()), 100, "Unexpected color usage dictionary contents."
                                                             "Expected 100 True values got {}".format(
                                                             sum(cd.color_usages.values())))
        self.assertEqual(max(cd.color_indexes.values()), 99, "Unexpected maximum index in index dictionary."
                                                             "Expected 99, got {}".format(cd.color_indexes.values()))

    def test_color_data_assign(self):
        cd = ColorData(100)
        s = set()
        for i in range(100):
            s.add(cd.assign())
        self.assertEqual(len(s), 100, "Unexpected number of unique colors provided {}".format(len(s)))
        self.assertEqual(cd.start_index, 99, "Unexpected start index {}".format(cd.start_index))
        s.add(cd.assign())
        self.assertEqual(len(s), 101, "Default color not added")
        self.assertTrue("#ffffff" in s, "Default color not assigned")
        self.assertEqual(cd.assign(), "#ffffff", "Default color not assigned")

    def test_color_data_assing_distant(self):
        cd = ColorData(100)
        s = set()
        c = cd.assign()
        s.add(c)
        for i in range(99):
            s.add(cd.assign_distant(c))
        self.assertEqual(len(s), 100, "Unexpected number of unique colors provided {}".format(len(s)))
        self.assertEqual(cd.start_index, 0, "Unexpected start index {}".format(cd.start_index))
        s.add(cd.assign_distant(c))
        self.assertEqual(len(s), 101, "Default color not added")
        self.assertTrue("#ffffff" in s, "Default color not assigned")
        self.assertEqual(cd.assign(), "#ffffff", "Default color not assigned")

    def test_edge_init(self):
        e = Edge("sn", numpy.asarray([1, 1]), "tn", numpy.asarray([2, 2]))
        self.assertEqual(e.start, "sn")
        self.assertEqual(e.end, "tn")
        self.assertEqual(e.start_coordinate[0], 1)
        self.assertEqual(e.start_coordinate[1], 1)
        self.assertEqual(e.end_coordinate[0], 2)
        self.assertEqual(e.end_coordinate[1], 2)
        self.assertAlmostEqual(e.weight, sqrt(2))

    def test_edge_comparisons(self):
        e1 = Edge("n1", numpy.asarray([1, 1]), "n2", numpy.asarray([2, 2]))
        e2 = Edge("n2", numpy.asarray([2, 2]), "n3", numpy.asarray([3, 3]))
        e3 = Edge("n1", numpy.asarray([1, 1]), "n3", numpy.asarray([3, 3]))
        with self.assertRaises(TypeError):
            e1 == "str"
        self.assertEqual(e1, e2)
        self.assertNotEqual(e1, e3)
        self.assertGreater(e3, e1)
        self.assertLess(e2, e3)
        self.assertGreaterEqual(e3, e2)
        self.assertGreaterEqual(e1, e2)
        self.assertLessEqual(e1, e3)
        self.assertLessEqual(e1, e2)

    def test_random_assign(self):
        cd = ColorData(10000, 0)
        self.assertTrue(cd.random_assign)
        self.assertEqual(len(cd.colors), 1528)
        self.assertEqual(cd.assign(), "#00e7ff")
        self.assertEqual(cd.assign_distant("#00e7ff"), "#ff1800")

    def test_select_colors(self):
        cd = ColorData(20)
        d = generate_data()["phoneme"]
        e1 = Edge('a', numpy.asarray([1, 1]), 'b', numpy.asarray([1, 2]))
        select_colors(e1, cd, d)
        self.assertTrue("color" in d['a'])
        self.assertTrue("color" in d['b'])
        self.assertTrue("color" not in d['c'])
        select_colors(e1, cd, d)
        self.assertTrue("color" in d['a'])
        self.assertTrue("color" in d['b'])
        self.assertTrue("color" not in d['c'])
        e2 = Edge('a', numpy.asarray([1, 1]), 'c', numpy.asarray([2, 1]))
        select_colors(e2, cd, d)
        self.assertTrue("color" in d['a'])
        self.assertTrue("color" in d['b'])
        self.assertTrue("color" in d['c'])

    def test_color_data_string(self):
        cd = ColorData(10)
        print(cd)
        self.assertEqual(str(cd), """Current index: 0. Colors assigned: 0. Total colors 10.
        Colors: ['#ff0000', '#ff9900', '#cbff00', '#33ff00', '#00ff66', '#00ffff', '#0066ff', '#3200ff', '#cc00ff', \
'#ff0098']
        Usages: {'#ff0000': True, '#ff9900': True, '#cbff00': True, '#33ff00': True, '#00ff66': True, '#00ffff': True, \
'#0066ff': True, '#3200ff': True, '#cc00ff': True, '#ff0098': True}
        Indexes: {'#ff0000': 0, '#ff9900': 1, '#cbff00': 2, '#33ff00': 3, '#00ff66': 4, '#00ffff': 5, '#0066ff': 6, \
'#3200ff': 7, '#cc00ff': 8, '#ff0098': 9}"""
                         )
