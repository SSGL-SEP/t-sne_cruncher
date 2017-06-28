from unittest import TestCase

from utils import UnionFind


class TestUnionFind(TestCase):
    def setUp(self):
        l = ['a', 'b', 'c', 'd']
        self.uf = UnionFind(l)
        self.items = set(l)

    def test_init(self):
        self.assertEqual(self.uf.components, 4)
        self.assertTrue(all(x in self.uf.parents.keys() for x in self.items))
        self.assertTrue(all(x == 1 for x in self.uf.sizes.values()))

    def test_root(self):
        self.assertTrue(all([self.uf[x] == x for x in self.items]))
        with self.assertRaises(ValueError):
            self.uf['f']
        self.uf.union('a', 'b')
        self.assertEqual(self.uf['b'], 'a')

    def test_find(self):
        self.assertFalse(self.uf.find('a', 'b'))
        with self.assertRaises(ValueError):
            self.uf.find('a', 'g')
        self.uf.union('a', 'b')
        self.assertTrue(self.uf.find('a', 'b'))

    def test_union(self):
        with self.assertRaises(ValueError):
            self.uf.union('a', 'g')
        self.assertEqual(self.uf.components, 4)
        self.assertTrue(self.uf.union('a', 'b'))
        self.assertEqual(self.uf.components, 3)
        self.assertEqual(self.uf['b'], 'a')
        self.assertTrue(self.uf.union('c', 'a'))
        self.assertEqual(self.uf.components, 2)
        self.assertEqual(self.uf['c'], 'a')
        self.assertFalse(self.uf.union('b', 'c'))
        self.assertEqual(self.uf.components, 2)
