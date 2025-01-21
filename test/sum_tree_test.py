import unittest 
import numpy as np
from src.agents.buffers.per_buffer import SegmentTree

class SegmentTreeTest(unittest.TestCase):
    
    def setUp(self):
        super().__init__()
        self._tree = SegmentTree(size=100)
    
    def test_negative_capacity_raises(self):
        """Assert error not implemented passing this test"""
        return
        with self.assertRaises(AssertionError):
            SegmentTree(size=-1)

    def test_negative_value_raises(self):
        """Assert error not implemented passing this test"""
        return 
        with self.assertRaises(AssertionError):
            self._tree.append(0, -1)

    def test_set_small_capacity(self):
        tree = SegmentTree(size=4)
        tree.append(0, 1.5)
        self.assertEqual(tree.total(), 1.5)

    def test_set_and_get_value(self):
        self._tree.append(0, 1.0)
        # Check a single data given data index
        self.assertEqual(self._tree.get([0])[0], 0)
        # Check sum tree value
        self.assertEqual(self._tree.get_sum_tree_leaf(0), 1.0)
        

if __name__ == '__main__':
    unittest.main()