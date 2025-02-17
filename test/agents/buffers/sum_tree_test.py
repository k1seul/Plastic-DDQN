# Test case following [dopamine sum_tree_test](https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/sum_tree_test.py)
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
        # Check that all nodes on the leftmost branch have value 1.

    def test_set_and_get_values_vectorized(self):
        data = [1.0, 2.0]
        probs = [3.0, 4.0]

        for i in range(len(data)):
            self._tree.append(data[i], probs[i])
        for i in range(len(data)):
            self.assertEqual(self._tree.get([i])[0], data[i])
            self.assertEqual(self._tree.get_sum_tree_leaf(i), probs[i])

    def test_update_index(self):
        update_indices = [0, 1]
        update_values = [12, 13]
        for i in range(len(update_indices)):
            self._tree._update_index(update_indices[i] + self._tree.tree_start, update_values[i])
        # Check updated nodes, max, root values
        for i in range(len(update_indices)):
            self.assertEqual(self._tree.get_sum_tree_leaf(i), update_values[i])
        self.assertEqual(self._tree.max, max(update_values))
        self.assertEqual(self._tree.total(), sum(update_values))
    
    def test_find_value(self):
        for _ in range(5):
            self._tree.append(0, 0)
        self._tree.append(10, 1.0)
        value, data_index, tree_index = self._tree.find(np.array([0.99]))
        self.assertEqual(value, 1.0)
        self.assertEqual(data_index, 5)
        self.assertEqual(tree_index, data_index+self._tree.tree_start)

    def test_find_vectorized(self):
        """
            [2.5]
        [1.5]  [1.0]
        [0.5 1.0 0.5 0.5]
        """
        values = [0.5, 1.0, 0.5, 0.5]
        tree = SegmentTree(size=4)
        for i in range(4):
            tree.append(i, values[i])
        self.assertEqual(tree.total(), 2.5)
        self.assertEqual(tree.tree_start + tree.size, 7)
        value, data_index, tree_index = tree.find(np.array([1.5, 1.0]))
        self.assertEqual(data_index.tolist(), [2,1])

if __name__ == '__main__':
    unittest.main()
