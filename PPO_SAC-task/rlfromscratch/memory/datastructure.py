import numpy as np
from numba import njit

# We directly use the implementation in tianshou for reference, which use njit to accelerate running, which is really helpful for prioritized replay buffer.

class SumTree:
    def __init__(self, size:int):
        bound = 1
        while bound < size:
            bound *= 2
        # bound = min_{2^n >= size} n
        self.size = size
        self.bound = bound
        self.tree = np.zeros([bound * 2],dtype=np.float64) # This is the real sumtree datastructure, the last $bound elements are leaf priorities, the first $bound-1 elements are tree nodes.
        self._compile()

    def __getitem__(self, index: int | np.ndarray):
        return self.tree[index + self.bound] 

    def __setitem__(self, index: int| np.ndarray, value: float | np.ndarray):
        if isinstance(index, int):
            index, value = np.array([index]), np.array([value])
        assert np.all(index >= 0)
        assert np.all(index < self.size)
        _setitem(self.tree, index + self.bound, value)
    
    def get_prefix_idx(self, ratios:np.ndarray) -> np.ndarray:
        indexes = _get_prefix_idx(self.tree, self.bound, ratios)
        return indexes 

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        i64 = np.array([0, 1], dtype=np.int64)
        _setitem(f64, i64, f64)
        _setitem(f64, i64, f32)
        _get_prefix_idx(f64, 1, f64)
        _get_prefix_idx(f32, 1, f64)




    





@njit
def _setitem(tree: np.ndarray, index: np.ndarray, value: np.ndarray):
    tree[index] = value
    # from leaf to root setting the father node with father.val = left.val + right.val 
    while index[0] > 1:
        index //= 2
        tree[index] = tree[index * 2] + tree[index * 2 + 1]
    

@njit 
def _get_prefix_idx(tree:np.ndarray, bound:int, ratios:np.ndarray):
    # start from the root 
    indexes = np.ones(shape=ratios.shape, dtype=np.int64)
    
    while indexes[0] < bound:
        # go to the left child node 
        indexes *= 2
        left_vals = tree[indexes]
        # whether to the left or right node 
        is_right = ratios > left_vals 
        # if goto the left, then there is no need to substract the left_val, if goto the right node, it is required to substract the left_val 
        ratios -= left_vals * is_right

        indexes += is_right
    
    # remember to substract the bound
    return indexes - bound  