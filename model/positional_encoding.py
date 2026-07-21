import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # 1. Create a blank grid of zeros
        PE = np.zeros((seq_len, d_model))
        
        # 2. Get positions as a vertical column: shape (seq_len, 1)
        position = np.arange(seq_len)[:, np.newaxis]
        
        # 3. Create frequencies for the pairs (steps by 2)
        div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
        
        # 4. Fill even columns with Sine and odd columns with Cosine
        half_d = d_model // 2
        PE[:, 0::2] = np.sin(position / div_term[:half_d])
        PE[:, 1::2] = np.cos(position / div_term[:half_d])
        
        return np.round(PE, 5)
