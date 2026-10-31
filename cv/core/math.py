from typing import Final

import numpy as np
from scipy.optimize import linear_sum_assignment

INF: Final[float] = 1e10


def rectangular_assignment(
    matrix: np.ndarray, max_threshold: float = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix[matrix > max_threshold] = INF
    row_indices, col_indices = linear_sum_assignment(matrix)

    all_rows = np.arange(matrix.shape[0])
    all_cols = np.arange(matrix.shape[1])

    valid_matches = matrix[row_indices, col_indices] < max_threshold
    row_indices = row_indices[valid_matches]
    col_indices = col_indices[valid_matches]

    unmatched_rows = np.setdiff1d(all_rows, row_indices)
    unmatched_cols = np.setdiff1d(all_cols, col_indices)

    return row_indices, col_indices, unmatched_rows, unmatched_cols
