from collections.abc import Callable
from typing import Final

import numpy as np
from scipy.optimize import linear_sum_assignment

INF: Final[float] = 1e10

type FilterFn = Callable[[np.ndarray], np.ndarray]


def rectangular_assignment(
    matrix: np.ndarray, filters: list[FilterFn] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_indices, col_indices = linear_sum_assignment(matrix)

    all_rows = np.arange(matrix.shape[0])
    all_cols = np.arange(matrix.shape[1])

    if filters is not None:
        filter_mask = np.ones(shape=matrix.shape)
        for filter in filters:
            filter_mask &= filter(matrix)

        mask = filter_mask[row_indices, col_indices]
        row_indices = row_indices[mask]
        col_indices = col_indices[mask]

    unmatched_rows = np.setdiff1d(all_rows, row_indices)
    unmatched_cols = np.setdiff1d(all_cols, col_indices)

    return row_indices, col_indices, unmatched_rows, unmatched_cols


def convert_to_corner_coords(bboxes: np.ndarray) -> np.ndarray:
    new_bboxes = np.empty(shape=bboxes.shape)

    new_bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2
    new_bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] / 2

    return new_bboxes


def create_iou_matrix(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    corner_prev_bboxes = convert_to_corner_coords(pos1)
    corner_curr_bboxes = convert_to_corner_coords(pos2)

    b1 = corner_prev_bboxes[:, np.newaxis, :]
    b2 = corner_curr_bboxes[np.newaxis, :, :]

    x_left = np.maximum(b1[..., 0], b2[..., 0])
    y_top = np.maximum(b1[..., 1], b2[..., 1])
    x_right = np.minimum(b1[..., 2], b2[..., 2])
    y_bottom = np.minimum(b1[..., 3], b2[..., 3])

    intersection_area = np.maximum(0, x_right - x_left) * np.maximum(
        0, y_bottom - y_top
    )

    area_b1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area_b2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    union_area = area_b1 + area_b2 - intersection_area

    iou = np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(union_area),
        where=union_area >= 0,
    )

    iou[np.isnan(iou)] = 0

    return iou


def create_distance_matrix(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    if len(pos1) == 0 or len(pos2) == 0:
        return np.empty((len(pos1), len(pos2)), dtype=np.float64)
    return np.linalg.norm(pos1[:, np.newaxis, ...] - pos2[np.newaxis, ...], axis=-1)
