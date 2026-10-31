from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from cv.core.frame_data import FrameMetaData
from cv.core.math import INF, rectangular_assignment
from cv.core.logging import logg


def create_distance_matrix(
    left_centers: np.ndarray,
    right_centers: np.ndarray,
    threshold: float,
) -> np.ndarray:
    matrix = np.sqrt(
        np.sum((left_centers[:, np.newaxis, :] - right_centers) ** 2, axis=-1)
    )

    y_dist = np.abs(left_centers[:, np.newaxis, 1] - right_centers[:, 1])
    y_filter = y_dist < threshold

    x_filter = left_centers[:, np.newaxis, 0] > right_centers[:, 0]

    valid_mask = x_filter & y_filter

    matrix[~valid_mask] = INF

    return matrix


class CompareMethod(Enum):
    GLOBAL = 0
    GREEDY = 1


@dataclass
class CompareResult:
    matched_left: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))
    matched_right: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))
    unmatched_left: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))
    unmatched_right: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))


class StereoPairComparator:
    def __init__(
        self,
        max_distance: float,
        epipolar_threshold: float,
        method: CompareMethod = CompareMethod.GLOBAL,
    ) -> None:
        self.method: CompareMethod = method
        self.max_distance = max_distance
        self.epipolar_threshold = epipolar_threshold

    @logg()
    def _global_compare(self, matrix: np.ndarray) -> CompareResult:
        left_idx, right_idx, unmatched_left, unmatched_right = rectangular_assignment(
            matrix, self.max_distance
        )
        return CompareResult(left_idx, right_idx, unmatched_left, unmatched_right)

    @logg()
    def _greedy_compare(self, matrix: np.ndarray) -> CompareResult:
        flat_indices = np.argsort(matrix.flatten())
        left_indices, right_indices = np.unravel_index(flat_indices, matrix.shape)
        used_right = set()
        used_left = set()
        left_idx = []
        right_idx = []

        for left_ind, right_ind in zip(left_indices, right_indices):
            if matrix[left_ind, right_ind] >= self.max_distance:
                break
            if right_ind not in used_right and left_ind not in used_left:
                used_left.add(left_ind)
                used_right.add(right_ind)
                left_idx.append(left_ind)
                right_idx.append(right_ind)

        left_idx = np.array(left_idx)
        right_idx = np.array(right_idx)

        all_left = np.arange(matrix.shape[0])
        all_right = np.arange(matrix.shape[1])
        unmatched_left = np.setdiff1d(all_left, left_idx)
        unmatched_right = np.setdiff1d(all_right, right_idx)

        return CompareResult(left_idx, right_idx, unmatched_left, unmatched_right)

    def _compare(self, matrix: np.ndarray) -> CompareResult:
        match self.method:
            case CompareMethod.GLOBAL:
                return self._global_compare(matrix)
            case CompareMethod.GREEDY:
                return self._greedy_compare(matrix)

    def process(
        self, metadata1: FrameMetaData, metadata2: FrameMetaData
    ) -> tuple[FrameMetaData, FrameMetaData, CompareResult]:
        left_centers = metadata1.bboxes[:, :2]
        right_centers = metadata2.bboxes[:, :2]

        matrix = create_distance_matrix(
            left_centers, right_centers, self.epipolar_threshold
        )
        return (
            metadata1,
            metadata2,
            self._compare(matrix),
        )
