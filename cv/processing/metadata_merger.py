from cv.core.frame_data import FrameContext

import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass


def dont_merge_metadata(
    context1: FrameContext, context2: FrameContext
) -> tuple[FrameContext, FrameContext]:
    return context1, context2


def rectangular_assignment(
    matrix: np.ndarray, max_threshold: float = 1e4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_indices, right_indices = linear_sum_assignment(matrix)

    all_left = np.arange(matrix.shape[0])
    all_right = np.arange(matrix.shape[1])

    valid_matches = matrix[left_indices, right_indices] < max_threshold
    left_indices = left_indices[valid_matches]
    right_indices = right_indices[valid_matches]

    unmatched_left = np.setdiff1d(all_left, left_indices)
    unmatched_right = np.setdiff1d(all_right, right_indices)

    return left_indices, right_indices, unmatched_left, unmatched_right


def create_distance_matrix(
    left_centers: np.ndarray,
    right_centers: np.ndarray,
    threshold: float = 1e-2,
) -> np.ndarray:
    matrix = np.sqrt(
        np.sum((left_centers[:, np.newaxis, :] - right_centers) ** 2, axis=-1)
    )

    y_dist = np.abs(left_centers[:, np.newaxis, 1] - right_centers[:, 1])
    y_filter = y_dist < threshold

    x_filter = left_centers[:, np.newaxis, 0] > right_centers[:, 0]

    valid_mask = x_filter & y_filter

    matrix[~valid_mask] = 1e5

    return matrix


def convert_to_corner_coords(bboxes: np.ndarray) -> np.ndarray:
    new_bboxes = np.empty(shape=bboxes.shape)

    new_bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2
    new_bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] / 2

    return new_bboxes


def create_iou_matrix(prev_bboxes: np.ndarray, curr_bboxes: np.ndarray) -> np.ndarray:
    corner_prev_bboxes = convert_to_corner_coords(prev_bboxes)
    corner_curr_bboxes = convert_to_corner_coords(curr_bboxes)

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
        where=union_area > 0,
    )

    return 1 - iou


@dataclass
class CompareResult:
    matched_left: np.ndarray
    matched_right: np.ndarray
    unmatched_left: np.ndarray
    unmatched_right: np.ndarray


class MetaDataMerger:
    def __init__(self):
        self.max_missed_frames = 10
        self._next_id: int = 0
        self._left_tracks: np.ndarray = np.array([])
        self._right_tracks: np.ndarray = np.array([])
        self._track_ids: np.ndarray = np.array([])
        self._track_missed = np.empty([])

    def merge_metadata(
        self, context1: FrameContext, context2: FrameContext
    ) -> tuple[FrameContext, FrameContext]:
        cmp_result = self._compare_steriopairs(context1, context2)
        return self._track_steriopairs(context1, context2, cmp_result)

    def _compare_steriopairs(
        self, context1: FrameContext, context2: FrameContext
    ) -> CompareResult:
        left_centers = context1.metadata.bboxes[:, :2]
        right_centers = context2.metadata.bboxes[:, :2]

        matrix = create_distance_matrix(left_centers, right_centers, 0.1)
        left_idx, right_idx, unmatched_left, unmatched_right = rectangular_assignment(
            matrix, 0.8
        )
        return CompareResult(left_idx, right_idx, unmatched_left, unmatched_right)

    def _track_steriopairs(
        self, context1: FrameContext, context2: FrameContext, cmp_result: CompareResult
    ) -> tuple[FrameContext, FrameContext]:
        left_bboxes = context1.metadata.bboxes[cmp_result.matched_left]
        right_bboxes = context2.metadata.bboxes[cmp_result.matched_right]

        if not self._left_tracks.size:
            self._left_tracks = left_bboxes
            self._track_ids = context1.metadata.object_ids

        iou = create_iou_matrix(self._left_tracks, left_bboxes)

        tid, cid, *_ = rectangular_assignment(iou)

        context1.metadata.bboxes = left_bboxes[cid]
        context2.metadata.bboxes = right_bboxes[cid]
        context1.metadata.object_ids = context2.metadata.object_ids = self._track_ids[
            tid
        ]

        self._left_tracks = context1.metadata.bboxes
        self._right_tracks = context2.metadata.bboxes
        self._track_ids = context1.metadata.object_ids

        return context1, context2
