from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from cv.core.frame_data import FrameMetaData
from cv.core.math import rectangular_assignment
from cv.processing.stereopairs_comporator import CompareResult


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

    iou = 1 - iou
    iou[np.isnan(iou)] = 1

    return iou


class TrackMethod(Enum):
    GLOBAL = 0
    GREEDY = 1


@dataclass
class Track:
    """format:\n
     left_bboxes:| matched indices | unmatched indices |        -          |\n
    right_bboxes:| matched indices |         -         | unmatched_indices |
    """

    left_bboxes: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0, 4), dtype=np.float32)
    )
    right_bboxes: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0, 4), dtype=np.float32)
    )
    num_matches: int = 0
    num_unmatched_left: int = 0
    num_unmatched_right: int = 0
    total_detections: int = 0

    @classmethod
    def create_from_bboxes(
        cls,
        left_bboxes: np.ndarray,
        right_bboxes: np.ndarray,
        cmp_result: CompareResult,
    ) -> "Track":
        num_matches: int = cmp_result.matched_left.size
        num_unmatched_left: int = cmp_result.unmatched_left.size
        num_unmatched_right: int = cmp_result.unmatched_right.size
        total_detections: int = num_matches + num_unmatched_left + num_unmatched_right

        formated_left_bboxes: np.ndarray = np.full(
            shape=(total_detections, 4), fill_value=np.nan, dtype=np.float32
        )
        formated_right_bboxes: np.ndarray = np.full(
            shape=(total_detections, 4), fill_value=np.nan, dtype=np.float32
        )

        if num_matches:
            formated_left_bboxes[:num_matches] = left_bboxes[cmp_result.matched_left]
            formated_right_bboxes[:num_matches] = right_bboxes[cmp_result.matched_right]

        if num_unmatched_left:
            start_idx = num_matches
            end_idx = start_idx + num_unmatched_left
            formated_left_bboxes[start_idx:end_idx] = left_bboxes[
                cmp_result.unmatched_left
            ]

        if num_unmatched_right:
            start_idx = num_matches + num_unmatched_left
            end_idx = start_idx + num_unmatched_right
            formated_right_bboxes[start_idx:end_idx] = right_bboxes[
                cmp_result.unmatched_right
            ]

        return Track(
            formated_left_bboxes,
            formated_right_bboxes,
            num_matches,
            num_unmatched_left,
            num_unmatched_right,
            total_detections,
        )

    @property
    def left_mask(self) -> np.ndarray:
        return ~np.isnan(self.left_bboxes).any(axis=1)

    @property
    def right_mask(self) -> np.ndarray:
        return ~np.isnan(self.right_bboxes).any(axis=1)


@dataclass
class TracksHistory:
    track: Track = field(default_factory=lambda: Track())
    track_ids: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )


@dataclass
class TrackResult:
    matched_curr_idx: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )
    matched_track_idx: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )
    new_idx: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )
    lost_idx: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )


class TimeTrackingNode:
    def __init__(
        self, min_iou: float, method: TrackMethod = TrackMethod.GLOBAL
    ) -> None:
        self.method = method
        self.min_iou = min_iou

        self._next_id: int = 0
        self._tracks_history: TracksHistory = TracksHistory()

    def _create_iou_matrix(self, track: Track) -> np.ndarray:
        iou_left = create_iou_matrix(
            self._tracks_history.track.left_bboxes, track.left_bboxes
        )
        iou_right = create_iou_matrix(
            self._tracks_history.track.right_bboxes, track.right_bboxes
        )

        iou_matrix = (iou_left + iou_right) / 2
        return iou_matrix

    def _greedy_tracking(self, iou_matrix: np.ndarray) -> TrackResult:
        flat_indices = np.argsort(iou_matrix.flatten())
        track_indices, curr_indices = np.unravel_index(flat_indices, iou_matrix.shape)

        used_track = set()
        used_curr = set()
        track_idx = []
        curr_idx = []

        for track_ind, curr_ind in zip(track_indices, curr_indices):
            if iou_matrix[track_ind, curr_ind] > (1 - self.min_iou):
                break
            if track_ind not in used_track and curr_ind not in used_curr:
                used_track.add(track_ind)
                used_curr.add(curr_ind)
                track_idx.append(track_ind)
                curr_idx.append(curr_ind)

        matched_track_idx = np.array(track_idx, dtype=np.int32)
        matched_curr_idx = np.array(curr_idx, dtype=np.int32)

        all_curr_idx = np.arange(iou_matrix.shape[1])
        all_track_idx = np.arange(iou_matrix.shape[0])

        lost_idx = np.setdiff1d(all_track_idx, matched_track_idx)
        new_idx = np.setdiff1d(all_curr_idx, matched_curr_idx)

        return TrackResult(matched_curr_idx, matched_track_idx, new_idx, lost_idx)

    def _assign_ids(
        self,
        metadata1: FrameMetaData,
        metadata2: FrameMetaData,
        track: Track,
        track_result: TrackResult,
    ) -> tuple[FrameMetaData, FrameMetaData, np.ndarray]:
        assign_idx = np.zeros(shape=track.total_detections, dtype=np.int32)
        assign_idx[track_result.matched_curr_idx] = self._tracks_history.track_ids[
            track_result.matched_track_idx
        ]

        if track_result.new_idx.size:
            new_ids = np.arange(
                self._next_id, self._next_id + track_result.new_idx.size
            )
            assign_idx[track_result.new_idx] = new_ids
            self._next_id += track_result.new_idx.size

        metadata1.bboxes = track.left_bboxes[track.left_mask]
        metadata1.object_ids = assign_idx[track.left_mask]
        metadata2.bboxes = track.right_bboxes[track.right_mask]
        metadata2.object_ids = assign_idx[track.right_mask]

        return metadata1, metadata2, assign_idx

    def _update_history(self, track: Track, track_ids: np.ndarray) -> None:
        self._tracks_history.track = track
        self._tracks_history.track_ids = track_ids

    def _global_tracking(self, iou_matrix: np.ndarray) -> TrackResult:
        tid, cid, lost, new = rectangular_assignment(iou_matrix)
        return TrackResult(cid, tid, new, lost)

    def _track(self, iou_matrix: np.ndarray) -> TrackResult:
        match self.method:
            case TrackMethod.GLOBAL:
                return self._global_tracking(iou_matrix)
            case TrackMethod.GREEDY:
                return self._greedy_tracking(iou_matrix)

    def process(
        self,
        metadata1: FrameMetaData,
        metadata2: FrameMetaData,
        cmp_result: CompareResult,
    ) -> tuple[FrameMetaData, FrameMetaData]:
        track = Track.create_from_bboxes(metadata1.bboxes, metadata2.bboxes, cmp_result)

        iou_matrix = self._create_iou_matrix(track)
        track_result = self._track(iou_matrix)
        metadata1, metadata2, track_ids = self._assign_ids(
            metadata1, metadata2, track, track_result
        )
        self._update_history(track, track_ids)

        return metadata1, metadata2
