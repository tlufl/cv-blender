from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, Self

import numpy as np

from cv.core.logging import log
from cv.core.math import (
    INF,
    create_distance_matrix,
    rectangular_assignment,
)


class HasDetections(Protocol):
    detections: np.ndarray


def make_track_dtype(ndim) -> np.dtype:
    return np.dtype(
        [
            ("age", "i4"),
            ("id", "i4"),
            ("data", "f8", (ndim,)),
            ("last_data", "f8", (ndim,)),
            ("velocity", "f8", (ndim,)),
        ]
    )


Track2D = make_track_dtype(4)
Track3D = make_track_dtype(3)

# ----- 3D Tracking -----


type PredictFn = Callable[[np.ndarray], np.ndarray]


def linear_predict(tracks: np.ndarray) -> np.ndarray:
    tracks["data"] += tracks["velocity"]
    return tracks


@dataclass
class MatchResult:
    track_idx: np.ndarray
    curr_idx: np.ndarray
    unmatched_curr_idx: np.ndarray

    @classmethod
    def empty(cls, num_curr: int = 0) -> Self:
        return cls(
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.arange(num_curr, dtype=int),
        )


@dataclass
class TrackHistory:
    track_dtype: np.dtype
    max_age: int = 0
    _predict: PredictFn | None = None
    _history: np.ndarray = field(init=False)
    _next_id: int = 1

    def __post_init__(self) -> None:
        self._history = np.empty((0,), dtype=self.track_dtype)

    def get_new_ids(self, count: int) -> np.ndarray:
        ids = np.arange(self._next_id, self._next_id + count, dtype="i4")
        self._next_id += count
        return ids

    def _update_history(
        self, update_tracks: np.ndarray, match_result: MatchResult
    ) -> None:
        self._history["age"] += 1

        if len(match_result.curr_idx) > 0:
            new_data = update_tracks["data"][match_result.curr_idx]
            old_data = self._history["last_data"][match_result.track_idx]

            self._history["velocity"][match_result.track_idx] = (
                new_data - old_data
            ) / self._history["age"][match_result.track_idx][:, np.newaxis]
            self._history["data"][match_result.track_idx] = new_data
            self._history["last_data"][match_result.track_idx] = new_data
            self._history["age"][match_result.track_idx] = 0

        new_tracks = update_tracks[match_result.unmatched_curr_idx]
        if len(new_tracks) > 0:
            self._history = np.concatenate((self._history, new_tracks))

    def _delete_old_tracks(self) -> None:
        self._history = self._history[self._history["age"] <= self.max_age]

    def predict(self) -> None:
        if self._predict is not None and len(self._history) > 0:
            self._history = self._predict(self._history)

    def update(self, context: HasDetections, match_result: MatchResult) -> None:
        num_update_tracks = len(context.detections)
        update_tracks = np.zeros(num_update_tracks, dtype=self.track_dtype)

        if num_update_tracks > 0:
            update_tracks[["id", "data"]] = context.detections[["id", "data"]]
            update_tracks["last_data"] = context.detections["data"]

        self._update_history(update_tracks, match_result)
        self._delete_old_tracks()

    @property
    def tracks(self) -> np.ndarray:
        return self._history

    @property
    def previous_positions(self) -> np.ndarray:
        return self._history["data"]


type MatchFn = Callable[[np.ndarray, float], MatchResult]


def global_distance_matching(matrix: np.ndarray, max_val: float) -> MatchResult:
    num_curr = matrix.shape[1]
    if matrix.size == 0:
        return MatchResult.empty(num_curr)

    cost_matrix = np.where(matrix > max_val, INF, matrix)
    matched_track_ind, matched_curr_ind, *_ = rectangular_assignment(cost_matrix)

    valid_mask = cost_matrix[matched_track_ind, matched_curr_ind] < INF
    matched_track_ind = matched_track_ind[valid_mask]
    matched_curr_ind = matched_curr_ind[valid_mask]
    unmatched_curr_ind = np.setdiff1d(np.arange(num_curr), matched_curr_ind)

    return MatchResult(matched_track_ind, matched_curr_ind, unmatched_curr_ind)


class WorldTrackingNode[ContextT: HasDetections]:
    def __init__(
        self,
        track_dtype: np.dtype,
        match_fn: MatchFn,
        max_age: int = 0,
        predict_fn: PredictFn | None = None,
        max_distance: float = 1.0,
    ) -> None:
        self._match = match_fn
        self._history = TrackHistory(track_dtype, max_age, predict_fn)
        self.max_distance = max_distance

    def _assign_ids(
        self,
        match_result: MatchResult,
        context: ContextT,
    ) -> ContextT:
        num_objects = len(context.detections)
        assigned_ids = np.full(num_objects, -1, dtype="i4")

        if len(match_result.curr_idx) > 0:
            assigned_ids[match_result.curr_idx] = self._history.tracks["id"][
                match_result.track_idx
            ]

        num_new = len(match_result.unmatched_curr_idx)
        if num_new > 0:
            assigned_ids[match_result.unmatched_curr_idx] = self._history.get_new_ids(
                num_new
            )

        context.detections["id"] = assigned_ids
        return context

    @log(title="3D tracking")
    def process(self, context: ContextT) -> ContextT:
        self._history.predict()

        distances = create_distance_matrix(
            self._history.previous_positions, context.detections["data"]
        )
        match_result = self._match(distances, self.max_distance)
        context = self._assign_ids(match_result, context)

        self._history.update(context, match_result)
        return context


from cv.core.frame_data import FrameMetaData
from cv.processing.stereopairs_comporator import CompareResult


class TmpNode:
    def process(
        self, metadata1: FrameMetaData, metadata2: FrameMetaData, cmp_res: CompareResult
    ) -> tuple[FrameMetaData, FrameMetaData]:
        metadata1.detections = metadata1.detections[cmp_res.matched_left]
        metadata2.detections = metadata2.detections[cmp_res.matched_right]
        return metadata1, metadata2
