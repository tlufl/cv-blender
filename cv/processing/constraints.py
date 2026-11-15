from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cv.core.frame_data import FrameMetaData
from cv.core.math import convert_to_corner_coords

type StereoPairFilterFn = Callable[
    [FrameMetaData, FrameMetaData], tuple[FrameMetaData, FrameMetaData]
]
type MetaDataFilterFn = Callable[[FrameMetaData], FrameMetaData]


@dataclass
class FilterNode:
    filters: list[StereoPairFilterFn] = field(default_factory=list)

    def process(
        self, metadata1: FrameMetaData, metadata2: FrameMetaData
    ) -> tuple[FrameMetaData, FrameMetaData]:
        for group_filter in self.filters:
            metadata1, metadata2 = group_filter(metadata1, metadata2)

        return metadata1, metadata2


def get_edge_distance_filter(x_margin: float, y_margin: float) -> MetaDataFilterFn:
    def edge_distance_filter(metadata: FrameMetaData) -> FrameMetaData:
        bboxes = convert_to_corner_coords(metadata.detections["data"])
        bboxes += np.array([-x_margin, -y_margin, x_margin, y_margin])

        mask = np.any(((bboxes > 1) | (bboxes < 0)), axis=1)

        metadata.detections = metadata.detections[~mask]

        return metadata

    return edge_distance_filter


def get_object_distance_filter(x_margin: float, y_margin: float) -> MetaDataFilterFn:
    def get_intersect_mask(segments: np.ndarray) -> np.ndarray:
        starts = segments[:, 0]
        ends = segments[:, 1]

        start_le_end = starts[:, np.newaxis] <= ends[np.newaxis, :]

        return start_le_end & start_le_end.T

    def object_distance_filter(metadata: FrameMetaData) -> FrameMetaData:
        bboxes = convert_to_corner_coords(metadata.detections["data"])
        bboxes += np.array([-x_margin, -y_margin, x_margin, y_margin])

        x = bboxes[:, [0, 2]]
        y = bboxes[:, [1, 3]]

        mask = get_intersect_mask(x) & get_intersect_mask(y)
        mask[np.eye(len(x), dtype=np.bool)] = False
        filter = np.any(mask, axis=1)

        metadata.detections = metadata.detections[~filter]

        return metadata

    return object_distance_filter
