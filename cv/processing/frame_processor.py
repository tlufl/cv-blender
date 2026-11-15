from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from cv.core.frame_data import FrameContext, FrameMetaData
from cv.core.logging import log

type ProcessFrameFn = Callable[[FrameContext], FrameMetaData]

from cv.core.frame_data import DetectionBBox


@dataclass
class ProcessFramesNode:
    process_frame: ProcessFrameFn

    def process(
        self, context1: FrameContext, context2: FrameContext
    ) -> tuple[FrameMetaData, FrameMetaData]:
        metadata1 = self.process_frame(context1)
        metadata2 = self.process_frame(context2)
        return metadata1, metadata2


type ColorRange = tuple[tuple[int, int, int], tuple[int, int, int]]


def detect_by_color(color_ranges: list[ColorRange]) -> ProcessFrameFn:
    if not color_ranges:
        raise RuntimeError("Color ranges is required")

    def create_mask(hsv: cv2.typing.MatLike) -> cv2.typing.MatLike:
        result_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_range in color_ranges:
            mask = cv2.inRange(hsv, *color_range)
            result_mask = cv2.bitwise_or(result_mask, mask)
        return result_mask

    def construct_metadata(
        context: FrameContext, contours: Sequence[cv2.typing.MatLike]
    ) -> FrameMetaData:
        if not contours:
            return context.metadata

        detections = []
        frame_h, frame_w = context.original.shape[:2]
        normalization_factor = np.array(
            [frame_w, frame_h, frame_w, frame_h], dtype=np.float32
        )

        for id, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            new_bbox = [x + w / 2, y + h / 2, w, h]
            detections.append((id, new_bbox))

        detections = np.array(detections, dtype=DetectionBBox)
        detections["data"] /= normalization_factor

        context.metadata.detections = np.concat(
            (context.metadata.detections, detections)
        )
        return context.metadata

    @log(title="color detection")
    def process_frame(context: FrameContext) -> FrameMetaData:
        hsv = cv2.cvtColor(context.processed, cv2.COLOR_BGR2HSV)
        mask = create_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return construct_metadata(context, contours)

    return process_frame


def detect_by_yolo(
    model: YOLO, allowed_classes: list[int] | None = None
) -> ProcessFrameFn:
    @log(title="yolo detection")
    def process_frame(context: FrameContext) -> FrameMetaData:
        results = cast(
            list[Results],
            model.predict(context.processed, classes=allowed_classes, verbose=False),
        )
        if not results:
            return context.metadata

        result: Results = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return context.metadata

        bboxes_array: np.ndarray = cast(np.ndarray, result.boxes.xywhn)
        ids_array = np.arange(len(bboxes_array), dtype=np.int64)

        detections = np.empty(len(bboxes_array), dtype=DetectionBBox)
        detections["data"] = bboxes_array
        detections["id"] = ids_array

        context.metadata.detections = np.concat(
            (context.metadata.detections, detections)
        )

        return context.metadata

    return process_frame
