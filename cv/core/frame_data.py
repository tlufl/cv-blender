from dataclasses import dataclass, field

import numpy as np

DetectionBBox = np.dtype([("id", "i4"), ("data", "f8", (4,))])


@dataclass
class FrameMetaData:
    frame_id: int
    frame_res: tuple[int, int]
    detections: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=(0,), dtype=DetectionBBox)
    )


class FrameContext:
    def __init__(self, image: np.ndarray, frame_id: int):
        self.original: np.ndarray = image
        self.original.flags.writeable = False
        self.processed: np.ndarray = image.copy()

        self.metadata: FrameMetaData = FrameMetaData(frame_id, image.shape[:2])
