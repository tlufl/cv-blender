from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameMetaData:
    frame_id: int
    bboxes: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0, 4), dtype=np.float32)
    )
    object_ids: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )


class FrameContext:
    def __init__(self, image: np.ndarray, frame_id: int):
        self.original: np.ndarray = image
        self.original.flags.writeable = False
        self.processed: np.ndarray = image.copy()

        self.metadata: FrameMetaData = FrameMetaData(frame_id)
