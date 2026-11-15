from dataclasses import dataclass, field

import numpy as np

Detection3D = np.dtype([("id", "i4"), ("data", "f8", (3,))])


@dataclass
class WorldContext:
    frame_id: int
    frame_res: tuple[int, int]
    detections: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=Detection3D)
    )
