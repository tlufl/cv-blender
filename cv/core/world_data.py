from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorldContext:
    frame_id: int
    positions: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0, 3), dtype=np.float32)
    )
    object_ids: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )
