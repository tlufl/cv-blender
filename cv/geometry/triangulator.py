import numpy as np
from dataclasses import dataclass

from cv.core.frame_data import FrameMetaData
from cv.core.world_data import WorldContext


@dataclass
class TriangulateNode:
    f_mm: float
    w_mm: float
    baseline: float

    def process(
        self, metadata1: FrameMetaData, metadata2: FrameMetaData
    ) -> WorldContext:
        world_ctx = WorldContext(
            frame_id=metadata1.frame_id, frame_res=metadata1.frame_res
        )
        bboxes_l = metadata1.bboxes
        bboxes_r = metadata2.bboxes

        if not bboxes_l.size or not bboxes_r.size:
            return world_ctx

        h_px, w_px = metadata1.frame_res[:2]
        f_px = self.f_mm * (w_px / self.w_mm)
        c_x = w_px / 2
        c_y = h_px / 2

        center_x_l = bboxes_l[:, 0] * w_px
        center_y_l = bboxes_l[:, 1] * h_px

        center_x_r = bboxes_r[:, 0] * w_px

        d = center_x_l - center_x_r
        d = np.where(d <= 0, 1e-3, d)

        z_coords = self.baseline * f_px / d
        x_coords = (center_x_l - c_x) * z_coords / f_px
        y_coords = (center_y_l - c_y) * z_coords / f_px

        world_ctx.positions = np.column_stack((x_coords, y_coords, z_coords))

        world_ctx.object_ids = metadata1.object_ids.copy()

        return world_ctx
