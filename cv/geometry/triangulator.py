import numpy as np

from cv.core.engine import TriangulateFn
from cv.core.frame_data import FrameContext
from cv.core.world_data import WorldContext


def create_triangulator(f_mm: float, w_mm: float, baseline: float) -> TriangulateFn:
    def triangulator(context1: FrameContext, context2: FrameContext) -> WorldContext:
        world_ctx = WorldContext(frame_id=context1.metadata.frame_id)
        bboxes_l = context1.metadata.bboxes
        bboxes_r = context2.metadata.bboxes

        if not bboxes_l.size or not bboxes_r.size:
            return world_ctx

        h_px, w_px = context1.original.shape[:2]
        f_px = f_mm * (w_px / w_mm)
        c_x = w_px / 2
        c_y = h_px / 2

        center_x_l = bboxes_l[:, 0] * w_px
        center_y_l = bboxes_l[:, 1] * h_px

        center_x_r = bboxes_r[:, 0] * w_px

        d = center_x_l - center_x_r
        d = np.where(d <= 0, 1e-3, d)

        z_coords = baseline * f_px / d
        x_coords = (center_x_l - c_x) * z_coords / f_px
        y_coords = (center_y_l - c_y) * z_coords / f_px

        world_ctx.positions = np.column_stack((x_coords, y_coords, z_coords))

        world_ctx.object_ids = context1.metadata.object_ids.copy()

        return world_ctx

    return triangulator
