import cv2

from cv.core.frame_data import FrameContext


class ImageSourceNode:
    def __init__(self, left_image_path: str, right_image_path: str):
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        self.close = False

    def process(self) -> tuple[FrameContext, FrameContext] | None:
        frame_l = cv2.imread(self.left_image_path, cv2.IMREAD_COLOR_BGR)
        frame_r = cv2.imread(self.right_image_path, cv2.IMREAD_COLOR_BGR)

        if self.close or frame_l is None or frame_r is None:
            return None

        self.close = True

        return FrameContext(frame_l, 0), FrameContext(frame_r, 0)
