import cv2
import numpy as np

from cv.core.frame_data import FrameContext


def process_frame(context: FrameContext) -> FrameContext:
    hsv = cv2.cvtColor(context.processed, cv2.COLOR_BGR2HSV)
    lower_bright = np.array([0, 100, 100])
    upper_bright = np.array([180, 255, 255])
    range1 = (np.array([0, 120, 70]), np.array([10, 255, 255]))
    range2 = (np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask1 = cv2.inRange(hsv, *range1)
    mask2 = cv2.inRange(hsv, *range2)
    mask3 = cv2.inRange(hsv, lower_bright, upper_bright)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for id, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        new_bbox = np.array([[x + w / 2, y + h / 2, w, h]], dtype=np.float32)

        frame_h, frame_w = context.original.shape[:2]
        new_bbox /= np.array([frame_w, frame_h, frame_w, frame_h], dtype=np.float32)

        context.metadata.bboxes = np.vstack((context.metadata.bboxes, new_bbox))
        context.metadata.object_ids = np.hstack((context.metadata.object_ids, [id]))

    return context
