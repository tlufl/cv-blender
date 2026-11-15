import cv2
import numpy as np

from cv.core.frame_data import FrameContext
from cv.processing.frame_processor import ColorRange, detect_by_color

color_range: ColorRange = ((0, 0, 30), (255, 255, 255))
process_frame = detect_by_color([color_range])


def test_basic_mask_detection():
    img = np.zeros(shape=(1000, 1000, 3), dtype=np.uint8)

    bbox = np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float32)
    cx, cy, w, h = np.array(bbox * np.array([1000, 1000, 1000, 1000]), dtype=int)

    xl = cx - w // 2
    yl = cy - h // 2
    xr = cx + w // 2
    yr = cy + h // 2

    img[yl:yr, xl:xr] = np.array([255, 255, 255], dtype=np.uint8)

    metadata = process_frame(FrameContext(image=img, frame_id=0))
    np.testing.assert_allclose(bbox, metadata.detections["data"][0])


def test_no_objects_detected():
    img = np.zeros(shape=(1000, 1000, 3), dtype=np.uint8)

    metadata = process_frame(FrameContext(image=img, frame_id=0))

    assert len(metadata.detections["data"]) == 0


def test_multiple_objects_detection():
    img = np.zeros(shape=(1000, 1000, 3), dtype=np.uint8)

    bbox1 = np.array([0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    bbox2 = np.array([0.7, 0.8, 0.15, 0.15], dtype=np.float32)

    cx1, cy1, w1, h1 = np.array(bbox1 * 1000, dtype=int)
    img[cy1 - h1 // 2 : cy1 + h1 // 2, cx1 - w1 // 2 : cx1 + w1 // 2] = [255, 255, 255]

    cx2, cy2, w2, h2 = np.array(bbox2 * 1000, dtype=int)
    img[cy2 - h2 // 2 : cy2 + h2 // 2, cx2 - w2 // 2 : cx2 + w2 // 2] = [255, 255, 255]

    metadata = process_frame(FrameContext(image=img, frame_id=1))

    assert len(metadata.detections["data"]) == 2

    detected_bboxes = sorted(metadata.detections["data"], key=lambda x: x[0])

    np.testing.assert_allclose(bbox1, detected_bboxes[0], atol=1e-2)
    np.testing.assert_allclose(bbox2, detected_bboxes[1], atol=1e-2)


def test_object_on_the_image_edge():
    img = np.zeros(shape=(1000, 1000, 3), dtype=np.uint8)

    bbox = np.array([0.95, 0.95, 0.1, 0.1], dtype=np.float32)

    cx, cy, w, h = np.array(bbox * 1000, dtype=int)
    xl, yl = cx - w // 2, cy - h // 2
    xr, yr = cx + w // 2, cy + h // 2

    img[yl:yr, xl:xr] = [255, 255, 255]

    metadata = process_frame(FrameContext(image=img, frame_id=2))

    assert len(metadata.detections["data"]) > 0
    np.testing.assert_allclose(bbox, metadata.detections["data"][0], atol=2e-2)


def test_blurred_mask_detection():
    img = np.zeros(shape=(1000, 1000, 3), dtype=np.uint8)
    bbox = np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32)

    cx, cy, w, h = np.array(bbox * 1000, dtype=int)
    img[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2] = [255, 255, 255]

    img = cv2.GaussianBlur(img, (15, 15), 0)

    metadata = process_frame(FrameContext(image=img, frame_id=3))

    assert len(metadata.detections["data"]) == 1
    np.testing.assert_allclose(bbox, metadata.detections["data"][0], atol=1e-2)
