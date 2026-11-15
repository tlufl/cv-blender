from typing import Final

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from cv.core.frame_data import FrameContext
from cv.processing.frame_processor import detect_by_yolo
from tests.test_data.mock_data_generator import crate_test_data, create_cost_matrix

RESOLUTION: Final[tuple[int, int]] = (768, 576)

model = YOLO()
process_frame = detect_by_yolo(model)


def test_basic():
    expected = np.array([[0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
    img = crate_test_data(expected, RESOLUTION)
    context = FrameContext(img, 0)

    res = process_frame(context)

    assert len(res.detections) > 0

    cost_matrix = create_cost_matrix(expected, res.detections["data"])
    exp_ids, res_ids = linear_sum_assignment(cost_matrix)

    expected = expected[exp_ids]
    res.detections = res.detections[res_ids]

    np.testing.assert_allclose(expected, res.detections["data"], atol=0.05)


def test_two_planes():
    expected = np.array(
        [[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.3, 0.4]],
        dtype=np.float32,
    )
    img = crate_test_data(expected, RESOLUTION)
    context = FrameContext(img, 0)

    res = process_frame(context)

    assert len(res.detections) > 0

    cost_matrix = create_cost_matrix(expected, res.detections["data"])
    exp_ids, res_ids = linear_sum_assignment(cost_matrix)

    expected = expected[exp_ids]
    res.detections = res.detections[res_ids]

    np.testing.assert_allclose(expected, res.detections["data"], atol=0.05)


def test_no_planes():
    expected = np.empty(shape=(0, 4), dtype=np.float32)
    img = crate_test_data(expected, RESOLUTION)
    context = FrameContext(img, 0)

    res = process_frame(context)

    assert len(res.detections) == 0


def test_plane_at_edge():
    expected = np.array([[0.1, 0.1, 0.15, 0.15]], dtype=np.float32)
    img = crate_test_data(expected, RESOLUTION)
    context = FrameContext(img, 0)

    res = process_frame(context)

    assert len(res.detections) == len(expected)

    cost_matrix = create_cost_matrix(expected, res.detections["data"])
    exp_ids, res_ids = linear_sum_assignment(cost_matrix)

    np.testing.assert_allclose(
        expected[exp_ids], res.detections[res_ids]["data"], atol=0.06
    )


def test_large_plane():
    expected = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    img = crate_test_data(expected, RESOLUTION)
    context = FrameContext(img, 0)

    res = process_frame(context)

    assert len(res.detections) == len(expected)

    cost_matrix = create_cost_matrix(expected, res.detections["data"])
    exp_ids, res_ids = linear_sum_assignment(cost_matrix)

    np.testing.assert_allclose(
        expected[exp_ids], res.detections[res_ids]["data"], atol=0.05
    )


def test_multiple_identical_planes_row():
    expected = np.array(
        [[0.2, 0.4, 0.15, 0.15], [0.5, 0.4, 0.15, 0.15], [0.8, 0.4, 0.15, 0.15]],
        dtype=np.float32,
    )
    img = crate_test_data(expected, RESOLUTION)
    context = FrameContext(img, 0)

    res = process_frame(context)

    assert len(res.detections) > 0

    cost_matrix = create_cost_matrix(expected, res.detections["data"])
    exp_ids, res_ids = linear_sum_assignment(cost_matrix)

    np.testing.assert_allclose(
        expected[exp_ids], res.detections[res_ids]["data"], atol=0.06
    )
