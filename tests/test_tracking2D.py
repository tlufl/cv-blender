import numpy as np
import pytest

from cv.core.frame_data import DetectionBBox, FrameMetaData
from cv.processing.time_tracker import (
    Track2D,
    TrackingNode,
    global_matching,
    linear_predict,
    create_iou_metric_matrix,
    create_iou_matrix,
)

# ----- Test create_iou_matrix and create_iou_metric_matrix -----


def test_iou_identical_boxes():
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype="f8")

    iou = create_iou_matrix(boxes, boxes)
    metric = create_iou_metric_matrix(boxes, boxes)

    np.testing.assert_allclose(iou, [[1.0]], atol=1e-12)
    np.testing.assert_allclose(metric, [[0.0]], atol=1e-12)


def test_iou_disjoint_boxes():
    boxes1 = np.array([[0.1, 0.1, 0.1, 0.1]], dtype="f8")
    boxes2 = np.array([[0.8, 0.8, 0.1, 0.1]], dtype="f8")

    iou = create_iou_matrix(boxes1, boxes2)
    metric = create_iou_metric_matrix(boxes1, boxes2)

    np.testing.assert_allclose(iou, [[0.0]], atol=1e-12)
    np.testing.assert_allclose(metric, [[1.0]], atol=1e-12)


def test_iou_partial_overlap():
    box1 = np.array([[0.2, 0.2, 0.2, 0.2]], dtype="f8")
    box2 = np.array([[0.3, 0.2, 0.2, 0.2]], dtype="f8")

    iou = create_iou_matrix(box1, box2)
    metric = create_iou_metric_matrix(box1, box2)

    np.testing.assert_allclose(iou, [[1.0 / 3.0]], atol=1e-12)
    np.testing.assert_allclose(metric, [[2.0 / 3.0]], atol=1e-12)


def test_iou_box_inside_another():
    outer = np.array([[0.5, 0.5, 0.4, 0.4]], dtype="f8")
    inner = np.array([[0.5, 0.5, 0.2, 0.2]], dtype="f8")

    iou = create_iou_matrix(outer, inner)

    np.testing.assert_allclose(iou, [[0.25]], atol=1e-12)


def test_iou_touching_edges():
    box1 = np.array([[0.1, 0.1, 0.2, 0.2]], dtype="f8")
    box2 = np.array([[0.3, 0.1, 0.2, 0.2]], dtype="f8")

    iou = create_iou_matrix(box1, box2)

    np.testing.assert_allclose(iou, [[0.0]], atol=1e-12)


def test_iou_matrix_shape_and_empty_inputs():
    boxes_n = np.array([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]], dtype="f8")
    boxes_m = np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]],
        dtype="f8",
    )
    empty = np.empty((0, 4), dtype="f8")

    assert create_iou_matrix(boxes_n, boxes_m).shape == (2, 3)
    assert create_iou_matrix(empty, boxes_m).shape == (0, 3)
    assert create_iou_matrix(boxes_n, empty).shape == (2, 0)
    assert create_iou_metric_matrix(empty, empty).shape == (0, 0)


def test_iou_zero_area_box():
    valid_box = np.array([[0.5, 0.5, 0.2, 0.2]], dtype="f8")
    zero_box = np.array([[0.5, 0.5, 0.0, 0.0]], dtype="f8")

    iou = create_iou_matrix(valid_box, zero_box)

    assert not np.isnan(iou).any()
    np.testing.assert_allclose(iou, [[0.0]], atol=1e-12)


# ----- Test Tracking 2D -----


def construct_detection2d(ids: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    detections = np.empty(len(ids), dtype=DetectionBBox)
    detections["id"] = ids
    detections["data"] = bboxes
    return detections


resolution = (0, 0)


@pytest.fixture
def tracker():
    return TrackingNode(
        Track2D,
        metric_fn=create_iou_metric_matrix,
        match_fn=global_matching,
        max_age=3,
        predict_fn=linear_predict,
        max_metric_value=0.7,
    )


def test_simple_2d_tracking(tracker):
    velocities = np.array([[0.01, 0.01, 0.0, 0.0], [0.0, -0.01, 0.0, 0.0]])
    bboxes = np.array([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])
    local_ids = np.array([0, 1])

    num_frames = 30
    for frame in range(num_frames):
        curr_bboxes = bboxes + velocities * frame
        expected_ids = np.array([1, 2])

        if frame % 2 == 1:
            curr_bboxes = curr_bboxes[::-1]
            expected_ids = expected_ids[::-1]

        detections = construct_detection2d(local_ids.copy(), curr_bboxes)

        frame_meta = FrameMetaData(frame, resolution, detections)
        frame_meta = tracker.process(frame_meta)

        np.testing.assert_array_equal(expected_ids, frame_meta.detections["id"])
        local_ids += np.array([1, 2])


def test_occlusion_within_max_age(tracker):
    vel = np.array([[0.02, 0.0, 0.0, 0.0]])
    box = np.array([[0.2, 0.2, 0.1, 0.1]])

    for frame in range(2):
        ctx = FrameMetaData(
            frame, resolution, construct_detection2d(np.array([0]), box + vel * frame)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1], ctx.detections["id"])

    for frame in range(2, 4):
        ctx = FrameMetaData(
            frame,
            resolution,
            construct_detection2d(np.array([], dtype=int), np.empty((0, 4))),
        )
        ctx = tracker.process(ctx)
        assert len(ctx.detections) == 0

    reappeared_box = box + vel * 4
    ctx = FrameMetaData(
        4, resolution, construct_detection2d(np.array([5]), reappeared_box)
    )
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([1], ctx.detections["id"])


def test_object_lost_after_max_age(tracker):
    vel = np.array([[0.02, 0.0, 0.0, 0.0]])
    box = np.array([[0.2, 0.2, 0.1, 0.1]])

    for frame in range(2):
        ctx = FrameMetaData(
            frame, resolution, construct_detection2d(np.array([0]), box + vel * frame)
        )
        tracker.process(ctx)

    for frame in range(2, 6):
        ctx = FrameMetaData(
            frame,
            resolution,
            construct_detection2d(np.array([], dtype=int), np.empty((0, 4))),
        )
        tracker.process(ctx)

    ctx = FrameMetaData(
        6, resolution, construct_detection2d(np.array([0]), box + vel * 6)
    )
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([2], ctx.detections["id"])


def test_jump_exceeds_max_iou_distance(tracker):
    box = np.array([[0.2, 0.2, 0.1, 0.1]])
    ctx = FrameMetaData(0, resolution, construct_detection2d(np.array([0]), box))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([1], ctx.detections["id"])

    jumped_box = np.array([[0.7, 0.2, 0.1, 0.1]])
    ctx = FrameMetaData(1, resolution, construct_detection2d(np.array([0]), jumped_box))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([2], ctx.detections["id"])


def test_new_object_spawns_mid_sequence(tracker):
    box1 = np.array([0.1, 0.1, 0.1, 0.1])
    vel1 = np.array([0.01, 0.0, 0.0, 0.0])

    box2 = np.array([0.6, 0.6, 0.1, 0.1])
    vel2 = np.array([0.0, 0.01, 0.0, 0.0])

    for frame in range(3):
        curr_b = np.array([box1 + vel1 * frame])
        ctx = FrameMetaData(
            frame, resolution, construct_detection2d(np.array([0]), curr_b)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1], ctx.detections["id"])

    for frame in range(3, 6):
        curr_b = np.array([box1 + vel1 * frame, box2 + vel2 * (frame - 3)])
        ctx = FrameMetaData(
            frame, resolution, construct_detection2d(np.array([0, 1]), curr_b)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1, 2], ctx.detections["id"])


def test_crossing_trajectories(tracker):
    boxes = np.array([[0.1, 0.3, 0.1, 0.1], [0.3, 0.1, 0.1, 0.1]])
    vel = np.array([[0.04, 0.0, 0.0, 0.0], [0.0, 0.04, 0.0, 0.0]])

    for frame in range(10):
        curr_boxes = boxes + vel * frame
        ctx = FrameMetaData(
            frame, resolution, construct_detection2d(np.array([0, 1]), curr_boxes)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1, 2], ctx.detections["id"])


def test_scaling_bbox_tracking(tracker):
    box = np.array([[0.5, 0.5, 0.1, 0.1]])
    scale_vel = np.array([[0.0, 0.0, 0.01, 0.01]])

    for frame in range(15):
        curr_box = box + scale_vel * frame
        ctx = FrameMetaData(
            frame, resolution, construct_detection2d(np.array([0]), curr_box)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1], ctx.detections["id"])


def test_empty_frames_sequence(tracker):
    for frame in range(5):
        ctx = FrameMetaData(
            frame,
            resolution,
            construct_detection2d(np.array([], dtype=int), np.empty((0, 4))),
        )
        ctx = tracker.process(ctx)
        assert len(ctx.detections) == 0
