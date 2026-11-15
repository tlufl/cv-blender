import numpy as np
import pytest

from cv.core.math import create_distance_matrix
from cv.core.world_data import Detection3D, WorldContext
from cv.processing.time_tracker import (
    Track3D,
    TrackingNode,
    global_matching,
    linear_predict,
)


def construct_detection3d(ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    detections = np.empty(len(ids), dtype=Detection3D)
    detections["id"] = ids
    detections["data"] = positions

    return detections


resolution = (0, 0)


@pytest.fixture
def tracker():
    return TrackingNode(
        Track3D,
        metric_fn=create_distance_matrix,
        match_fn=global_matching,
        max_age=3,
        predict_fn=linear_predict,
        max_metric_value=5,
    )


def test_simple_3d_tracking(tracker):
    velocities = np.array([[1.0, 1.0, 1.0], [0.0, -1.0, 0.0]])
    positions = np.array([[100.0, 100.0, 100.0], [50.0, 50.0, 70.0]])
    local_ids = np.array([0, 1])

    num_frames = 30
    for frame in range(num_frames):
        curr_positions = positions + velocities * frame
        expected_ids = np.array([1, 2])

        if frame % 2 == 1:
            curr_positions = curr_positions[::-1]
            expected_ids = expected_ids[::-1]

        detections = construct_detection3d(local_ids.copy(), curr_positions)

        world_context = WorldContext(frame, resolution, detections)
        world_context = tracker.process(world_context)

        np.testing.assert_array_equal(expected_ids, world_context.detections["id"])
        local_ids += np.array([1, 2])


def test_occlusion_within_max_age(tracker):
    vel = np.array([[1.0, 0.0, 0.0]])
    pos = np.array([[10.0, 10.0, 10.0]])

    for frame in range(2):
        ctx = WorldContext(
            frame, resolution, construct_detection3d(np.array([0]), pos + vel * frame)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1], ctx.detections["id"])

    for frame in range(2, 4):
        ctx = WorldContext(
            frame,
            resolution,
            construct_detection3d(np.array([], dtype=int), np.empty((0, 3))),
        )
        ctx = tracker.process(ctx)
        assert len(ctx.detections) == 0

    reappeared_pos = pos + vel * 4
    ctx = WorldContext(
        4, resolution, construct_detection3d(np.array([5]), reappeared_pos)
    )
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([1], ctx.detections["id"])


def test_object_lost_after_max_age(tracker):
    vel = np.array([[1.0, 0.0, 0.0]])
    pos = np.array([[10.0, 10.0, 10.0]])

    for frame in range(2):
        ctx = WorldContext(
            frame, resolution, construct_detection3d(np.array([0]), pos + vel * frame)
        )
        tracker.process(ctx)

    for frame in range(2, 6):
        ctx = WorldContext(
            frame,
            resolution,
            construct_detection3d(np.array([], dtype=int), np.empty((0, 3))),
        )
        tracker.process(ctx)

    ctx = WorldContext(
        6, resolution, construct_detection3d(np.array([0]), pos + vel * 6)
    )
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([2], ctx.detections["id"])


def test_jump_exceeds_max_distance(tracker):
    pos = np.array([[10.0, 10.0, 10.0]])
    ctx = WorldContext(0, resolution, construct_detection3d(np.array([0]), pos))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([1], ctx.detections["id"])

    jumped_pos = np.array([[16.0, 10.0, 10.0]])
    ctx = WorldContext(1, resolution, construct_detection3d(np.array([0]), jumped_pos))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([2], ctx.detections["id"])


def test_new_object_spawns_mid_sequence(tracker):
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.5, 0.0, 0.0])

    pos2 = np.array([50.0, 50.0, 50.0])
    vel2 = np.array([0.0, 0.5, 0.0])

    for frame in range(3):
        curr_p = np.array([pos1 + vel1 * frame])
        ctx = WorldContext(
            frame, resolution, construct_detection3d(np.array([0]), curr_p)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1], ctx.detections["id"])

    for frame in range(3, 6):
        curr_p = np.array([pos1 + vel1 * frame, pos2 + vel2 * (frame - 3)])
        ctx = WorldContext(
            frame, resolution, construct_detection3d(np.array([0, 1]), curr_p)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1, 2], ctx.detections["id"])


def test_crossing_trajectories(tracker):
    pos = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    for frame in range(10):
        curr_pos = pos + vel * frame
        ctx = WorldContext(
            frame, resolution, construct_detection3d(np.array([0, 1]), curr_pos)
        )
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1, 2], ctx.detections["id"])


def test_empty_frames_sequence(tracker):
    for frame in range(5):
        ctx = WorldContext(
            frame,
            resolution,
            construct_detection3d(np.array([], dtype=int), np.empty((0, 3))),
        )
        ctx = tracker.process(ctx)
        assert len(ctx.detections) == 0
