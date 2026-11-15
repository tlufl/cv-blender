import numpy as np

from cv.core.frame_data import DetectionBBox, FrameMetaData
from cv.processing.constraints import (
    get_edge_distance_filter,
    get_object_distance_filter,
)

resolution = (0, 0)  # does not metter


# ----- edge distance filter -----


def test_edge_distance_filter_one_object():
    edge_distance_filter = get_edge_distance_filter(0.05, 0.05)
    detections = np.array([(0, [0.7, 0.8, 0.3, 0.1])], dtype=DetectionBBox)
    metadata = FrameMetaData(0, resolution, detections)

    metadata = edge_distance_filter(metadata)
    assert len(metadata.detections) != 0


def test_edge_distance_filter_object_perfectly_in_center():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array([(10, [0.5, 0.5, 0.2, 0.2])], dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 1
    assert filtered_metadata.detections["id"][0] == 10


def test_edge_distance_filter_object_too_close_to_left_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array([(20, [0.12, 0.5, 0.1, 0.2])], dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 0


def test_edge_distance_filter_object_too_close_to_right_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array([(30, [0.86, 0.5, 0.1, 0.2])], dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 0


def test_edge_distance_filter_object_too_close_to_top_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array([(40, [0.5, 0.14, 0.2, 0.1])], dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 0


def test_edge_distance_filter_object_too_close_to_bottom_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array([(50, [0.5, 0.86, 0.2, 0.1])], dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 0


def test_edge_distance_filter_multiple_objects_mixed_filtering():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array(
        [
            (100, [0.5, 0.5, 0.2, 0.2]),
            (101, [0.05, 0.5, 0.05, 0.1]),
            (102, [0.4, 0.4, 0.1, 0.1]),
            (103, [0.5, 0.95, 0.1, 0.2]),
        ],
        dtype=DetectionBBox,
    )

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 2
    np.testing.assert_array_equal(
        filtered_metadata.detections["id"], np.array([100, 102])
    )


def test_edge_distance_filter_empty_frame():
    edge_distance_filter = get_edge_distance_filter(0.05, 0.05)
    detections = np.empty(0, dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 0


def test_edge_distance_filter_zero_margins():
    edge_distance_filter = get_edge_distance_filter(0.0, 0.0)
    detections = np.array(
        [
            (1, [0.05, 0.5, 0.1, 0.2]),
            (2, [0.95, 0.5, 0.1, 0.2]),
        ],
        dtype=DetectionBBox,
    )

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.detections) == 2


# ----- object distance filter -----


def test_object_distance_filter_objects_too_close():
    distance_filter = get_object_distance_filter(x_margin=0.05, y_margin=0.3)
    detections = np.array(
        [
            (10, [0.5, 0.5, 0.1, 0.1]),
            (20, [0.5, 0.7, 0.1, 0.1]),
        ],
        dtype=DetectionBBox,
    )

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.detections) == 0


def test_object_distance_filter_objects_far_away():
    distance_filter = get_object_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array(
        [(10, [0.2, 0.2, 0.1, 0.1]), (20, [0.8, 0.8, 0.1, 0.1])], dtype=DetectionBBox
    )

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.detections) == 2
    np.testing.assert_array_equal(
        filtered_metadata.detections["id"], np.array([10, 20])
    )


def test_object_distance_filtert_objects_close_only_by_one_axis():
    distance_filter = get_object_distance_filter(x_margin=0.1, y_margin=0.1)
    detections = np.array(
        [(10, [0.2, 0.5, 0.1, 0.1]), (20, [0.8, 0.5, 0.1, 0.1])], dtype=DetectionBBox
    )

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.detections) == 2
    np.testing.assert_array_equal(
        filtered_metadata.detections["id"], np.array([10, 20])
    )


def test_object_distance_filter_empty_frame():
    distance_filter = get_object_distance_filter(x_margin=0.05, y_margin=0.05)
    detections = np.empty(0, dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.detections) == 0


def test_object_distance_filter_single_object():
    distance_filter = get_object_distance_filter(x_margin=0.9, y_margin=0.9)
    detections = np.array([(42, [0.5, 0.5, 0.2, 0.2])], dtype=DetectionBBox)

    metadata = FrameMetaData(0, resolution, detections)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.detections) == 1
    assert filtered_metadata.detections["id"][0] == 42
