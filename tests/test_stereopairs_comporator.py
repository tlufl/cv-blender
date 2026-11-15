import numpy as np

from cv.core.math import INF, create_distance_matrix
from cv.processing.stereopairs_comporator import epipolar_filter


def test_create_distance_matrix():
    left_centers = np.array([[1, 0], [2, 0], [3, 0]])
    right_centers = np.array([[0, 0], [0, 0], [0, 0]])
    expected_matrix = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    result_matrix = create_distance_matrix(left_centers, right_centers)
    result_matrix = epipolar_filter(result_matrix, left_centers, right_centers, INF)
    np.testing.assert_allclose(result_matrix, expected_matrix)


def test_create_distance_matrix_with_threshold():
    left_centers = np.array([[3, 1], [2, 2], [3, 3]])
    right_centers = np.array([[1, 1], [5, 2.5], [-1, -1]])
    expected_matrix = np.array(
        [
            [2, INF, INF],
            [INF, INF, INF],
            [INF, INF, INF],
        ]
    )

    result_matrix = create_distance_matrix(left_centers, right_centers)
    result_matrix = epipolar_filter(result_matrix, left_centers, right_centers, 0.7)
    np.testing.assert_allclose(result_matrix, expected_matrix)


def test_cerate_distance_matrix_epipolar_constrains():
    left_centers = np.array([[1, 1], [2, 2], [3, 3]])
    right_centers = np.array([[100, 1], [200, 2], [300, 3]])
    expected_matrix = np.array([[INF] * 3] * 3)

    result_matrix = create_distance_matrix(left_centers, right_centers)
    result_matrix = epipolar_filter(result_matrix, left_centers, right_centers, INF)
    np.testing.assert_allclose(result_matrix, expected_matrix)
