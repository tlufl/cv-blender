import numpy as np

from cv.evaluation.data import TrajectoryData, TrajectoryErrors


def calculate_metrics(
    true_arr: np.ndarray, proc_arr: np.ndarray
) -> tuple[TrajectoryData, TrajectoryErrors]:
    min_len = min(len(true_arr), len(proc_arr))
    true_arr = true_arr[:min_len]
    proc_arr = proc_arr[:min_len].copy()

    true_distance = np.linalg.norm(true_arr, axis=1)
    proc_distance = np.linalg.norm(proc_arr, axis=1)

    errors = TrajectoryErrors(
        distance=true_distance - proc_distance,
        x=true_arr[:, 0] - proc_arr[:, 0],
        y=true_arr[:, 1] - proc_arr[:, 1],
        z=true_arr[:, 2] - proc_arr[:, 2],
    )

    data = TrajectoryData(
        true_arr=true_arr, proc_arr=proc_arr, true_distance=true_distance
    )
    return data, errors
