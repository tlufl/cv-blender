import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cv.evaluation.metrics import calculate_metrics
from cv.evaluation.plotting import plot_3d_trail, plot_error_graphs


def analize_two_objects(
    true_data_path: str, processed_data_path: str, save_path: str
) -> None:
    true_df = pd.read_csv(true_data_path, sep=";")
    proc_df = pd.read_csv(processed_data_path)
    proc_df[["X"]] -= 15
    proc_df[["X", "Y"]] *= -1
    proc_df[["X", "Y", "Z"]] = proc_df[["Z", "X", "Y"]]

    proc_arr1 = proc_df.loc[proc_df["Object_id"] == 0, ["X", "Y", "Z"]].to_numpy()
    proc_arr2 = proc_df.loc[proc_df["Object_id"] == 1, ["X", "Y", "Z"]].to_numpy()

    true_arr1 = true_df[["Xl", "Yl", "Zl"]].to_numpy()
    true_arr2 = true_df[["Xr", "Yr", "Zr"]].to_numpy()

    dist1 = np.sum((true_arr1[0] - proc_arr1[0]) ** 2)
    dist2 = np.sum((true_arr1[0] - proc_arr2[0]) ** 2)

    if dist1 > dist2:
        proc_arr1, proc_arr2 = proc_arr2, proc_arr1

    data1, errors1 = calculate_metrics(true_arr1, proc_arr1)
    data2, errors2 = calculate_metrics(true_arr2, proc_arr2)

    figure = plt.figure(figsize=(30, 10))
    grid = figure.add_gridspec(4, 7, wspace=1.5, hspace=0.2)

    axis_3d = figure.add_subplot(grid[:, :3], projection="3d")

    axes_2d_first_object = [
        figure.add_subplot(grid[0, 3:5]),
        figure.add_subplot(grid[1, 3:5]),
        figure.add_subplot(grid[2, 3:5]),
        figure.add_subplot(grid[3, 3:5]),
    ]
    axes_2d_second_object = [
        figure.add_subplot(grid[0, 5:]),
        figure.add_subplot(grid[1, 5:]),
        figure.add_subplot(grid[2, 5:]),
        figure.add_subplot(grid[3, 5:]),
    ]

    plot_3d_trail(axis_3d, data1)
    plot_3d_trail(axis_3d, data2)
    plot_error_graphs(axes_2d_first_object, data1, errors1)
    plot_error_graphs(axes_2d_second_object, data2, errors2)

    plt.savefig(save_path, bbox_inches="tight")


def analize(true_data_path: str, processed_data_path: str, save_path: str) -> None:
    true_df = pd.read_csv(true_data_path)
    proc_df = pd.read_csv(processed_data_path)

    true_arr = true_df[["X", "Y", "Z"]].to_numpy()
    proc_arr = proc_df[["X", "Y", "Z"]].to_numpy()

    data, errors = calculate_metrics(true_arr, proc_arr)

    figure = plt.figure(figsize=(20, 10))
    grid = figure.add_gridspec(4, 6, wspace=1.5, hspace=0.2)

    axis_3d = figure.add_subplot(grid[:, :3], projection="3d")

    axes_2d = [
        figure.add_subplot(grid[0, 3:]),
        figure.add_subplot(grid[1, 3:]),
        figure.add_subplot(grid[2, 3:]),
        figure.add_subplot(grid[3, 3:]),
    ]

    plot_3d_trail(axis_3d, data)
    plot_error_graphs(axes_2d, data, errors)

    plt.savefig(save_path, bbox_inches="tight")
