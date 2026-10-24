from config import config
from cv.core.pipeline import Pipeline
from cv.evaluation.flows import analize_two_objects
from cv.geometry.triangulator import TriangulateNode
from cv.io.csv_result_handler import CSVResultHandlerNode
from cv.io.video_source import VideoSourceNode
from cv.processing.frame_processor import ProcessFramesNode, process_frame
from cv.processing.stereopairs_comporator import CompareMethod, StereoPairComparator
from cv.processing.time_tracker import TimeTrackingNode, TrackMethod

if __name__ == "__main__":
    with CSVResultHandlerNode(config.path.proc_data) as csv_result_handler_node:
        pipeline = (
            Pipeline()
            .add(
                VideoSourceNode(
                    config.path.left_camera_src, config.path.right_camera_src
                )
            )
            .set_endpoint()
            .add(ProcessFramesNode(process_frame))
            .add(
                StereoPairComparator(
                    config.stereo_compare.max_distance,
                    config.stereo_compare.epipolar_threshold,
                    method=CompareMethod.GREEDY,
                )
            )
            .add(
                TimeTrackingNode(
                    config.time_tracking.min_iou, method=TrackMethod.GREEDY
                )
            )
            .add(
                TriangulateNode(
                    config.camera.focal_length_mm,
                    config.camera.sensor_width_mm,
                    config.camera.baseline,
                )
            )
            .add(csv_result_handler_node)
        )
        pipeline.run()

    analize_two_objects(
        config.path.true_data, config.path.proc_data, config.path.output_metrics
    )
