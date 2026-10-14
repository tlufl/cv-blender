from config import config
from cv.core.pipeline import Pipeline
from cv.evaluation.flows import analize, analize_two_objects
from cv.geometry.triangulator import TriangulateNode
from cv.io.csv_result_handler import CSVResultHandlerNode
from cv.io.video_source import VideoSourceNode
from cv.processing.frame_processor import ProcessFramesNode, process_frame
from cv.processing.metadata_merger import MergeMetaDataNode

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
            .add(MergeMetaDataNode())
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
