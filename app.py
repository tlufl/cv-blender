from config import config
from cv.core.engine import Engine
from cv.core.pipeline import Pipeline
from cv.evaluation.flows import analize, analize_two_objects
from cv.geometry.triangulator import create_triangulator
from cv.io.csv_result_handler import CSVResultHandler
from cv.io.console_handler import ConsoleResultHandler
from cv.io.video_source import VideoSource
from cv.io.image_source import ImageSource
from cv.processing.frame_processor import process_frame
from cv.processing.metadata_merger import dont_merge_metadata, MetaDataMerger

if __name__ == "__main__":
    triangulate = create_triangulator(
        f_mm=config.camera.focal_length_mm,
        w_mm=config.camera.sensor_width_mm,
        baseline=config.camera.baseline,
    )
    frame_reader = VideoSource(
        config.path.left_camera_src, config.path.right_camera_src
    )
    merger = MetaDataMerger()
    engine = Engine(process_frame, merger.merge_metadata, triangulate)
    res_handler = CSVResultHandler(config.path.proc_data)
    with res_handler:
        pipeline = Pipeline(engine, frame_reader, res_handler)
        pipeline.start()

    analize(config.path.true_data, config.path.proc_data, config.path.output_metrics)
