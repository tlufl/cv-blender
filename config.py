import os

import yaml
from dotenv import load_dotenv

load_dotenv()


class EnvConfig:
    left_camera_src = os.getenv("LEFT_CAMERA_SOURCE_PATH", "")
    right_camera_src = os.getenv("RIGHT_CAMERA_SOURCE_PATH", "")
    true_data = os.getenv("TRUE_DATA_PATH", "")
    proc_data = os.getenv("PROCESSED_DATA_PATH", "")
    output_metrics = os.getenv("OUTPUT_METRICS_PATH", "")


class CameraConfig:
    def __init__(self, camera_data: dict[str, str]):
        self.baseline = float(camera_data.get("baseline", 30))
        self.focal_length_mm = float(camera_data.get("focal_length_mm", 50))
        self.sensor_width_mm = float(camera_data.get("sensor_width_mm", 36))


class VideoConfig:
    def __init__(self, video_data: dict[str, str]):
        self.resolution = list(video_data.get("resolution", [768, 576]))
        self.fps = int(video_data.get("fps", 25))


class ProjectConfig:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as file:
            self._yaml_data = yaml.safe_load(file)

        camera_data = self._yaml_data.get("camera", {})
        self.camera = CameraConfig(camera_data)

        video_data = self._yaml_data.get("video", {})
        self.video = VideoConfig(video_data)

        self.path = EnvConfig()


config = ProjectConfig("config.yaml")

if __name__ == "__main__":
    print(config)
