import csv

from cv.core.world_data import WorldContext


class CSVResultHandlerNode:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._writer = None

    def open(self) -> None:
        self._file = open(self.file_path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["Frame_id", "Object_id", "X", "Y", "Z"])

    def process(self, world_context: WorldContext) -> None:
        print("process CSV")
        if self._writer is None:
            raise RuntimeError()

        if world_context.positions.size == 0:
            return

        frame_id = world_context.frame_id
        for obj_id, position in zip(world_context.object_ids, world_context.positions):
            x, y, z = position
            row = [
                frame_id,
                obj_id,
                round(float(x), 4),
                round(float(y), 4),
                round(float(z), 4),
            ]
            self._writer.writerow(row)

    def close(self) -> None:
        if self._file:
            self._file.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
