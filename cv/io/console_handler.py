from cv.core.world_data import WorldContext


class ConsoleResultHandlerNode:
    def process(self, world_context: WorldContext) -> None:
        print(
            f"3D pos: {world_context.detections['data']}, id: {world_context.detections['id']}"
        )
