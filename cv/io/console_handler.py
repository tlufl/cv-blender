from cv.core.world_data import WorldContext


class ConsoleResultHandlerNode:
    def process(self, world_context: WorldContext) -> None:
        print(f"3D pos: {world_context.positions}, id: {world_context.object_ids}")
