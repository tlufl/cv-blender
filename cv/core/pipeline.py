from dataclasses import dataclass, field
from typing import Any, Protocol, Self


class ProcessingNode(Protocol):
    def process(self, *args, **kwargs) -> Any:
        pass


@dataclass
class Pipeline:
    _nodes: list[ProcessingNode] = field(default_factory=list)
    _endpoint: ProcessingNode | None = None

    def set_endpoint(self) -> Self:
        if self._nodes:
            self._endpoint = self._nodes[-1]
        return self

    def add(self, node: ProcessingNode) -> Self:
        self._nodes.append(node)
        return self

    def __rshift__(self, node: ProcessingNode) -> Self:
        return self.add(node)

    def run(self, *args) -> None:
        if not self._nodes:
            return

        data = tuple()

        while True:
            for node in self._nodes:
                data = node.process(*data)

                if self._endpoint and node is self._endpoint and data is None:
                    return

                if data is None:
                    data = tuple()
                elif not isinstance(data, tuple):
                    data = (data,)
