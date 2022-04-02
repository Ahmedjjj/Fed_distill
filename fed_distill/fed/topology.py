from typing import Iterable

from fed_distill.fed.node import Node


class Topology:
    def __init__(self, student_nodes: Iterable[Node], master_node: Node) -> None:
        self.student_nodes = list(student_nodes)
        self.master_node = master_node
        for node in self.student_nodes:
            node.register_topology(self)

        self.nodes_by_id = dict(zip(range(len(student_nodes)), student_nodes))
        self.nodes_by_id[len(student_nodes)] = master_node

    def __getitem__(self, id: int) -> Node:
        return self.nodes_by_id[id]

    @property
    def master_id(self) -> int:
        return len(self.student_nodes)
