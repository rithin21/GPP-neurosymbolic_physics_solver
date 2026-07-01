from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PhysicsKnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph, meta: dict[str, str]):
        self.graph = graph
        self.meta = meta

    @classmethod
    def from_json(cls, path: Path | str) -> "PhysicsKnowledgeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cls._validate_graph_data(data)
        graph = nx.MultiDiGraph()
        for node in data["nodes"]:
            node_id = node["id"]
            attrs = {key: value for key, value in node.items() if key != "id"}
            graph.add_node(node_id, **attrs)
        for edge in data["edges"]:
            graph.add_edge(edge["from"], edge["to"], type=edge["type"])
        return cls(graph, meta=data["meta"])

    @staticmethod
    def _validate_graph_data(data: dict[str, Any]) -> None:
        required_top_level = {"meta", "nodes", "edges"}
        missing = required_top_level - set(data)
        if missing:
            raise ValueError(f"Knowledge graph missing keys: {sorted(missing)}")
        for key in ("domain", "subject", "chapter"):
            if key not in data["meta"]:
                raise ValueError(f"Knowledge graph meta missing {key!r}")
        node_ids = set()
        for node in data["nodes"]:
            if "id" not in node or "type" not in node:
                raise ValueError(f"Invalid graph node: {node}")
            node_ids.add(node["id"])
        for edge in data["edges"]:
            if not {"from", "to", "type"}.issubset(edge):
                raise ValueError(f"Invalid graph edge: {edge}")
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                raise ValueError(f"Edge references unknown node: {edge}")

    def node(self, node_id: str) -> dict[str, Any]:
        return dict(self.graph.nodes[node_id])

    #give all the nodes of type node_type
    #return a list of tuples that stores the id and dict storing the other details
    def nodes_by_type(self, node_type: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (node_id, attrs)
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == node_type
        ]

    def outgoing(self, node_id: str, edge_type: str) -> list[str]:
        results: list[str] = []
        for _, target, attrs in self.graph.out_edges(node_id, data=True):
            if attrs.get("type") == edge_type:
                results.append(target)
        return results

    def incoming(self, node_id: str, edge_type: str) -> list[str]:
        results: list[str] = []
        for source, _, attrs in self.graph.in_edges(node_id, data=True):
            if attrs.get("type") == edge_type:
                results.append(source)
        return results


def load_default_graph() -> PhysicsKnowledgeGraph:
    return PhysicsKnowledgeGraph.from_json(PROJECT_ROOT / "data" / "kinematics_graph.json")
