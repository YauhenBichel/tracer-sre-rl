"""Hierarchical failure taxonomy for distributed system incidents.

Taxonomy definition is loaded from config/taxonomy.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import taxonomy_config
from src.constants import TAXONOMY_SEPARATOR


@dataclass(frozen=True)
class TaxonomyNode:
    name: str
    path: str
    description: str
    children: dict[str, "TaxonomyNode"] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_all_leaves(self) -> list["TaxonomyNode"]:
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children.values():
            leaves.extend(child.get_all_leaves())
        return leaves

    def find(self, path: str) -> "TaxonomyNode | None":
        parts = path.split(TAXONOMY_SEPARATOR, 1)
        child = self.children.get(parts[0])
        if child is None:
            return None
        return child if len(parts) == 1 else child.find(parts[1])


class TaxonomyBuilder:
    """Builds a mutable tree, then freezes it via build()."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._nodes[""] = {"name": "root", "path": "", "description": "Root", "children": {}}

    def add(self, path: str, description: str) -> "TaxonomyBuilder":
        parts = path.split(TAXONOMY_SEPARATOR)
        for i, part in enumerate(parts):
            node_path = TAXONOMY_SEPARATOR.join(parts[:i + 1])
            parent_path = TAXONOMY_SEPARATOR.join(parts[:i]) if i > 0 else ""
            if node_path not in self._nodes:
                self._nodes[node_path] = {
                    "name": part,
                    "path": node_path,
                    "description": description if i == len(parts) - 1 else "",
                    "children": {},
                }
                self._nodes[parent_path]["children"][part] = node_path
            elif i == len(parts) - 1:
                self._nodes[node_path]["description"] = description
        return self

    def build(self) -> TaxonomyNode:
        return self._build_node("")

    def _build_node(self, path: str) -> TaxonomyNode:
        data = self._nodes[path]
        children = {
            name: self._build_node(child_path)
            for name, child_path in data["children"].items()
        }
        return TaxonomyNode(
            name=data["name"],
            path=data["path"],
            description=data["description"],
            children=children,
        )


def build_default_taxonomy() -> TaxonomyNode:
    """Load taxonomy from config/taxonomy.yaml and build the tree."""
    builder = TaxonomyBuilder()
    for path, description in taxonomy_config().items():
        builder.add(path, description)
    return builder.build()
