"""La voiture : position le long de son itinéraire, déplacement, changement d'itinéraire."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from simulator.world.pathfinding import EdgeKey, Route


@dataclass
class Car:
    """Position = un tronçon de l'itinéraire (``edge_index``) et la distance parcourue sur ce tronçon."""

    graph: nx.MultiDiGraph
    route: Route
    edge_index: int = 0
    progress_m: float = 0.0
    driven_edges: list[EdgeKey] = field(default_factory=list)  # tronçons réellement parcourus
    distance_m: float = 0.0

    @property
    def arrived(self) -> bool:
        return self.edge_index >= len(self.route.edges)

    @property
    def current_edge(self) -> EdgeKey | None:
        return None if self.arrived else self.route.edges[self.edge_index]

    @property
    def next_node(self) -> int:
        """Intersection au bout du tronçon actuel (où un nouvel itinéraire peut commencer)."""
        return self.route.nodes[self.edge_index + 1] if not self.arrived else self.route.nodes[-1]

    def edge_length(self, edge: EdgeKey) -> float:
        return float(self.graph.edges[edge]["length"])

    def distance_to(self, edge_index: int, offset_m: float) -> float:
        """Distance à parcourir, le long de l'itinéraire, pour atteindre un point d'un tronçon devant la voiture."""
        if edge_index < self.edge_index:
            return float("inf")
        if edge_index == self.edge_index:
            return offset_m - self.progress_m
        ahead = self.edge_length(self.route.edges[self.edge_index]) - self.progress_m
        ahead += sum(self.edge_length(e) for e in self.route.edges[self.edge_index + 1 : edge_index])
        return ahead + offset_m

    def advance(self, distance_m: float) -> None:
        """Avance de ``distance_m`` mètres, en passant d'un tronçon au suivant si besoin."""
        while distance_m > 0 and not self.arrived:
            edge = self.route.edges[self.edge_index]
            remaining = self.edge_length(edge) - self.progress_m
            step = min(distance_m, remaining)
            self.progress_m += step
            self.distance_m += step
            distance_m -= step
            if self.progress_m >= self.edge_length(edge) - 1e-9:
                self.driven_edges.append(edge)
                self.edge_index += 1
                self.progress_m = 0.0

    def follow(self, new_route: Route, keep_current_edge: bool = True) -> None:
        """Remplace la fin de l'itinéraire.

        ``keep_current_edge=True`` : la voiture finit son tronçon actuel puis suit ``new_route``,
        qui doit partir du bout de ce tronçon. ``False`` : demi-tour, la voiture revient au début du
        tronçon actuel et suit ``new_route``, qui doit partir de là (simplification : demi-tour autorisé partout).
        """
        if self.arrived:
            return
        if keep_current_edge:
            current = self.route.edges[self.edge_index]
            assert new_route.nodes[0] == current[1], "le nouvel itinéraire doit partir du bout du tronçon actuel"
            self.route = Route(
                nodes=(current[0], *new_route.nodes),
                edges=(current, *new_route.edges),
                cost=new_route.cost,
                length_m=self.edge_length(current) + new_route.length_m,
            )
        else:
            assert new_route.nodes[0] == self.route.edges[self.edge_index][0]
            self.distance_m += self.progress_m  # chemin de retour jusqu'au début du tronçon
            self.route = new_route
            self.progress_m = 0.0
        self.edge_index = 0

    def heading(self, look_ahead_m: float = 6.0) -> tuple[float, float]:
        """Direction de la voiture (vecteur unitaire en mètres), mesurée vers un point un peu plus loin sur l'itinéraire."""
        here = self.position_xy()
        if self.arrived:
            edge = self.route.edges[-1] if self.route.edges else None
            if edge is None:
                return 0.0, 1.0
            ahead = point_on_edge(self.graph, edge, self.edge_length(edge))
            here = point_on_edge(self.graph, edge, max(self.edge_length(edge) - look_ahead_m, 0.0))
        else:
            index, offset = self.edge_index, self.progress_m + look_ahead_m
            while index < len(self.route.edges) - 1 and offset > self.edge_length(self.route.edges[index]):
                offset -= self.edge_length(self.route.edges[index])
                index += 1
            edge = self.route.edges[index]
            ahead = point_on_edge(self.graph, edge, min(offset, self.edge_length(edge)))
        dx, dy = ahead[0] - here[0], ahead[1] - here[1]
        norm = (dx * dx + dy * dy) ** 0.5
        return (dx / norm, dy / norm) if norm > 1e-6 else (0.0, 1.0)

    def position_xy(self) -> tuple[float, float]:
        """Coordonnées (mètres) de la voiture, en suivant la forme réelle de la rue si elle est connue."""
        if self.arrived:
            node = self.graph.nodes[self.route.nodes[-1]]
            return node["x"], node["y"]
        return point_on_edge(self.graph, self.route.edges[self.edge_index], self.progress_m)


def point_on_edge(graph: nx.MultiDiGraph, edge: EdgeKey, offset_m: float) -> tuple[float, float]:
    """Point situé à ``offset_m`` mètres du début d'un tronçon."""
    data = graph.edges[edge]
    length = float(data["length"]) or 1.0
    geometry = data.get("geometry")
    if geometry is not None:  # rue courbe : on suit la ligne OSM
        point = geometry.interpolate(min(offset_m / length, 1.0), normalized=True)
        return point.x, point.y
    u, v = graph.nodes[edge[0]], graph.nodes[edge[1]]
    t = min(offset_m / length, 1.0)
    return u["x"] + t * (v["x"] - u["x"]), u["y"] + t * (v["y"] - u["y"])
