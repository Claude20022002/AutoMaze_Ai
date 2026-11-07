"""La voiture : position le long de son itinéraire, déplacement, changement d'itinéraire.

Demi-tour : la voiture revient réellement, en sens inverse, jusqu'au début du tronçon où elle se trouvait
(``return_m`` mètres à parcourir), puis suit son nouvel itinéraire. Cela prend du temps, comme en vrai.

Trace (``record_path``, activée par les vues) : la voiture garde les points réellement parcourus, repérés par la
distance totale parcourue. Une vue peut ainsi placer la carrosserie exactement sur le chemin entre deux pas du
moteur (``pose_at``), au lieu de tirer un trait droit qui couperait les virages. Le mode sans affichage ne
l'active pas : aucun coût pour ``compare.py``.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field

import networkx as nx

from simulator.world.pathfinding import EdgeKey, Route

TRAIL_STEP_M = 2.0      # écart maximal entre deux points de la trace
TRAIL_KEEP_M = 120.0    # longueur de trace conservée derrière la voiture

TrailPoint = tuple[float, float, float, EdgeKey, bool]   # (distance parcourue, x, y, tronçon, en marche arrière sur ce tronçon)


@dataclass
class Car:
    """Position = un tronçon de l'itinéraire (``edge_index``) et la distance parcourue sur ce tronçon."""

    graph: nx.MultiDiGraph
    route: Route
    edge_index: int = 0
    progress_m: float = 0.0
    driven_edges: list[EdgeKey] = field(default_factory=list)  # tronçons réellement parcourus
    distance_m: float = 0.0
    return_edge: EdgeKey | None = None   # demi-tour en cours : tronçon parcouru en sens inverse
    return_m: float = 0.0                # distance restante jusqu'au début de ce tronçon
    record_path: bool = False            # garder la trace des points parcourus (vues)
    trail: list[TrailPoint] = field(default_factory=list)

    @property
    def arrived(self) -> bool:
        return self.edge_index >= len(self.route.edges) and self.return_m <= 0

    @property
    def returning(self) -> bool:
        """Vrai pendant le retour qui suit un demi-tour."""
        return self.return_m > 0

    @property
    def current_edge(self) -> EdgeKey | None:
        """Tronçon de l'itinéraire en cours (pendant un demi-tour : le premier tronçon après le retour)."""
        return None if self.edge_index >= len(self.route.edges) else self.route.edges[self.edge_index]

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
            return self.return_m + offset_m - self.progress_m
        ahead = self.edge_length(self.route.edges[self.edge_index]) - self.progress_m
        ahead += sum(self.edge_length(e) for e in self.route.edges[self.edge_index + 1 : edge_index])
        return self.return_m + ahead + offset_m

    def advance(self, distance_m: float) -> None:
        """Avance de ``distance_m`` mètres : fin du retour d'un demi-tour, puis tronçon après tronçon."""
        self._record()
        while distance_m > 1e-9 and self.return_m > 0:
            step = min(distance_m, self.return_m, TRAIL_STEP_M)
            self.return_m -= step
            self.distance_m += step
            distance_m -= step
            if self.return_m <= 1e-9:
                self.return_m, self.return_edge = 0.0, None
            self._record()
        while distance_m > 1e-9 and not self.arrived:
            edge = self.route.edges[self.edge_index]
            remaining = self.edge_length(edge) - self.progress_m
            step = min(distance_m, remaining, TRAIL_STEP_M if self.record_path else remaining)
            self.progress_m += step
            self.distance_m += step
            distance_m -= step
            if self.progress_m >= self.edge_length(edge) - 1e-9:
                self.driven_edges.append(edge)
                self.edge_index += 1
                self.progress_m = 0.0
            self._record()

    def follow(self, new_route: Route, keep_current_edge: bool = True) -> None:
        """Remplace la fin de l'itinéraire.

        ``keep_current_edge=True`` : la voiture finit son tronçon actuel puis suit ``new_route``,
        qui doit partir du bout de ce tronçon. ``False`` : demi-tour, la voiture revient au début du
        tronçon actuel (en sens inverse, ``return_m`` mètres) et suit ``new_route``, qui doit partir de là
        (simplification : demi-tour autorisé partout, même dans une rue à sens unique).
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
            if not self.returning:              # déjà en train de revenir : le retour continue tel quel
                self.return_edge, self.return_m = self.route.edges[self.edge_index], self.progress_m
            self.route = new_route
            self.progress_m = 0.0
        self.edge_index = 0

    def start_leg(self, new_route: Route) -> None:
        """Étape suivante d'un trajet, depuis l'arrêt où se trouve la voiture (distance parcourue conservée)."""
        assert self.arrived and new_route.nodes[0] == self.route.nodes[-1], "l'étape doit partir de l'arrêt"
        self.route, self.edge_index, self.progress_m = new_route, 0, 0.0

    # ------------------------------------------------------------------ position
    def _point_ahead(self, ahead_m: float) -> tuple[float, float]:
        """Point situé ``ahead_m`` mètres plus loin sur le chemin que la voiture va suivre (retour compris)."""
        if self.returning:
            if ahead_m <= self.return_m:
                return point_on_edge(self.graph, self.return_edge, self.return_m - ahead_m)
            ahead_m -= self.return_m
            index, offset = self.edge_index, ahead_m
        else:
            index, offset = self.edge_index, self.progress_m + ahead_m
        if not self.route.edges:
            node = self.graph.nodes[self.route.nodes[-1]]
            return node["x"], node["y"]
        index = min(index, len(self.route.edges) - 1)
        while index < len(self.route.edges) - 1 and offset > self.edge_length(self.route.edges[index]):
            offset -= self.edge_length(self.route.edges[index])
            index += 1
        edge = self.route.edges[index]
        return point_on_edge(self.graph, edge, min(offset, self.edge_length(edge)))

    def heading(self, look_ahead_m: float = 6.0) -> tuple[float, float]:
        """Direction de la voiture (vecteur unitaire en mètres), mesurée vers un point un peu plus loin sur l'itinéraire."""
        if self.arrived:
            edge = self.route.edges[-1] if self.route.edges else None
            if edge is None:
                return 0.0, 1.0
            ahead = point_on_edge(self.graph, edge, self.edge_length(edge))
            here = point_on_edge(self.graph, edge, max(self.edge_length(edge) - look_ahead_m, 0.0))
        else:
            here, ahead = self.position_xy(), self._point_ahead(look_ahead_m)
        dx, dy = ahead[0] - here[0], ahead[1] - here[1]
        norm = (dx * dx + dy * dy) ** 0.5
        return (dx / norm, dy / norm) if norm > 1e-6 else (0.0, 1.0)

    def position_xy(self) -> tuple[float, float]:
        """Coordonnées (mètres) de la voiture, en suivant la forme réelle de la rue si elle est connue."""
        if self.returning:
            return point_on_edge(self.graph, self.return_edge, self.return_m)
        if self.arrived:
            node = self.graph.nodes[self.route.nodes[-1]]
            return node["x"], node["y"]
        return point_on_edge(self.graph, self.route.edges[self.edge_index], self.progress_m)

    def _edge_here(self) -> tuple[EdgeKey | None, bool]:
        """Tronçon où se trouve physiquement la voiture, et s'il est parcouru en sens inverse."""
        if self.returning:
            return self.return_edge, True
        if self.edge_index < len(self.route.edges):
            return self.route.edges[self.edge_index], False
        return (self.route.edges[-1] if self.route.edges else None), False

    # ------------------------------------------------------------------ trace (vues)
    def _record(self) -> None:
        if not self.record_path:
            return
        x, y = self.position_xy()
        edge, backwards = self._edge_here()
        if self.trail and self.trail[-1][0] >= self.distance_m - 1e-9:
            self.trail[-1] = (self.distance_m, x, y, edge, backwards)      # même distance : on remplace
        else:
            self.trail.append((self.distance_m, x, y, edge, backwards))
        cut = bisect_left(self.trail, self.distance_m - TRAIL_KEEP_M, key=lambda p: p[0])
        if cut > 1:
            del self.trail[: cut - 1]

    def trail_point(self, distance_m: float) -> TrailPoint:
        """Point de la trace à une distance parcourue donnée (interpolé entre deux points enregistrés)."""
        if not self.trail:
            self._record_now()
        trail = self.trail
        if distance_m <= trail[0][0]:
            return trail[0]
        if distance_m >= trail[-1][0]:
            return trail[-1]
        i = bisect_left(trail, distance_m, key=lambda p: p[0])
        (d0, x0, y0, _, _), (d1, x1, y1, edge, backwards) = trail[i - 1], trail[i]
        t = (distance_m - d0) / (d1 - d0) if d1 > d0 else 1.0
        return distance_m, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, edge, backwards

    def _record_now(self) -> None:
        recording, self.record_path = self.record_path, True
        self._record()
        self.record_path = recording

    def pose_at(self, distance_m: float, body_m: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Avant de la voiture à ``distance_m`` (distance parcourue) et cap donné par son arrière, ``body_m`` mètres
        plus tôt sur la trace : la carrosserie suit le chemin comme une vraie voiture dans un virage."""
        _, fx, fy, _, _ = self.trail_point(distance_m)
        _, rx, ry, _, _ = self.trail_point(distance_m - body_m)
        dx, dy = fx - rx, fy - ry
        norm = math.hypot(dx, dy)
        if norm < 0.5:                         # début de trace : pas encore de recul, on regarde devant
            return (fx, fy), self.heading()
        return (fx, fy), (dx / norm, dy / norm)


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
