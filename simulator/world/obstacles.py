"""Placement des obstacles sur l'itinéraire et suivi de leur état pendant l'épisode."""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from simulator.contracts import CLASSES, Obstacle
from simulator.world.pathfinding import Route


@dataclass
class ObstacleState:
    """État d'un obstacle pendant un épisode (l'``Obstacle`` du contrat, lui, ne change pas)."""

    obstacle: Obstacle
    detected_step: int | None = None   # pas où la voiture l'a vu et fait classer
    cleared_step: int | None = None    # pas où il libère la voie (None = immobile ou pas encore vu)
    handled: bool = False              # une décision a été prise : on ne le reclasse pas
    predicted_label: str | None = None # classe donnée par le classifieur (affichée sur la carte)
    predicted_score: float | None = None

    def present(self, step: int) -> bool:
        """L'obstacle bloque-t-il encore physiquement la voie ?"""
        return self.cleared_step is None or step < self.cleared_step


def place_obstacles(
    graph: nx.MultiDiGraph,
    route: Route,
    count: int,
    rng: random.Random,
    labels: tuple[str, ...] = CLASSES,
    min_gap: int = 2,
) -> list[Obstacle]:
    """Place jusqu'à ``count`` obstacles sur des tronçons distincts de l'itinéraire.

    Les deux premiers et le dernier tronçon restent libres (la voiture doit avoir le temps de voir l'obstacle),
    et deux obstacles sont séparés d'au moins ``min_gap`` tronçons. La vraie classe est tirée au hasard.
    """
    candidates = list(range(2, len(route.edges) - 1))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for index in candidates:
        if len(chosen) == count:
            break
        if all(abs(index - other) >= min_gap for other in chosen):
            chosen.append(index)

    obstacles = []
    for index in sorted(chosen):
        edge = route.edges[index]
        length = float(graph.edges[edge]["length"])
        obstacles.append(Obstacle(edge=edge, true_label=rng.choice(labels), image_path=None, offset_m=length / 2))
    return obstacles
