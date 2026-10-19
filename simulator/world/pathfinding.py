"""Recherche du plus court chemin sur le réseau routier : Dijkstra écrit à la main, coûts dynamiques.

Le graphe est un ``networkx.MultiDiGraph`` (format OSMnx) :
- les nœuds sont des intersections, les arêtes des tronçons de rue orientés (sens uniques respectés) ;
- deux intersections peuvent être reliées par plusieurs tronçons parallèles, distingués par une clé ``k`` ;
- l'attribut ``length`` d'une arête est sa longueur en mètres.

Les obstacles modifient les coûts pendant l'épisode (``CostModel``) : un tronçon peut être bloqué,
définitivement ou pendant quelques pas, ou rendu plus coûteux. Après chaque changement, la voiture
relance ``dijkstra`` depuis sa position actuelle.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from itertools import count

import networkx as nx

EdgeKey = tuple[int, int, int]  # (nœud de départ, nœud d'arrivée, clé du tronçon parallèle)


class NoPathError(RuntimeError):
    """Aucun chemin n'existe entre les deux nœuds avec les blocages actuels."""


@dataclass(frozen=True)
class Route:
    nodes: tuple[int, ...]       # intersections traversées, départ et arrivée compris
    edges: tuple[EdgeKey, ...]   # tronçons empruntés, dans l'ordre (len(edges) == len(nodes) - 1)
    cost: float                  # coût total (en mètres « pondérés » par les pénalités)
    length_m: float              # longueur réelle, sans pénalités


@dataclass
class CostModel:
    """Coût de chaque tronçon = longueur × pénalité, ou infini s'il est bloqué."""

    graph: nx.MultiDiGraph
    blocked: dict[EdgeKey, int | None] = field(default_factory=dict)  # tronçon -> pas de fin (None = définitif)
    penalties: dict[EdgeKey, float] = field(default_factory=dict)     # tronçon -> multiplicateur (>= 1)

    def block(self, edge: EdgeKey, until_step: int | None = None, both_directions: bool = True) -> None:
        """Bloque un tronçon. ``both_directions`` bloque aussi le sens inverse s'il existe (rue à double sens)."""
        for e in self._with_reverse(edge) if both_directions else [edge]:
            self.blocked[e] = until_step

    def penalize(self, edge: EdgeKey, factor: float, both_directions: bool = True) -> None:
        """Rend un tronçon ``factor`` fois plus coûteux (par exemple un vélo devant : on ralentit)."""
        if factor < 1:
            raise ValueError("une pénalité doit être >= 1")
        for e in self._with_reverse(edge) if both_directions else [edge]:
            self.penalties[e] = max(factor, self.penalties.get(e, 1.0))

    def release_expired(self, step: int) -> list[EdgeKey]:
        """Débloque les tronçons dont le blocage temporaire est terminé. Renvoie les tronçons libérés."""
        freed = [e for e, until in self.blocked.items() if until is not None and until <= step]
        for e in freed:
            del self.blocked[e]
        return freed

    def is_blocked(self, edge: EdgeKey) -> bool:
        return edge in self.blocked

    def cost(self, edge: EdgeKey) -> float:
        if edge in self.blocked:
            return math.inf
        length = float(self.graph.edges[edge].get("length", 1.0))
        return length * self.penalties.get(edge, 1.0)

    def _with_reverse(self, edge: EdgeKey) -> list[EdgeKey]:
        u, v, _ = edge
        reverse = [(v, u, k) for k in self.graph[v][u]] if self.graph.has_edge(v, u) else []
        return [edge, *reverse]


def dijkstra(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    costs: CostModel | None = None,
) -> Route:
    """Plus court chemin de ``source`` à ``target`` (algorithme de Dijkstra, file de priorité).

    Principe : on fixe les nœuds un par un, toujours celui dont la distance provisoire est la plus petite.
    Comme tous les coûts sont positifs, cette distance ne peut plus diminuer : elle est définitive.
    Complexité : O((N + A) log N) avec un tas binaire (N nœuds, A arêtes).
    """
    if source not in graph or target not in graph:
        raise KeyError("nœud de départ ou d'arrivée absent du graphe")
    costs = costs or CostModel(graph)

    dist: dict[int, float] = {source: 0.0}
    previous: dict[int, EdgeKey] = {}       # nœud -> tronçon par lequel on l'a atteint au mieux
    settled: set[int] = set()
    tie = count()                           # départage les égalités sans comparer les nœuds
    heap: list[tuple[float, int, int]] = [(0.0, next(tie), source)]

    while heap:
        d, _, u = heapq.heappop(heap)
        if u in settled:
            continue                        # entrée périmée : u a déjà été fixé avec une distance plus petite
        settled.add(u)
        if u == target:
            break
        for v, parallel in graph[u].items():
            if v in settled:
                continue
            # Plusieurs tronçons parallèles u -> v : on garde le moins coûteux.
            k, edge_cost = min(((k, costs.cost((u, v, k))) for k in parallel), key=lambda kc: kc[1])
            if math.isinf(edge_cost):
                continue                    # tous les tronçons u -> v sont bloqués
            new_d = d + edge_cost
            if new_d < dist.get(v, math.inf):
                dist[v] = new_d
                previous[v] = (u, v, k)
                heapq.heappush(heap, (new_d, next(tie), v))

    if target not in settled:
        raise NoPathError(f"aucun chemin de {source} à {target} avec les blocages actuels")

    edges: list[EdgeKey] = []
    node = target
    while node != source:
        edge = previous[node]
        edges.append(edge)
        node = edge[0]
    edges.reverse()
    nodes = (source, *(e[1] for e in edges))
    length = sum(float(graph.edges[e].get("length", 1.0)) for e in edges)
    return Route(nodes=nodes, edges=tuple(edges), cost=dist[target], length_m=length)
