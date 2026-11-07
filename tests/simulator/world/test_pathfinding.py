"""Tests de Dijkstra et des coûts dynamiques, sur des graphes construits à la main et aléatoires."""

import math
import random

import networkx as nx
import pytest

from simulator.world.pathfinding import CostModel, NoPathError, dijkstra


def small_city() -> nx.MultiDiGraph:
    """Petit quartier :

        0 --100--> 1 --100--> 3        chemin court : 0-1-3 (200 m)
        |                     ^
        +---150--> 2 --150----+        détour : 0-2-3 (300 m)

    Toutes les rues sont à double sens, sauf 2 -> 3 (sens unique).
    Deux tronçons parallèles relient 0 et 1 : la clé 1 (100 m) et la clé 0 (400 m).
    """
    g = nx.MultiDiGraph()
    for u, v, length in [(0, 1, 400), (0, 1, 100), (1, 3, 100), (0, 2, 150)]:
        g.add_edge(u, v, length=length)
        g.add_edge(v, u, length=length)
    g.add_edge(2, 3, length=150)  # sens unique
    return g


def test_shortest_route_and_parallel_edges():
    route = dijkstra(small_city(), 0, 3)
    assert route.nodes == (0, 1, 3)
    assert route.edges[0] == (0, 1, 1)  # le tronçon parallèle le plus court
    assert route.length_m == 200 and route.cost == 200


def test_same_source_and_target():
    route = dijkstra(small_city(), 2, 2)
    assert route.nodes == (2,) and route.edges == () and route.length_m == 0


def test_one_way_street_is_respected():
    # 3 -> 2 n'existe pas : pour aller de 3 à 2, il faut repasser par 1 puis 0.
    assert dijkstra(small_city(), 3, 2).nodes == (3, 1, 0, 2)


def test_blocked_edge_forces_detour():
    g = small_city()
    costs = CostModel(g)
    costs.block((1, 3, 0))
    route = dijkstra(g, 0, 3, costs)
    assert route.nodes == (0, 2, 3) and route.length_m == 300
    assert costs.is_blocked((3, 1, 0)), "le sens inverse d'une rue à double sens doit être bloqué aussi"


def test_penalty_changes_choice_but_not_length():
    g = small_city()
    costs = CostModel(g)
    costs.penalize((1, 3, 0), factor=2.0)  # 0-1-3 coûte maintenant 100 + 200 = 300, égalité avec le détour
    costs.penalize((0, 1, 1), factor=2.0)  # 0-1-3 coûte 400 : le détour (300) gagne
    route = dijkstra(g, 0, 3, costs)
    assert route.nodes == (0, 2, 3) and route.cost == 300 and route.length_m == 300


def test_temporary_block_is_released():
    g = small_city()
    costs = CostModel(g)
    costs.block((1, 3, 0), until_step=5)
    assert costs.release_expired(4) == []
    assert sorted(costs.release_expired(5)) == [(1, 3, 0), (3, 1, 0)]
    assert dijkstra(g, 0, 3, costs).nodes == (0, 1, 3)


def test_no_path_raises():
    g = small_city()
    costs = CostModel(g)
    costs.block((1, 3, 0))
    costs.block((2, 3, 0))
    with pytest.raises(NoPathError):
        dijkstra(g, 0, 3, costs)


def test_penalty_below_one_is_rejected():
    with pytest.raises(ValueError):
        CostModel(small_city()).penalize((0, 1, 0), factor=0.5)


@pytest.mark.parametrize("seed", range(20))
def test_matches_networkx_on_random_graphs(seed):
    """Notre Dijkstra doit trouver le même coût que celui de networkx (implémentation de référence)."""
    rng = random.Random(seed)
    g = nx.MultiDiGraph(nx.gnp_random_graph(40, 0.1, seed=seed, directed=True))
    for u, v, k in g.edges(keys=True):
        g.edges[u, v, k]["length"] = rng.uniform(10, 500)
    source, target = rng.sample(list(g.nodes), 2)
    try:
        expected = nx.dijkstra_path_length(g, source, target, weight="length")
    except nx.NetworkXNoPath:
        with pytest.raises(NoPathError):
            dijkstra(g, source, target)
        return
    route = dijkstra(g, source, target)
    assert math.isclose(route.cost, expected)
    assert route.nodes[0] == source and route.nodes[-1] == target
