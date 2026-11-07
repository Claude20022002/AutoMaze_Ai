"""Tests sur la vraie carte de Casablanca (simulator/assets/casablanca_drive.graphml)."""

import math
import random

import networkx as nx
import pytest

from simulator.config import LANDMARKS, MAP_FILE
from simulator.world.pathfinding import CostModel, dijkstra

pytestmark = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente : python -m simulator.world.road_map --download")


@pytest.fixture(scope="module")
def city():
    from simulator.world.road_map import load_map

    return load_map()


def test_map_is_strongly_connected(city):
    # Depuis chaque intersection, on peut rejoindre toutes les autres en respectant les sens uniques.
    assert len(city) > 1000
    assert nx.is_strongly_connected(city)


def test_map_is_in_meters(city):
    lengths = [d["length"] for _, _, d in city.edges(data=True)]
    assert 0 < min(lengths) and max(lengths) < 2000  # longueurs de tronçons plausibles en mètres


def test_landmarks_are_distinct_intersections(city):
    from simulator.world.road_map import landmark_node

    nodes = {landmark_node(city, name) for name in LANDMARKS}
    assert len(nodes) == len(LANDMARKS)


def test_matches_networkx_on_city(city):
    rng = random.Random(0)
    nodes = list(city.nodes)
    for _ in range(20):
        source, target = rng.sample(nodes, 2)
        expected = nx.dijkstra_path_length(city, source, target, weight="length")
        assert math.isclose(dijkstra(city, source, target).cost, expected)


def test_blocking_a_street_finds_a_detour(city):
    from simulator.world.road_map import landmark_node

    start = landmark_node(city, "Twin Center (Maârif)")
    goal = landmark_node(city, "Place Mohammed V (centre-ville)")
    best = dijkstra(city, start, goal)
    costs = CostModel(city)
    costs.block(best.edges[len(best.edges) // 2])  # un obstacle au milieu de l'itinéraire
    detour = dijkstra(city, start, goal, costs)
    assert detour.edges != best.edges
    assert detour.length_m >= best.length_m
