"""Trafic d'ambiance : suivi du véhicule de devant, feux, priorité de l'ego, piétons, déterminisme, épisode intact."""

import networkx as nx
import pytest

from simulator.config import MAP_FILE, SimConfig
from simulator.render3d import traffic as tr


def line_graph(n: int = 5, spacing: float = 100.0, attrs: dict | None = None) -> nx.MultiDiGraph:
    """Rue droite à double sens : nœuds 0..n-1 espacés de ``spacing`` mètres."""
    graph = nx.MultiDiGraph()
    for i in range(n):
        graph.add_node(i, x=i * spacing, y=0.0, **(attrs or {}).get(i, {}))
    for i in range(n - 1):
        for u, v in ((i, i + 1), (i + 1, i)):
            graph.add_edge(u, v, 0, length=spacing, highway="secondary", oneway=False)
    return graph


def cross_graph() -> nx.MultiDiGraph:
    """Carrefour à feux (nœud 0) et quatre branches de 100 m, prolongées de 100 m."""
    graph = nx.MultiDiGraph()
    graph.add_node(0, x=0.0, y=0.0, highway="traffic_signals")
    for i, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1)), start=1):
        graph.add_node(i, x=100.0 * dx, y=100.0 * dy)
        graph.add_node(i + 10, x=200.0 * dx, y=200.0 * dy)
        for u, v in ((0, i), (i, 0), (i, i + 10), (i + 10, i)):
            graph.add_edge(u, v, 0, length=100.0, highway="secondary", oneway=False)
    return graph


def vehicle(sim: tr.TrafficSim, edge, s: float, v: float, v0: float = 10.0, kind: str = "car") -> tr.Vehicle:
    vehicle = tr.Vehicle(sim._new_id(), kind, tr.VEHICLE_TYPES[kind][0], v0, edge, s, v)
    sim._extend_route(vehicle)
    sim.vehicles.append(vehicle)
    return vehicle


def run(sim: tr.TrafficSim, seconds: float, ego: tr.EgoState | None = None) -> None:
    for _ in range(int(seconds / 0.1)):
        sim.set_ego(ego)
        sim.step(0.1)


def test_follower_stops_behind_a_stopped_vehicle():
    sim = tr.TrafficSim(line_graph(), n_vehicles=0, n_pedestrians=0)
    leader = vehicle(sim, (1, 2, 0), 50.0, 0.0, v0=0.01)
    follower = vehicle(sim, (0, 1, 0), 0.0, 10.0)
    smallest = float("inf")
    for _ in range(400):
        sim.step(0.1)
        if follower.edge == leader.edge:
            smallest = min(smallest, leader.s - leader.length - follower.s)
    assert follower.edge == leader.edge and follower.v < 0.3
    assert 1.0 < smallest < 6.0                              # arrêté derrière, sans le toucher


def test_red_light_stops_traffic_and_green_lets_it_go():
    sim = tr.TrafficSim(cross_graph(), n_vehicles=0, n_pedestrians=0)
    controller = sim.signals[0]
    approach = (1, 0, 0)
    controller.phase = 3 if approach in controller.groups[0] else 0   # vert pour l'autre axe : rouge ici
    car = vehicle(sim, approach, 40.0, 8.0)
    run(sim, 15.0)
    assert sim.signal_state(approach) in ("red", "yellow")
    assert car.edge == approach and car.v < 0.3
    assert car.s <= 100.0 - tr.STOP_BEFORE_NODE_M + 0.5
    run(sim, 30.0)
    assert car.edge != approach                              # passé au vert


def test_signal_turns_green_for_the_ego():
    sim = tr.TrafficSim(cross_graph(), n_vehicles=0, n_pedestrians=0)
    controller = sim.signals[0]
    approach = (1, 0, 0)
    controller.phase = 3 if approach in controller.groups[0] else 0
    ego = tr.EgoState(approach, 40.0, 8.0, 12.0, (60.0, 0.0), [((0, 2, 0), 60.0)], (11, 1, 0), (-1.0, 0.0))
    run(sim, 8.0, ego)
    assert sim.signal_state(approach) == "green"


def test_traffic_avoids_the_ego_route_and_yields_to_it():
    graph = line_graph(8)
    sim = tr.TrafficSim(graph, seed=3, n_vehicles=12, n_pedestrians=0, radius_m=1000)
    upcoming = [((i, i + 1, 0), 90.0 + 100 * (i - 1)) for i in range(1, 4)]
    ego = tr.EgoState((0, 1, 0), 10.0, 8.0, 12.0, (10.0, 0.0), upcoming, None, (1.0, 0.0))
    sim.set_ego(ego)
    sim.populate()
    for _ in range(300):
        sim.set_ego(ego)
        sim.step(0.1)
        ahead = [v for v in sim.vehicles if v.edge in {(0, 1, 0)} | {e for e, _ in upcoming} and v.s > ego.s - 1]
        assert not ahead
    assert all(owner == "ego" for node, (owner, _) in sim.reservations.items() if node == 1)


def test_vehicles_stop_for_a_crossing_pedestrian():
    graph = line_graph(3, attrs={1: {"highway": "crossing"}})
    sim = tr.TrafficSim(graph, n_vehicles=0, n_pedestrians=0)
    sim.pedestrians.append(tr.Pedestrian(sim._new_id(), (0, 1, 0), 97.0, 1, 1.2, crossing=0.0, cross_duration=1e6))
    car = vehicle(sim, (0, 1, 0), 20.0, 10.0)
    oncoming = vehicle(sim, (2, 1, 0), 40.0, 10.0)
    run(sim, 20.0)
    assert car.edge == (0, 1, 0) and car.v < 0.3 and car.s < 97.0 - 2.0
    assert oncoming.edge == (2, 1, 0) and oncoming.v < 0.3


def test_pedestrians_walk_and_eventually_cross():
    graph = line_graph(3, attrs={1: {"highway": "crossing"}})
    sim = tr.TrafficSim(graph, seed=1, n_vehicles=0, n_pedestrians=0)
    walker = tr.Pedestrian(sim._new_id(), (0, 1, 0), 90.0, 1, 1.3)
    sim.pedestrians.append(walker)
    crossed = False
    for _ in range(600):
        sim.step(0.1)
        crossed |= walker.crossing is not None
    assert crossed


# ---------------------------------------------------------------------- vraie carte
needs_map = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente")


@pytest.fixture(scope="module")
def city():
    from simulator.main import load_perception
    from simulator.world.road_map import load_map

    graph, config = load_map(), SimConfig()
    classifier, images = load_perception("simulation", config)
    return graph, config, classifier, images


def _episode(city, number: int = 1):
    from simulator.engine import Episode

    graph, config, classifier, images = city
    return Episode(graph, classifier, config, number, image_for=images)


@needs_map
def test_traffic_is_deterministic(city):
    positions = []
    for _ in range(2):
        episode = _episode(city)
        sim = tr.TrafficSim(city[0], seed=11, blocked_edges={s.obstacle.edge for s in episode.obstacles})
        sim.set_ego(tr.ego_state(episode, 4.3))
        sim.populate()
        for _ in range(20):
            episode.step()
            run(sim, 1.0, tr.ego_state(episode, 4.3))
        positions.append([(v.edge, round(v.s, 6)) for v in sim.vehicles])
    assert positions[0] == positions[1]


@needs_map
def test_traffic_does_not_change_the_episode_and_stays_clear_of_the_ego(city):
    plain = _episode(city).run()
    episode = _episode(city)
    sim = tr.TrafficSim(city[0], seed=5, blocked_edges={s.obstacle.edge for s in episode.obstacles})
    sim.set_ego(tr.ego_state(episode, 4.3))
    sim.populate()
    overlaps = set()
    while not episode.done:
        episode.step()
        for _ in range(10):
            ego = tr.ego_state(episode, 4.3)
            sim.set_ego(ego)
            sim.step(0.1)
            for v in sim.vehicles:                              # rien juste devant l'ego, sur sa voie
                assert not (v.edge == ego.edge and 0 <= v.s - v.length - ego.s < 20), (v.kind, v.s, ego.s)
            by_edge = {}
            for v in sim.vehicles:
                by_edge.setdefault(v.edge, []).append(v)
            for vs in by_edge.values():
                vs.sort(key=lambda o: o.s)
                overlaps |= {(a.vid, b.vid) for a, b in zip(vs, vs[1:]) if b.s - b.length - a.s < -0.5}
    assert episode.result.status == plain.status and episode.step_count == plain.steps
    assert [e.step for e in episode.result.encounters] == [e.step for e in plain.encounters]
    assert len(overlaps) <= 2                                   # tolérance : anti-blocage aux carrefours
