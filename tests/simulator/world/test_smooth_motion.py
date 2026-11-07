"""Mouvement affiché de la voiture : pas de téléportation au demi-tour, pas de saut entre deux images,
voiture toujours sur la chaussée, et événements lisibles pendant les scénarios."""

import math

import pytest

from simulator.config import MAP_FILE, SimConfig
from simulator.contracts import CLASSES, Classifier, Prediction
from simulator.scenarios import build_episode

pytestmark = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente")


class Oracle(Classifier):
    name = "oracle (test)"

    def predict(self, image_rgb, true_label=None):
        return Prediction(true_label, 1.0, tuple(float(c == true_label) for c in CLASSES), 0.0)


@pytest.fixture(scope="module")
def city():
    from simulator.world.road_map import load_map

    return load_map()


def test_u_turn_drives_back_instead_of_teleporting(city):
    """Le moteur fait revenir la voiture au carrefour : d'un pas à l'autre, elle ne bouge jamais de plus d'un pas."""
    config = SimConfig()
    ep = build_episode("contournement", city, Oracle(), config, 0)
    step_m = config.car_speed_mps * config.step_seconds
    returning_seen = False
    while not ep.done:
        before = ep.car.position_xy()
        ep.step()
        after = ep.car.position_xy()
        returning_seen |= ep.car.returning
        assert math.dist(before, after) <= step_m * 1.01, f"saut de {math.dist(before, after):.1f} m au pas {ep.step_count}"
    assert ep.result.u_turns == 1 and returning_seen


def test_distance_counts_the_way_back(city):
    ep = build_episode("contournement", city, Oracle(), SimConfig(), 0)
    ep.run()
    driven = sum(float(city.edges[e]["length"]) for e in ep.car.driven_edges)
    assert ep.result.distance_m > driven            # le retour après le demi-tour est bien parcouru


def test_lane_offset_keeps_cars_on_the_road():
    pytest.importorskip("shapely")
    from simulator.render3d.world_mesh import LANE_OFFSET_M, lane_offset, road_width

    for data in ({"highway": "residential", "lanes": "1", "oneway": True}, {"width": "4", "oneway": False},
                 {"highway": "primary", "lanes": "6", "oneway": True}, {"highway": "secondary"}):
        offset = lane_offset(data)
        assert 0 <= offset <= LANE_OFFSET_M
        assert offset + 0.9 <= road_width(data) / 2 + 1e-9    # demi-largeur d'une voiture : 0,9 m


@pytest.mark.parametrize("key", ["contournement", "journee"])
def test_drawn_car_moves_smoothly_and_stays_on_its_street(city, key):
    """Rejoue l'affichage à 30 images/s : pas de saut, et le centre de la voiture reste sur sa rue."""
    pytest.importorskip("panda3d")
    from simulator.render3d.actors import ego_body
    from simulator.render3d.world_mesh import DRAW_LAG_M, road_width

    ep = build_episode(key, city, Oracle(), SimConfig(), 0)
    ep.car.record_path = True
    last = None
    while not ep.done:
        start = ep.car.distance_m
        ep.step()
        for frame in range(1, 31):
            s = start + (ep.car.distance_m - start) * frame / 30
            (x, y), (hx, hy) = ego_body(city, ep.car, s, 4.4)
            assert abs(math.hypot(hx, hy) - 1) < 1e-6
            if last is not None:
                assert math.dist(last, (x, y)) < 1.0, f"{key} : saut de {math.dist(last, (x, y)):.2f} m au pas {ep.step_count}"
            last = (x, y)
            # Axe des rues sous la voiture (dans un virage, son milieu passe dans le carrefour).
            front = s - DRAW_LAG_M
            nearby = [p for p in ep.car.trail if front - 8 <= p[0] <= front + 1]
            if nearby:                                   # tout début : la trace ne couvre pas encore la voiture
                gap = min(math.dist((x, y), (p[1], p[2])) for p in nearby)
                assert gap <= max(road_width(city.edges[p[3]]) for p in nearby) / 2 + 0.5


def test_events_tell_what_happens(city):
    ep = build_episode("pieton", city, Oracle(), SimConfig(), 0)
    ep.run()
    texts = [text for _, text in ep.events]
    assert any("Personne" in t and "s'arrêter" in t for t in texts)
    assert texts[-1] == "Arrivée à destination"
    detour = build_episode("contournement", city, Oracle(), SimConfig(), 0)
    detour.run()
    assert any(t.startswith("Demi-tour") for _, t in detour.events)
