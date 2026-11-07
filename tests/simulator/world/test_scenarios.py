"""Les scénarios du dossier Go/NoGo et les trajets quotidiens se réalisent quand chaque obstacle est bien classé."""

import pytest

from simulator.config import MAP_FILE, SimConfig
from simulator.contracts import CLASSES, Classifier, Prediction
from simulator.engine import ARRIVED, Episode, Stop
from simulator.scenarios import GONOGO, QUOTIDIEN, SCENARIOS, build_episode, check, main, realized

pytestmark = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente")


def one_hot(label: str) -> tuple[float, ...]:
    return tuple(1.0 if c == label else 0.0 for c in CLASSES)


class Oracle(Classifier):
    name = "oracle (test)"

    def predict(self, image_rgb, true_label=None):
        return Prediction(label=true_label, score=1.0, probs=one_hot(true_label), infer_ms=0.0)


class AlwaysBicycle(Classifier):
    name = "toujours vélo (test)"

    def predict(self, image_rgb, true_label=None):
        return Prediction(label="bicycle", score=1.0, probs=one_hot("bicycle"), infer_ms=0.0)


@pytest.fixture(scope="module")
def city():
    from simulator.world.road_map import load_map

    return load_map()


def play(city, key: str, episode: int = 0, classifier: Classifier | None = None) -> Episode:
    ep = build_episode(key, city, classifier or Oracle(), SimConfig(), episode)
    ep.run()
    return ep


def test_four_gonogo_scenarios_exist():
    assert [s.key for s in SCENARIOS.values() if s.family == GONOGO] == [
        "route_libre", "pieton", "route_bloquee", "contournement"]


@pytest.mark.parametrize("key", [k for k, s in SCENARIOS.items() if s.family == GONOGO])
def test_gonogo_scenario_is_realized(city, key):
    for episode in range(3):
        ep = play(city, key, episode)
        failed = [(c.label, c.detail) for c in check(key, ep) if not c.ok]
        assert not failed, f"{key}, épisode {episode} : {failed}"


@pytest.mark.parametrize("key", [k for k, s in SCENARIOS.items() if s.family == QUOTIDIEN])
def test_daily_trip_is_realized(city, key):
    for episode in range(3):
        ep = play(city, key, episode)
        assert realized(check(key, ep)), f"{key}, épisode {episode} : {check(key, ep)}"
        assert ep.obstacles, "un trajet quotidien comporte des obstacles"


def test_scenario_1_drives_the_dijkstra_path(city):
    ep = play(city, "route_libre")
    assert ep.result.distance_m == pytest.approx(ep.planned.length_m)
    assert not ep.obstacles and ep.result.waiting_steps == 0


def test_scenario_2_stops_then_keeps_the_same_route(city):
    ep = play(city, "pieton")
    assert [log.action for log in ep.result.encounters] == ["STOP_WAIT"]
    assert ep.result.waiting_steps >= SimConfig().wait_steps and ep.result.replans == 0


def test_scenario_3_and_4_differ_by_the_u_turn(city):
    blocked, detour = play(city, "route_bloquee"), play(city, "contournement")
    assert blocked.result.u_turns == 0 and blocked.result.replans >= 1
    assert detour.result.u_turns >= 1
    for ep in (blocked, detour):
        assert ep.obstacles[0].obstacle.edge not in ep.car.driven_edges


def test_misclassification_makes_the_scenario_fail(city):
    """Une personne prise pour un vélo : la voiture ralentit au lieu de s'arrêter, le scénario n'est pas réalisé."""
    ep = play(city, "pieton", classifier=AlwaysBicycle())
    failed = {c.label for c in check("pieton", ep) if not c.ok}
    assert "Arrêt devant la personne, puis départ" in failed


def test_same_obstacles_for_every_classifier(city):
    a = build_episode("heure_de_pointe", city, Oracle(), SimConfig(), 4)
    b = build_episode("heure_de_pointe", city, AlwaysBicycle(), SimConfig(), 4)
    assert [s.obstacle for s in a.obstacles] == [s.obstacle for s in b.obstacles]


def test_daily_trip_changes_with_the_episode(city):
    days = {tuple(s.obstacle for s in build_episode("journee", city, Oracle(), SimConfig(), e).obstacles)
            for e in range(4)}
    assert len(days) > 1


def test_planned_stops_are_reached_in_order_without_counting_as_waiting(city):
    ep = play(city, "courses")
    assert [name.split(" (")[0] for name, _ in ep.result.stops_reached] == ["Marché du Maârif", "Parc de la Ligue Arabe"]
    steps = [step for _, step in ep.result.stops_reached]
    assert steps == sorted(steps)
    assert ep.result.status == ARRIVED and ep.car.arrived


def test_stops_in_a_plain_episode(city):
    """Un arrêt sur un épisode sans scénario : la voiture le rejoint, y reste, puis termine le trajet."""
    from simulator.world.road_map import landmark_node

    stop = Stop(landmark_node(city, "Boulevard d'Anfa"), dwell_steps=7, name="Anfa")
    ep = Episode(city, Oracle(), SimConfig(), 0, obstacles=[], stops=[stop])
    ep.run()
    assert ep.result.status == ARRIVED and [n for n, _ in ep.result.stops_reached] == ["Anfa"]
    assert ep.result.waiting_steps == 0                       # un arrêt prévu n'est pas une attente subie
    assert ep.result.distance_m == pytest.approx(ep.planned.length_m)


def test_command_line(city, capsys):
    assert main(["--scenario", "route_libre", "--episodes", "1"]) == 0
    out = capsys.readouterr().out
    assert "RÉALISÉ" in out and "1 / 1 épisodes conformes" in out
