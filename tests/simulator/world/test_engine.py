"""Tests de la boucle d'épisode avec de faux classifieurs (sans aucun modèle)."""

import pytest

from simulator.config import MAP_FILE, SimConfig
from simulator.contracts import CLASSES, Classifier, Prediction
from simulator.engine import ARRIVED, BLOCKED, TIMEOUT, Episode, save_encounters_csv

pytestmark = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente")


def one_hot(label: str) -> tuple[float, ...]:
    return tuple(1.0 if c == label else 0.0 for c in CLASSES)


class Oracle(Classifier):
    """Connaît toujours la vraie classe : la voiture doit arriver sans aucune collision."""

    name = "oracle (test)"

    def predict(self, image_rgb, true_label=None):
        return Prediction(label=true_label, score=1.0, probs=one_hot(true_label), infer_ms=0.0)


class AlwaysBicycle(Classifier):
    """Voit des vélos partout : ralentit au lieu de contourner les véhicules arrêtés."""

    name = "toujours vélo (test)"

    def predict(self, image_rgb, true_label=None):
        return Prediction(label="bicycle", score=1.0, probs=one_hot("bicycle"), infer_ms=0.0)


@pytest.fixture(scope="module")
def city():
    from simulator.world.road_map import load_map

    return load_map()


def test_oracle_100_seeds_reach_goal_without_collision(city):
    config = SimConfig()
    for episode in range(100):
        result = Episode(city, Oracle(), config, episode).run()
        assert result.status in (ARRIVED, BLOCKED), f"épisode {episode} : {result.status}"
        assert result.collisions == 0, f"épisode {episode} : collision avec un classifieur parfait"
        assert all(log.correct for log in result.encounters)


def test_wrong_classifier_causes_collisions(city):
    config = SimConfig()
    results = [Episode(city, AlwaysBicycle(), config, e).run() for e in range(30)]
    assert all(r.status in (ARRIVED, BLOCKED, TIMEOUT) for r in results)
    assert sum(r.collisions for r in results) > 0


def test_same_seed_same_obstacles_for_every_classifier(city):
    config = SimConfig()
    a, b = Episode(city, Oracle(), config, 3), Episode(city, AlwaysBicycle(), config, 3)
    assert [s.obstacle for s in a.obstacles] == [s.obstacle for s in b.obstacles]


def test_rerouting_after_a_parked_car(city):
    config = SimConfig(n_obstacles=1)
    for episode in range(50):
        ep = Episode(city, Oracle(), config, episode)
        if ep.obstacles and ep.obstacles[0].obstacle.true_label == "car":
            result = ep.run()
            assert result.status == ARRIVED and result.replans >= 1
            assert ep.obstacles[0].obstacle.edge not in ep.car.driven_edges
            return
    pytest.skip("aucune voiture tirée sur 50 épisodes")


def test_encounters_csv(city, tmp_path):
    results = [Episode(city, Oracle(), SimConfig(), e).run() for e in range(3)]
    path = save_encounters_csv(results, tmp_path / "log.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("episode,step,classifier,true_label")
    assert len(lines) == 1 + sum(len(r.encounters) for r in results)
