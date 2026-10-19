"""Tests des contrats partagés : ils protègent l'interface entre les équipes A et B."""

import json

import numpy as np
import pytest

from simulator.config import DECISION_RULES, SimConfig, check_metadata_classes
from simulator.contracts import (
    CLASSES,
    UNKNOWN,
    Action,
    Classifier,
    EncounterLog,
    Obstacle,
    Prediction,
)

UNIFORM = tuple([1 / len(CLASSES)] * len(CLASSES))


def test_classes_order_matches_notebooks():
    assert CLASSES == ("person", "bicycle", "car", "motorcycle", "bus", "truck")


def test_prediction_valid():
    p = Prediction(label="car", score=0.9, probs=(0.02, 0.02, 0.9, 0.02, 0.02, 0.02), infer_ms=3.1)
    assert p.label_fr == "Voiture"


def test_prediction_unknown_label_fr():
    assert Prediction(label=UNKNOWN, score=0.2, probs=UNIFORM, infer_ms=1.0).label_fr == "Inconnu"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": "tram"},                          # classe inexistante
        {"probs": (0.5, 0.5)},                      # mauvais nombre de probabilités
        {"probs": (0.5, 0.5, 0.5, 0.0, 0.0, 0.0)},  # somme différente de 1
        {"score": 1.5},
        {"infer_ms": -1.0},
    ],
)
def test_prediction_rejects_invalid(kwargs):
    base = {"label": "car", "score": 0.5, "probs": UNIFORM, "infer_ms": 1.0}
    with pytest.raises(ValueError):
        Prediction(**{**base, **kwargs})


def test_obstacle_requires_trained_class():
    Obstacle(edge=(1, 2, 0), true_label="bus", image_path=None)
    with pytest.raises(ValueError):
        Obstacle(edge=(1, 2, 0), true_label=UNKNOWN, image_path=None)


def test_classifier_is_abstract():
    with pytest.raises(TypeError):
        Classifier()


def test_minimal_classifier_honours_contract():
    class AlwaysCar(Classifier):
        name = "test"

        def predict(self, image_rgb, true_label=None):
            return Prediction(label="car", score=1.0, probs=(0, 0, 1, 0, 0, 0), infer_ms=0.0)

    pred = AlwaysCar().predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert pred.label in CLASSES


def test_decision_rules_cover_every_label():
    assert set(DECISION_RULES) == set(CLASSES) | {UNKNOWN}
    assert all(isinstance(a, Action) for a in DECISION_RULES.values())
    assert SimConfig().decision_rules == DECISION_RULES


def test_encounter_log_severe_error():
    log = EncounterLog(0, 10, "test", "person", "bicycle", 0.7, Action.SLOW_DOWN.name, 2.0)
    assert not log.correct and log.severe_error


def test_check_metadata_classes(tmp_path):
    good = tmp_path / "metadata.json"
    good.write_text(json.dumps({"classes": list(CLASSES)}), encoding="utf-8")
    check_metadata_classes(good)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"classes": sorted(CLASSES)}), encoding="utf-8")  # ordre alphabétique d'ImageFolder
    with pytest.raises(ValueError):
        check_metadata_classes(bad)
