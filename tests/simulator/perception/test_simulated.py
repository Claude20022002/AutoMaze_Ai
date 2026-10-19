"""Tests du mode SIMULATION."""

import numpy as np
import pytest

from simulator.config import SimConfig
from simulator.contracts import CLASSES, UNKNOWN
from simulator.perception import CLASSIFIER_MODES, build_classifier
from simulator.perception.simulated import SIMULATION_NAME, SimulatedClassifier

IMAGE = np.zeros((32, 32, 3), dtype=np.uint8)


def _labels(clf: SimulatedClassifier, n: int, true_label: str = "car") -> list[str]:
    return [clf.predict(IMAGE, true_label=true_label).label for _ in range(n)]


def test_displayed_name_says_no_real_model():
    assert SimulatedClassifier().name == SIMULATION_NAME == "SIMULATION — modèle réel non chargé"


def test_no_error_returns_true_label():
    clf = SimulatedClassifier(error_rate=0.0)
    for label in CLASSES:
        assert clf.predict(IMAGE, true_label=label).label == label


def test_full_error_never_returns_true_label():
    labels = _labels(SimulatedClassifier(error_rate=1.0, seed=3), 200)
    assert "car" not in labels
    assert set(labels) == set(CLASSES) - {"car"}  # les erreurs se répartissent sur les 5 autres classes


def test_error_rate_is_respected():
    labels = _labels(SimulatedClassifier(error_rate=0.2, seed=42), 5000)
    observed = sum(label != "car" for label in labels) / len(labels)
    assert observed == pytest.approx(0.2, abs=0.02)


def test_same_seed_same_predictions_and_reset():
    a = SimulatedClassifier(error_rate=0.5, seed=7)
    b = SimulatedClassifier(error_rate=0.5, seed=7)
    first = _labels(a, 50)
    assert first == _labels(b, 50)
    a.reset(7)
    assert _labels(a, 50) == first


def test_different_seeds_differ():
    assert _labels(SimulatedClassifier(error_rate=0.5, seed=1), 50) != _labels(
        SimulatedClassifier(error_rate=0.5, seed=2), 50
    )


def test_probs_follow_chosen_label():
    pred = SimulatedClassifier(error_rate=0.0, confidence=0.8).predict(IMAGE, true_label="bus")
    assert pred.score == pytest.approx(0.8)
    assert pred.probs[CLASSES.index("bus")] == pytest.approx(0.8)


def test_low_confidence_gives_unknown():
    pred = SimulatedClassifier(error_rate=0.0, confidence=0.4, threshold=0.5).predict(IMAGE, true_label="person")
    assert pred.label == UNKNOWN
    assert pred.score == pytest.approx(0.4)


def test_requires_true_label():
    with pytest.raises(ValueError):
        SimulatedClassifier().predict(IMAGE)
    with pytest.raises(ValueError):
        SimulatedClassifier().predict(IMAGE, true_label=UNKNOWN)


@pytest.mark.parametrize("kwargs", [{"error_rate": -0.1}, {"error_rate": 1.5}, {"confidence": 1 / 6}, {"threshold": 2}])
def test_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SimulatedClassifier(**kwargs)


def test_build_classifier_uses_config():
    clf = build_classifier("simulation", SimConfig(seed=11, simulated_error_rate=0.0))
    assert isinstance(clf, SimulatedClassifier)
    assert clf.seed == 11 and clf.error_rate == 0.0


def test_build_classifier_modes():
    assert CLASSIFIER_MODES == ("simulation", "reference", "ours")
    for mode in ("reference", "ours"):
        with pytest.raises(NotImplementedError):  # aucun modèle de secours
            build_classifier(mode)
    with pytest.raises(ValueError):
        build_classifier("yolo")
