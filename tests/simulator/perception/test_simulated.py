"""Tests du mode SIMULATION."""

import numpy as np
import pytest

from simulator.contracts import CLASSES, UNKNOWN
from simulator.perception.simulated import SIMULATION_NAME, SimulatedClassifier

IMAGE = np.zeros((32, 32, 3), dtype=np.uint8)


def _images(n: int, seed: int = 0) -> list[np.ndarray]:
    """n petites images toutes différentes (comme n obstacles distincts)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8) for _ in range(n)]


def _labels(clf: SimulatedClassifier, images: list[np.ndarray], true_label: str = "car") -> list[str]:
    return [clf.predict(image, true_label=true_label).label for image in images]


def test_displayed_name_says_no_real_model():
    assert SimulatedClassifier().name == SIMULATION_NAME == "SIMULATION — modèle réel non chargé"


def test_no_error_returns_true_label():
    clf = SimulatedClassifier(error_rate=0.0)
    for label in CLASSES:
        assert clf.predict(IMAGE, true_label=label).label == label


def test_full_error_never_returns_true_label():
    labels = _labels(SimulatedClassifier(error_rate=1.0, seed=3), _images(200))
    assert "car" not in labels
    assert set(labels) == set(CLASSES) - {"car"}  # les erreurs se répartissent sur les 5 autres classes


def test_error_rate_is_respected():
    labels = _labels(SimulatedClassifier(error_rate=0.2, seed=42), _images(5000))
    observed = sum(label != "car" for label in labels) / len(labels)
    assert observed == pytest.approx(0.2, abs=0.02)


def test_same_image_same_answer_whatever_the_order():
    """Le tirage dépend de la graine et de l'image, pas de l'ordre des appels (épisode précédent, autre classifieur)."""
    images = _images(50)
    a = SimulatedClassifier(error_rate=0.5, seed=7)
    b = SimulatedClassifier(error_rate=0.5, seed=7)
    forward = _labels(a, images)
    _labels(b, _images(30, seed=99))                 # b a déjà classé d'autres images
    backward = _labels(b, images[::-1])[::-1]        # puis les mêmes, dans l'ordre inverse
    assert forward == backward
    assert _labels(a, images) == forward             # et a, interrogé une seconde fois, ne change pas d'avis


def test_different_seeds_differ():
    images = _images(50)
    assert _labels(SimulatedClassifier(error_rate=0.5, seed=1), images) != _labels(
        SimulatedClassifier(error_rate=0.5, seed=2), images
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

