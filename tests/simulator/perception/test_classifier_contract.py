"""Tests de contrat communs aux trois classifieurs (section 5 : « Terminé pour B »).

Chaque classifieur disponible est ajouté à ``FACTORIES`` ; tous passent exactement les mêmes tests.
``reference`` et ``ours`` y seront ajoutés quand leurs adaptateurs existeront.
"""

import numpy as np
import pytest

from simulator.contracts import CLASSES, VALID_LABELS, Classifier, Prediction
from simulator.perception.images import placeholder_image
from simulator.perception.simulated import SimulatedClassifier

FACTORIES = {
    "simulation": lambda: SimulatedClassifier(error_rate=0.3, seed=0),
}

# Tailles volontairement variées : chaque classifieur fait lui-même son redimensionnement.
SHAPES = [(32, 32, 3), (64, 48, 3), (240, 320, 3)]


@pytest.fixture(params=sorted(FACTORIES))
def classifier(request) -> Classifier:
    return FACTORIES[request.param]()


def test_is_classifier_with_name(classifier):
    assert isinstance(classifier, Classifier)
    assert isinstance(classifier.name, str) and classifier.name


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("true_label", CLASSES)
def test_prediction_respects_contract(classifier, shape, true_label):
    image = np.random.default_rng(1).integers(0, 256, size=shape, dtype=np.uint8)
    pred = classifier.predict(image, true_label=true_label)

    assert isinstance(pred, Prediction)
    assert pred.label in VALID_LABELS
    assert len(pred.probs) == len(CLASSES)
    assert all(isinstance(p, float) for p in pred.probs)
    assert sum(pred.probs) == pytest.approx(1.0, abs=1e-3)
    assert pred.score == pytest.approx(max(pred.probs))
    assert pred.infer_ms >= 0.0


def test_accepts_placeholder_images(classifier):
    for label in CLASSES:
        assert classifier.predict(placeholder_image(label), true_label=label).label in VALID_LABELS


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((32, 32), dtype=np.uint8),       # niveaux de gris, sans canal
        np.zeros((32, 32, 4), dtype=np.uint8),    # RGBA
        np.zeros((32, 32, 3), dtype=np.float32),  # mauvais type
        np.zeros((0, 32, 3), dtype=np.uint8),     # image vide
    ],
)
def test_rejects_invalid_images(classifier, bad_image):
    with pytest.raises(ValueError):
        classifier.predict(bad_image, true_label="car")
