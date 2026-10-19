"""Tests du seuil « Inconnu »."""

import pytest

from simulator.contracts import CLASSES, UNKNOWN
from simulator.perception.threshold import apply_threshold, make_prediction, threshold_from_metadata

CAR_70 = (0.06, 0.06, 0.70, 0.06, 0.06, 0.06)


def test_above_threshold_keeps_best_class():
    assert apply_threshold(CAR_70, 0.5) == ("car", pytest.approx(0.70))


def test_below_threshold_gives_unknown_and_keeps_score():
    assert apply_threshold(CAR_70, 0.8) == (UNKNOWN, pytest.approx(0.70))


def test_threshold_is_inclusive():
    assert apply_threshold(CAR_70, 0.70)[0] == "car"


def test_threshold_zero_never_unknown():
    uniform = tuple([1 / len(CLASSES)] * len(CLASSES))
    assert apply_threshold(uniform, 0.0)[0] in CLASSES


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        apply_threshold(CAR_70, threshold)


def test_wrong_number_of_probs():
    with pytest.raises(ValueError):
        apply_threshold((0.5, 0.5), 0.5)


def test_threshold_from_metadata():
    assert threshold_from_metadata({"confidence_threshold": 0.65}, default=0.5) == 0.65
    assert threshold_from_metadata({}, default=0.5) == 0.5
    assert threshold_from_metadata({"confidence_threshold": None}, default=0.5) == 0.5
    with pytest.raises(ValueError):
        threshold_from_metadata({"confidence_threshold": 3}, default=0.5)


def test_make_prediction_converts_to_float():
    import numpy as np

    pred = make_prediction(np.array(CAR_70, dtype=np.float32), threshold=0.5, infer_ms=np.float32(2.5))
    assert pred.label == "car"
    assert all(type(p) is float for p in pred.probs)
    assert type(pred.score) is float and type(pred.infer_ms) is float
