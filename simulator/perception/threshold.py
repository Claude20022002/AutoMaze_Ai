"""Seuil « Inconnu » : sous un score de confiance minimal, le label devient UNKNOWN.

Les trois classifieurs passent par ``make_prediction`` : la règle du seuil est donc la même partout,
et la ``Prediction`` est validée par le contrat au même endroit.

Le seuil vient des métadonnées du modèle (clé ``confidence_threshold``) quand elle existe,
sinon de ``SimConfig.confidence_threshold`` (voir l'issue #1, point 8).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from simulator.contracts import CLASSES, UNKNOWN, Prediction

METADATA_KEY = "confidence_threshold"


def check_threshold(threshold: float) -> float:
    """Vérifie qu'un seuil est compris entre 0 et 1 et le renvoie en ``float``."""
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"le seuil doit être compris entre 0 et 1, reçu {threshold}")
    return threshold


def threshold_from_metadata(metadata: Mapping[str, Any], default: float) -> float:
    """Seuil lu dans les métadonnées d'un modèle ; ``default`` si la clé est absente ou vaut None."""
    value = metadata.get(METADATA_KEY)
    return check_threshold(default if value is None else value)


def apply_threshold(probs: Sequence[float], threshold: float) -> tuple[str, float]:
    """Renvoie (label, score). Le label est la classe de score maximal, ou UNKNOWN sous le seuil.

    Le score renvoyé reste le score softmax maximal, même quand le label devient UNKNOWN :
    la vue caméra et les journaux montrent ainsi à quel point le modèle hésitait.
    """
    if len(probs) != len(CLASSES):
        raise ValueError(f"probs doit contenir {len(CLASSES)} valeurs, reçu {len(probs)}")
    best = max(range(len(CLASSES)), key=lambda i: probs[i])
    score = float(probs[best])
    label = CLASSES[best] if score >= check_threshold(threshold) else UNKNOWN
    return label, score


def make_prediction(probs: Sequence[float], threshold: float, infer_ms: float) -> Prediction:
    """Construit une ``Prediction`` validée par le contrat à partir des probabilités d'un classifieur."""
    probs_tuple = tuple(float(p) for p in probs)
    label, score = apply_threshold(probs_tuple, threshold)
    return Prediction(label=label, score=score, probs=probs_tuple, infer_ms=float(infer_ms))
