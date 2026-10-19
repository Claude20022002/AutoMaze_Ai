"""Contrats partagés entre l'équipe A (monde) et l'équipe B (perception).

Ce fichier est la seule dépendance entre les deux équipes. Il est figé le 21 octobre 2026 :
toute modification passe par une PR approuvée par les deux équipes
(voir docs/simulateur/PLAN_SIMULATEUR.md, section 3).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

# Ordre = indices de sortie des modèles. Doit être identique à CLASSES dans les notebooks
# et à "classes" dans models/metadata.json (vérifié par config.check_metadata_classes).
CLASSES: tuple[str, ...] = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
CLASSES_FR: dict[str, str] = {
    "person": "Personne",
    "bicycle": "Vélo",
    "car": "Voiture",
    "motorcycle": "Moto",
    "bus": "Bus",
    "truck": "Camion",
}

# Affiché quand le score est sous le seuil de confiance. Ce n'est jamais une classe entraînée.
UNKNOWN = "unknown"
VALID_LABELS: frozenset[str] = frozenset(CLASSES) | {UNKNOWN}

# Tolérance sur la somme des probabilités (arrondis du softmax en float32).
_PROBS_SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Prediction:
    """Résultat d'un classifieur pour une image d'obstacle."""

    label: str                 # une valeur de CLASSES, ou UNKNOWN
    score: float               # score softmax maximal (pas une probabilité calibrée)
    probs: tuple[float, ...]   # len(CLASSES) valeurs, dans l'ordre de CLASSES
    infer_ms: float            # temps d'inférence réellement mesuré, en millisecondes

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(f"label inconnu : {self.label!r} (attendu : {sorted(VALID_LABELS)})")
        if len(self.probs) != len(CLASSES):
            raise ValueError(f"probs doit contenir {len(CLASSES)} valeurs, reçu {len(self.probs)}")
        if any(not 0.0 <= p <= 1.0 for p in self.probs):
            raise ValueError("chaque probabilité doit être comprise entre 0 et 1")
        if not math.isclose(sum(self.probs), 1.0, abs_tol=_PROBS_SUM_TOLERANCE):
            raise ValueError(f"la somme des probs doit valoir 1, reçu {sum(self.probs):.4f}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score doit être compris entre 0 et 1")
        if self.infer_ms < 0:
            raise ValueError("infer_ms ne peut pas être négatif")

    @property
    def label_fr(self) -> str:
        """Nom affiché à l'écran."""
        return CLASSES_FR.get(self.label, "Inconnu")


class Classifier(ABC):
    """Interface commune aux trois modes : simulation, référence préentraînée, notre CNN."""

    #: Affiché dans l'interface : "SIMULATION — modèle réel non chargé", "Référence préentraînée", "Notre CNN".
    name: str

    @abstractmethod
    def predict(self, image_rgb: np.ndarray, true_label: str | None = None) -> Prediction:
        """Classe une image d'obstacle.

        image_rgb : tableau uint8 de forme (H, W, 3), canaux dans l'ordre RGB, taille quelconque.
        Chaque classifieur applique lui-même son prétraitement (taille, normalisation).
        true_label : vraie classe, utilisée uniquement par le mode simulation ; les vrais modèles l'ignorent.
        """


@dataclass(frozen=True)
class Obstacle:
    """Un obstacle placé dans la grille. Le simulateur connaît sa position et sa vraie classe."""

    cell: tuple[int, int]      # (ligne, colonne) dans la grille
    true_label: str            # vraie classe, une valeur de CLASSES
    image_path: Path | None    # image du jeu de test ; None = image de remplacement générée

    def __post_init__(self) -> None:
        if self.true_label not in CLASSES:
            raise ValueError(f"true_label doit être une classe entraînée, reçu {self.true_label!r}")


class Action(Enum):
    """Réaction de la voiture à une classe prédite."""

    STOP_WAIT = "s'arrêter et attendre"
    SLOW_DOWN = "ralentir"
    REROUTE = "contourner"
    REROUTE_WIDE = "contourner largement"
    CAUTIOUS_STOP = "arrêt prudent"


@dataclass(frozen=True)
class EncounterLog:
    """Une ligne par obstacle rencontré. Écrite en CSV par l'équipe A, analysée par l'équipe B."""

    episode: int
    step: int
    classifier: str
    true_label: str
    predicted_label: str
    score: float
    action: str                # Action.name, par exemple "REROUTE"
    infer_ms: float

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.true_label

    @property
    def severe_error(self) -> bool:
        """Faute grave : une personne qui n'a pas été reconnue comme telle."""
        return self.true_label == "person" and self.predicted_label != "person"
