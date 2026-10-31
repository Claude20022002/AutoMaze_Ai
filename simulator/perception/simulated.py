"""Mode SIMULATION : renvoie la vraie classe, sauf dans une part réglable des cas.

Ce mode ne charge aucun modèle. Il sert à tester la logique du simulateur (décisions, recalculs, journaux)
avant que notre CNN existe. Ses probabilités ne sont pas des sorties de modèle : la classe choisie reçoit
``confidence`` et les autres se partagent le reste. Le nom affiché le dit explicitement (section 9, règle 4).
"""

from __future__ import annotations

import time
import zlib

import numpy as np

from simulator.config import SimConfig
from simulator.contracts import CLASSES, Classifier, Prediction
from simulator.perception.images import check_image_rgb
from simulator.perception.threshold import check_threshold, make_prediction

SIMULATION_NAME = "SIMULATION — modèle réel non chargé"

_DEFAULTS = SimConfig()


class SimulatedClassifier(Classifier):
    """Classifieur sans modèle : sa réponse ne dépend que de la graine, de l'image et de la vraie classe."""

    name = SIMULATION_NAME

    def __init__(
        self,
        error_rate: float = _DEFAULTS.simulated_error_rate,
        seed: int = _DEFAULTS.seed,
        confidence: float = 0.9,
        threshold: float = _DEFAULTS.confidence_threshold,
    ) -> None:
        """
        error_rate : part des prédictions volontairement fausses (0 = toujours juste, 1 = toujours faux).
        confidence : score donné à la classe choisie ; sous ``threshold``, le label devient UNKNOWN,
            ce qui permet aussi de tester l'arrêt prudent sans modèle.
        """
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"error_rate doit être compris entre 0 et 1, reçu {error_rate}")
        # Strictement au-dessus de 1/6 : sinon la classe choisie ne serait plus celle de score maximal.
        if not 1 / len(CLASSES) < confidence <= 1.0:
            raise ValueError(f"confidence doit être dans ]1/{len(CLASSES)}, 1], reçu {confidence}")
        self.error_rate = float(error_rate)
        self.confidence = float(confidence)
        self.threshold = check_threshold(threshold)
        self.seed = seed

    def predict(self, image_rgb: np.ndarray, true_label: str | None = None) -> Prediction:
        start = time.perf_counter()
        check_image_rgb(image_rgb)
        if true_label not in CLASSES:
            raise ValueError(f"le mode simulation a besoin de la vraie classe, reçu {true_label!r}")

        # Tirage calculé à partir de la graine et de l'image, et non d'un générateur partagé :
        # la même image donne toujours la même réponse, quel que soit l'ordre des appels
        # (épisode précédent, autre classifieur comparé), comme le ferait un vrai modèle.
        rng = np.random.default_rng(_image_seed(self.seed, image_rgb, true_label))
        chosen = true_label
        if rng.random() < self.error_rate:
            others = [c for c in CLASSES if c != true_label]
            chosen = others[rng.integers(len(others))]

        rest = (1.0 - self.confidence) / (len(CLASSES) - 1)
        probs = [self.confidence if c == chosen else rest for c in CLASSES]
        infer_ms = (time.perf_counter() - start) * 1000.0
        return make_prediction(probs, self.threshold, infer_ms)


def _image_seed(seed: int, image_rgb: np.ndarray, true_label: str) -> list[int]:
    """Graine tirée de l'image elle-même : même graine + même image = même tirage."""
    h, w = image_rgb.shape[:2]
    return [seed, zlib.crc32(np.ascontiguousarray(image_rgb).tobytes()), h, w, CLASSES.index(true_label)]
