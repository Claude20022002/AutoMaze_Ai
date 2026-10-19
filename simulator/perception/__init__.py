"""Perception (équipe B) : classifieurs interchangeables et banque d'images d'obstacles.

Point d'entrée pour l'équipe A (``main.py``, ``engine.py``) : ``build_classifier(mode, config)``.
"""

from __future__ import annotations

from simulator.config import SimConfig
from simulator.contracts import Classifier

CLASSIFIER_MODES: tuple[str, ...] = ("simulation", "reference", "ours")


def build_classifier(mode: str, config: SimConfig | None = None) -> Classifier:
    """Crée le classifieur demandé au lancement (``--classifier simulation|reference|ours``)."""
    config = config or SimConfig()
    if mode == "simulation":
        from simulator.perception.simulated import SimulatedClassifier

        return SimulatedClassifier(
            error_rate=config.simulated_error_rate,
            seed=config.seed,
            threshold=config.confidence_threshold,
        )
    if mode in ("reference", "ours"):
        # Pas de modèle de secours (section 9, règle 3) : on refuse plutôt que d'inventer des prédictions.
        raise NotImplementedError(
            f"le mode {mode!r} n'est pas encore disponible ; utilisez --classifier simulation en attendant"
        )
    raise ValueError(f"mode de classifieur inconnu : {mode!r} (attendu : {CLASSIFIER_MODES})")
