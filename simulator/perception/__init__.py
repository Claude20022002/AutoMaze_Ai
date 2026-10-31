"""Perception (équipe B) : classifieurs interchangeables et images des obstacles.

Point d'entrée pour l'équipe A (``engine.py``, ``main.py``), section 3 du plan :
- ``make_classifier(name, config)`` : le classifieur choisi au lancement ;
- ``make_image_provider(config)`` : la fonction qui donne l'image RGB de chaque obstacle.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from simulator.config import SimConfig
from simulator.contracts import CLASSES, Classifier, Obstacle

CLASSIFIER_MODES: tuple[str, ...] = ("simulation", "reference", "ours")

ImageProvider = Callable[[Obstacle], np.ndarray]


def make_classifier(name: str, config: SimConfig) -> Classifier:
    """Crée le classifieur demandé au lancement (``--classifier simulation|reference|ours``)."""
    if name == "simulation":
        from simulator.perception.simulated import SimulatedClassifier

        return SimulatedClassifier(
            error_rate=config.simulated_error_rate,
            seed=config.seed,
            threshold=config.confidence_threshold,
        )
    if name in ("reference", "ours"):
        # Pas de modèle de secours (section 9, règle 3) : on refuse plutôt que d'inventer des prédictions.
        raise NotImplementedError(
            f"le classifieur {name!r} n'est pas encore disponible ; utilisez --classifier simulation en attendant"
        )
    raise ValueError(f"classifieur inconnu : {name!r} (attendu : {CLASSIFIER_MODES})")


def make_image_provider(config: SimConfig, crops_dir: Path | None = None) -> ImageProvider:
    """Renvoie une fonction ``obstacle -> image RGB uint8 (H, W, 3)``.

    - ``obstacle.image_path`` renseigné : cette image ;
    - sinon : une image de ``data/crops/test/<vraie classe>/`` tirée à partir de la graine et de l'obstacle,
      donc la même pour tous les classifieurs et à chaque passage ;
    - sans recadrage pour cette classe : une image de remplacement, propre à l'obstacle (léger bruit).
    """
    from simulator.perception.images import placeholder_image
    from simulator.perception.obstacle_bank import ObstacleBank

    bank = ObstacleBank() if crops_dir is None else ObstacleBank(crops_dir)
    cache: dict[Obstacle, np.ndarray] = {}

    def image_for(obstacle: Obstacle) -> np.ndarray:
        if obstacle not in cache:
            path = obstacle.image_path
            seed = _obstacle_seed(config.seed, obstacle)
            if path is None:
                path = bank.sample_image_path(obstacle.true_label, np.random.default_rng(seed))
            if path is None:
                cache[obstacle] = placeholder_image(obstacle.true_label, variant=seed)
            else:
                cache[obstacle] = bank.load_image(Obstacle(obstacle.edge, obstacle.true_label, path, obstacle.offset_m))
        return cache[obstacle]

    return image_for


def _obstacle_seed(seed: int, obstacle: Obstacle) -> list[int]:
    """Graine tirée de l'obstacle lui-même (et non de l'ordre des appels) : même obstacle = même image."""
    u, v, key = obstacle.edge
    return [seed, u, v, key, CLASSES.index(obstacle.true_label), round(obstacle.offset_m * 1000)]
