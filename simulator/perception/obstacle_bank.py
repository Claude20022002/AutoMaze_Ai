"""Banque d'images d'obstacles : recadrages du jeu de test, ou images de remplacement.

Utilisation par l'équipe A (signatures proposées dans l'issue #1, point 1) :

    bank = ObstacleBank()
    path = bank.sample_image_path("car", rng)          # au placement de l'obstacle (Path ou None)
    obstacle = Obstacle(edge=..., true_label="car", image_path=path)
    image_rgb = bank.load_image(obstacle)              # quand la voiture arrive près de l'obstacle

Seul le jeu de **test** est utilisé (``data/crops/test/<classe>/``) : le modèle n'a jamais vu ces images.
Tant que les recadrages n'existent pas (jusqu'à mi-novembre), ``sample_image_path`` renvoie None et
``load_image`` génère une image de remplacement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simulator.config import CROPS_TEST_DIR
from simulator.contracts import CLASSES, Obstacle
from simulator.perception.images import load_rgb, placeholder_image

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


class ObstacleBank:
    """Index des images de test par classe. L'index est construit une fois, à la création."""

    def __init__(self, crops_dir: Path = CROPS_TEST_DIR) -> None:
        self.crops_dir = Path(crops_dir)
        self._paths: dict[str, tuple[Path, ...]] = {label: self._list_images(label) for label in CLASSES}

    def _list_images(self, label: str) -> tuple[Path, ...]:
        folder = self.crops_dir / label
        if not folder.is_dir():
            return ()
        # Tri par nom : le même tirage donne la même image sur toutes les machines.
        return tuple(sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))

    def count(self, label: str) -> int:
        """Nombre d'images de test disponibles pour une classe."""
        self._check_label(label)
        return len(self._paths[label])

    @property
    def has_real_images(self) -> bool:
        """True si chaque classe a au moins une image de test ; sinon, des remplacements seront utilisés."""
        return all(self._paths[label] for label in CLASSES)

    def sample_image_path(self, label: str, rng: np.random.Generator) -> Path | None:
        """Tire une image de test de la classe ``label`` ; None si la classe n'en a aucune.

        Le tirage ne dépend que de ``rng`` : avec la même graine, chaque classifieur voit les mêmes images.
        """
        self._check_label(label)
        paths = self._paths[label]
        if not paths:
            return None
        return paths[int(rng.integers(len(paths)))]

    def load_image(self, obstacle: Obstacle) -> np.ndarray:
        """Image de l'obstacle en uint8 RGB : le recadrage s'il existe, sinon une image de remplacement."""
        if obstacle.image_path is None:
            return placeholder_image(obstacle.true_label)
        return load_rgb(obstacle.image_path)

    @staticmethod
    def _check_label(label: str) -> None:
        if label not in CLASSES:
            raise ValueError(f"classe inconnue : {label!r} (attendu : {CLASSES})")
