"""Outils d'images partagés par la perception : vérification, lecture RGB, images de remplacement.

Les images sont toujours manipulées en ``np.ndarray`` uint8 de forme (H, W, 3), canaux dans l'ordre RGB,
comme l'exige ``Classifier.predict``. On lit avec Pillow (RGB) et jamais avec OpenCV (BGR) : une inversion
des canaux dégrade le modèle sans aucun message d'erreur (piège de la section 5 du plan).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulator.contracts import CLASSES, CLASSES_FR

PLACEHOLDER_SIZE = 128  # côté des images de remplacement, en pixels

# Une couleur de fond par classe, bien distinctes, pour reconnaître l'image de remplacement d'un coup d'œil.
PLACEHOLDER_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (214, 39, 40),
    "bicycle": (44, 160, 44),
    "car": (31, 119, 180),
    "motorcycle": (255, 127, 14),
    "bus": (148, 103, 189),
    "truck": (140, 86, 75),
}


def check_image_rgb(image_rgb: np.ndarray) -> None:
    """Lève ValueError si l'image ne respecte pas le contrat : uint8, forme (H, W, 3)."""
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError(f"image attendue en np.ndarray, reçu {type(image_rgb).__name__}")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"image attendue en uint8, reçu {image_rgb.dtype}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
        raise ValueError(f"image attendue de forme (H, W, 3), reçu {image_rgb.shape}")


def load_rgb(path: Path) -> np.ndarray:
    """Lit une image sur le disque et la renvoie en uint8 RGB (les images en niveaux de gris ou RGBA sont converties)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def placeholder_image(label: str, size: int = PLACEHOLDER_SIZE) -> np.ndarray:
    """Image de remplacement : fond de la couleur de la classe, nom de la classe et mention « remplacement ».

    Ce n'est pas une donnée du jeu de test : la mention est écrite sur l'image pour qu'on ne la prenne
    jamais pour un vrai recadrage dans une démo ou une capture d'écran.
    """
    if label not in CLASSES:
        raise ValueError(f"classe inconnue : {label!r}")
    img = Image.new("RGB", (size, size), PLACEHOLDER_COLORS[label])
    draw = ImageDraw.Draw(img)
    _draw_centered(draw, CLASSES_FR[label], size, y=size * 0.42, font_size=max(10, size // 7))
    _draw_centered(draw, "remplacement", size, y=size * 0.70, font_size=max(8, size // 12))
    return np.asarray(img, dtype=np.uint8).copy()


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, size: int, y: float, font_size: int) -> None:
    font = ImageFont.load_default(size=font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = (size - (right - left)) / 2 - left
    draw.text((x, y - (bottom - top) / 2 - top), text, fill=(255, 255, 255), font=font)
