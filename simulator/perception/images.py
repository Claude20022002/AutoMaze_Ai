"""Outils d'images partagés par la perception : vérification, lecture RGB, images de remplacement.

Les images sont toujours manipulées en ``np.ndarray`` uint8 de forme (H, W, 3), canaux dans l'ordre RGB,
comme l'exige ``Classifier.predict``. On lit avec Pillow (RGB) et jamais avec OpenCV (BGR) : une inversion
des canaux dégrade le modèle sans aucun message d'erreur (piège de la section 5 du plan).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulator.contracts import CLASSES, CLASSES_FR

PLACEHOLDER_SIZE = 128  # côté des images de remplacement, en pixels
PLACEHOLDER_NOISE = 3   # amplitude du bruit qui distingue deux images de remplacement de la même classe

# Une couleur de fond par classe : les mêmes que theme.CLASS_COLORS (carte, journal, vue caméra).
# Recopiées ici pour que la perception n'importe pas pygame ; test_camera_panel vérifie qu'elles concordent.
PLACEHOLDER_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (214, 72, 64),
    "bicycle": (46, 140, 74),
    "car": (37, 99, 200),
    "motorcycle": (140, 82, 190),
    "bus": (222, 134, 26),
    "truck": (120, 90, 60),
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


def placeholder_image(label: str, size: int = PLACEHOLDER_SIZE,
                      variant: int | Sequence[int] | None = None) -> np.ndarray:
    """Image de remplacement : fond de la couleur de la classe, nom de la classe et mention « remplacement ».

    Ce n'est pas une donnée du jeu de test : la mention est écrite sur l'image pour qu'on ne la prenne
    jamais pour un vrai recadrage dans une démo ou une capture d'écran.

    variant : graine d'un léger bruit (± PLACEHOLDER_NOISE niveaux, invisible à l'œil). Deux obstacles de la même
        classe ont ainsi deux images différentes, comme deux vraies photos : sinon, un classifieur dont la réponse
        dépend de l'image (mode simulation compris) donnerait la même réponse pour tous les obstacles d'une classe.
    """
    if label not in CLASSES:
        raise ValueError(f"classe inconnue : {label!r}")
    img = Image.new("RGB", (size, size), PLACEHOLDER_COLORS[label])
    draw = ImageDraw.Draw(img)
    _draw_centered(draw, CLASSES_FR[label], size, y=size * 0.42, font_size=max(10, size // 7))
    _draw_centered(draw, "remplacement", size, y=size * 0.70, font_size=max(8, size // 12))
    pixels = np.asarray(img, dtype=np.uint8).copy()
    if variant is None:
        return pixels
    noise = np.random.default_rng(variant).integers(-PLACEHOLDER_NOISE, PLACEHOLDER_NOISE + 1, size=pixels.shape)
    return np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _accented_font_path() -> str | None:
    """Police DejaVu Sans livrée avec matplotlib : la police par défaut de Pillow n'a pas les accents (« Vélo »)."""
    try:
        from matplotlib import font_manager

        return font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    except (ImportError, ValueError):
        return None


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, size: int, y: float, font_size: int) -> None:
    font_path = _accented_font_path()
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:  # sans matplotlib : police par défaut, texte sans accents
        font = ImageFont.load_default(size=font_size)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = (size - (right - left)) / 2 - left
    draw.text((x, y - (bottom - top) / 2 - top), text, fill=(255, 255, 255), font=font)
