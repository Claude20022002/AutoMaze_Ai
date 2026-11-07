"""Textures procédurales de la ville (Pillow) : façades par étage, vitrines, bitume, pavés, sol.

Générées par le code, donc sans licence à gérer et identiques sur toutes les machines (graines fixes).
Les façades sont en niveaux clairs : la couleur de chaque bâtiment vient de ses sommets et multiplie la texture.
Un motif de façade couvre UN étage et une travée : il se répète sur toute la hauteur et toute la longueur des murs
(technique classique des villes procédurales, pour peu de mémoire).

Repères de répétition (mètres couverts par une répétition de la texture) : ``FACADE_BAY_M`` × ``FLOOR_M``
pour les étages, ``SHOP_BAY_M`` × ``GROUND_FLOOR_M`` pour le rez-de-chaussée.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FLOOR_M = 3.2
FACADE_BAY_M = 4.0
GROUND_FLOOR_M = 4.2
SHOP_BAY_M = 6.0
GROUND_TILE_M = 8.0          # sol, chaussée et trottoirs : une répétition tous les 8 m

FACADE_STYLES = ("art_deco", "moderne", "residentiel")
SIZE = 256

_GLASS = (58, 72, 88)
_FRAME = (238, 236, 230)
_SHUTTER = (120, 128, 132)
_AWNINGS = ((176, 46, 40), (32, 110, 76), (34, 86, 150), (196, 128, 32))


def _wall(rng: np.random.Generator, size: int = SIZE, base: int = 246) -> Image.Image:
    """Enduit clair légèrement irrégulier (façades « passées » du centre-ville)."""
    noise = rng.normal(0, 4.0, (size, size)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([img] * 3, axis=-1)).filter(ImageFilter.GaussianBlur(0.6))


@lru_cache(maxsize=None)
def facade(style: str) -> np.ndarray:
    """Un étage et une travée de façade, RGB uint8 (SIZE, SIZE, 3). La ligne 0 est le HAUT de l'étage."""
    if style not in FACADE_STYLES:
        raise ValueError(f"style inconnu : {style!r} (attendu : {FACADE_STYLES})")
    rng = np.random.default_rng(FACADE_STYLES.index(style) + 11)
    img = _wall(rng)
    d = ImageDraw.Draw(img)
    s = SIZE
    if style == "art_deco":
        # Haute fenêtre à la française, balcon filant, bandeau horizontal en haut d'étage
        d.rectangle([0, 0, s, int(s * 0.06)], fill=(226, 222, 212))
        d.rectangle([int(s * 0.30), int(s * 0.16), int(s * 0.70), int(s * 0.86)], fill=_FRAME)
        d.rectangle([int(s * 0.34), int(s * 0.20), int(s * 0.66), int(s * 0.84)], fill=_GLASS)
        d.line([int(s * 0.50), int(s * 0.20), int(s * 0.50), int(s * 0.84)], fill=_FRAME, width=4)
        d.rectangle([int(s * 0.24), int(s * 0.80), int(s * 0.76), int(s * 0.88)], fill=(90, 92, 96))   # balcon
        for x in range(int(s * 0.26), int(s * 0.76), 10):
            d.line([x, int(s * 0.70), x, int(s * 0.80)], fill=(90, 92, 96), width=2)
    elif style == "moderne":
        # Bandeau vitré continu, allège pleine
        d.rectangle([0, int(s * 0.18), s, int(s * 0.70)], fill=_GLASS)
        for x in range(0, s, s // 4):
            d.rectangle([x, int(s * 0.18), x + 5, int(s * 0.70)], fill=(200, 202, 204))
        d.rectangle([0, int(s * 0.70), s, int(s * 0.74)], fill=(214, 214, 210))
    else:
        # Petites fenêtres à volets, comme les immeubles d'habitation
        for cx in (0.27, 0.73):
            x0, x1 = int(s * (cx - 0.13)), int(s * (cx + 0.13))
            d.rectangle([x0, int(s * 0.28), x1, int(s * 0.76)], fill=_GLASS)
            d.rectangle([x0 - 22, int(s * 0.28), x0, int(s * 0.76)], fill=_SHUTTER)
            d.rectangle([x1, int(s * 0.28), x1 + 22, int(s * 0.76)], fill=_SHUTTER)
            d.rectangle([x0 - 4, int(s * 0.76), x1 + 4, int(s * 0.80)], fill=(222, 220, 214))
    return np.asarray(img, dtype=np.uint8).copy()


@lru_cache(maxsize=None)
def shopfront(variant: int) -> np.ndarray:
    """Rez-de-chaussée commercial : vitrine, porte et store coloré. La ligne 0 est le HAUT du rez-de-chaussée."""
    rng = np.random.default_rng(100 + variant)
    img = _wall(rng, base=236)
    d = ImageDraw.Draw(img)
    s = SIZE
    awning = _AWNINGS[variant % len(_AWNINGS)]
    d.rectangle([int(s * 0.06), int(s * 0.30), int(s * 0.94), s], fill=(70, 74, 80))          # encadrement
    d.rectangle([int(s * 0.10), int(s * 0.36), int(s * 0.62), int(s * 0.92)], fill=_GLASS)     # vitrine
    d.rectangle([int(s * 0.68), int(s * 0.40), int(s * 0.90), s], fill=(96, 70, 52))           # porte
    for i, x in enumerate(range(int(s * 0.04), int(s * 0.96), 16)):                          # store rayé
        d.rectangle([x, int(s * 0.14), x + 15, int(s * 0.30)], fill=awning if i % 2 == 0 else (240, 236, 226))
    return np.asarray(img, dtype=np.uint8).copy()


def _noise_tile(seed: int, base: tuple[int, int, int], amplitude: float, blur: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amplitude, (SIZE, SIZE)).astype(np.float32)
    noise = np.asarray(Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)),
                       dtype=np.float32) - 128
    return np.clip(np.asarray(base, np.float32) + noise[..., None], 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def asphalt() -> np.ndarray:
    """Bitume clair : couleur multipliée par celle des sommets de la chaussée."""
    return _noise_tile(7, (232, 232, 232), 18.0, 0.8)


@lru_cache(maxsize=None)
def pavers() -> np.ndarray:
    """Trottoir en dalles de 50 cm (16 dalles par répétition de 8 m)."""
    img = _noise_tile(8, (238, 236, 230), 6.0, 0.6)
    step = SIZE // 16
    img[::step, :] = (img[::step, :] * 0.82).astype(np.uint8)
    img[:, ::step] = (img[:, ::step] * 0.82).astype(np.uint8)
    return img


@lru_cache(maxsize=None)
def ground() -> np.ndarray:
    """Sol entre les rues : terre et poussière claires."""
    return _noise_tile(9, (236, 232, 222), 10.0, 2.5)


def panda_texture(image_rgb: np.ndarray, name: str, repeat: bool = True):
    """Texture Panda3D avec mipmaps (pas de scintillement des fenêtres au loin)."""
    from panda3d.core import SamplerState, Texture

    h, w = image_rgb.shape[:2]
    tex = Texture(name)
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
    tex.setRamImageAs(np.ascontiguousarray(image_rgb[::-1]).tobytes(), "RGB")   # Panda3D : origine en bas
    tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    tex.setMagfilter(SamplerState.FT_linear)
    tex.setAnisotropicDegree(4)
    if not repeat:
        tex.setWrapU(SamplerState.WM_clamp)
        tex.setWrapV(SamplerState.WM_clamp)
    return tex
