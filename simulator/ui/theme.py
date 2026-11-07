"""Couleurs, polices et petites aides d'affichage communes à toute l'interface pygame."""

from __future__ import annotations

import re
from functools import lru_cache

import pygame

BACKGROUND = (244, 243, 239)
PANEL = (255, 255, 255)
BORDER = (214, 212, 205)
STREET = (206, 204, 198)
TEXT = (32, 32, 30)
QUIET = (112, 110, 104)
ROUTE = (37, 99, 200)
PLANNED = (172, 197, 235)
DRIVEN = (95, 95, 92)
BLOCKED = (206, 60, 48)
CAR = (20, 20, 20)
GOOD = (46, 140, 74)
WARNING = (222, 134, 26)

# Une couleur par classe, la même partout dans l'interface (carte, journal, vue caméra).
CLASS_COLORS = {
    "person": (214, 72, 64),
    "bicycle": (46, 140, 74),
    "car": (37, 99, 200),
    "motorcycle": (140, 82, 190),
    "bus": (222, 134, 26),
    "truck": (120, 90, 60),
    "unknown": (130, 130, 130),
}
CLASS_LETTERS = {"person": "P", "bicycle": "V", "car": "A", "motorcycle": "M", "bus": "B", "truck": "C", "unknown": "?"}

OSM_ATTRIBUTION = "© contributeurs OpenStreetMap (ODbL)"

_NON_LATIN = re.compile(r"[֐-ࣿיִ-﷿ﹰ-﻿]+")


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("segoeui,arial,dejavusans", size, bold=bold)


def latin(text: str) -> str:
    """Garde la partie latine d'un nom de rue bilingue (pygame ne sait pas afficher l'arabe correctement)."""
    return " ".join(_NON_LATIN.sub(" ", text).split()) or text


def draw_text(surface: pygame.Surface, text: str, pos: tuple[int, int], size: int = 15,
              color=TEXT, bold: bool = False, max_width: int | None = None) -> int:
    """Écrit une ligne (tronquée avec « … » si besoin) et renvoie sa hauteur."""
    f = font(size, bold)
    if max_width is not None and f.size(text)[0] > max_width:
        while text and f.size(text + "…")[0] > max_width:
            text = text[:-1]
        text += "…"
    image = f.render(text, True, color)
    surface.blit(image, pos)
    return image.get_height()
