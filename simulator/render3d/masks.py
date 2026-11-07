"""Masques de caméra : quels objets voit chaque caméra (vue principale, minicarte, carte d'ombre du soleil).

Un objet est visible par toutes les caméras, sauf s'il est caché pour l'une d'elles avec ``node.hide(masque)``.
"""

from __future__ import annotations

from panda3d.core import BitMask32

MAIN_MASK = BitMask32.bit(0)      # caméra principale
MINIMAP_MASK = BitMask32.bit(1)   # minicarte
SHADOW_MASK = BitMask32.bit(2)    # caméra du soleil (carte d'ombre)


def only_on_minimap(node) -> None:
    node.hide(MAIN_MASK | SHADOW_MASK)
    node.show(MINIMAP_MASK)


def only_on_main_view(node) -> None:
    node.hide(MINIMAP_MASK | SHADOW_MASK)


def no_shadow(node) -> None:
    """Objet qui ne doit pas projeter d'ombre (ligne d'itinéraire, étiquettes, repères)."""
    node.hide(SHADOW_MASK)
