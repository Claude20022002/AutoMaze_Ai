"""Interface 2D par-dessus la vue 3D : classifieur, état de l'épisode, vue caméra (entrée du modèle), crédits.

La vue caméra montre l'image **réellement donnée au classifieur** (banque d'obstacles), pas le rendu 3D :
les modèles 3D ne servent qu'à l'affichage (skill ``sim3d-architecture``).
"""

from __future__ import annotations

import re

import numpy as np

from simulator.contracts import CLASSES, CLASSES_FR
from simulator.engine import Detection, Episode
from simulator.render3d.actors import verdict

CREDITS = ("Carte et bâtiments : © contributeurs OpenStreetMap (ODbL) · Modèles 3D : Kenney, Quaternius (CC0) ; "
           "« Bicycle », « Motorcycle » par Poly by Google, « Truckk » par KolosStudios (CC-BY 3.0, poly.pizza)")
HELP = "Espace pause · → pas à pas · ↑/↓ vitesse · V vue · R recommencer · N épisode suivant · Échap quitter"

# Couleurs par classe : celles de ui/theme.py, de 0 à 1.
CLASS_COLORS = {
    "person": (0.84, 0.28, 0.25), "bicycle": (0.18, 0.55, 0.29), "car": (0.15, 0.39, 0.78),
    "motorcycle": (0.55, 0.32, 0.75), "bus": (0.87, 0.53, 0.10), "truck": (0.47, 0.35, 0.24),
}
TEXT, QUIET, PANEL = (0.12, 0.12, 0.12, 1), (0.38, 0.38, 0.38, 1), (1, 1, 1, 0.86)
_NON_LATIN = re.compile(r"[֐-ࣿיִ-﷿ﹰ-﻿]+")   # comme ui/theme.py


def latin(text: str) -> str:
    """Partie latine d'un nom de rue bilingue : l'arabe s'afficherait sans liaisons ni sens de lecture."""
    return " ".join(_NON_LATIN.sub(" ", text).split()) or text


def accented_font(loader):
    """Police DejaVu Sans (accents) livrée avec matplotlib ; police par défaut de Panda3D sinon."""
    try:
        from matplotlib import font_manager
        from panda3d.core import Filename

        path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
        return loader.loadFont(Filename.fromOsSpecific(path).getFullpath())
    except (ImportError, ValueError):
        return None


def image_texture(image_rgb: np.ndarray):
    from panda3d.core import Texture

    h, w = image_rgb.shape[:2]
    tex = Texture("entree_du_modele")
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
    tex.setRamImageAs(np.ascontiguousarray(image_rgb[::-1]).tobytes(), "RGB")   # Panda3D : origine en bas
    return tex


class Hud:
    def __init__(self, base, font=None) -> None:
        from direct.gui.DirectGui import DirectFrame
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self.base, self.font = base, font

        def text(parent, pos, scale, fg=TEXT, align=TextNode.ALeft, wordwrap=None):
            return OnscreenText(parent=parent, text="", pos=pos, scale=scale, fg=fg, align=align, font=font,
                                mayChange=True, wordwrap=wordwrap)

        top_left, top_right = base.a2dTopLeft, base.a2dTopRight
        DirectFrame(parent=top_left, frameColor=PANEL, frameSize=(0.02, 1.52, -0.36, -0.02))
        self.banner = text(top_left, (0.05, -0.09), 0.055)
        self.status = text(top_left, (0.05, -0.17), 0.042, QUIET, wordwrap=34)

        # Vue caméra : image classée + prédiction + 6 probabilités
        DirectFrame(parent=top_right, frameColor=PANEL, frameSize=(-0.98, -0.02, -1.02, -0.02))
        text(top_right, (-0.94, -0.08), 0.042, QUIET).setText("Vue caméra · entrée du modèle")
        self.image = None
        self.image_box = (-0.94, -0.13, 0.40)         # coin haut gauche (x, y) et côté, en unités aspect2d
        self.prediction = text(top_right, (-0.50, -0.20), 0.052)
        self.details = text(top_right, (-0.50, -0.28), 0.038, QUIET, wordwrap=12)
        self.waiting = text(top_right, (-0.94, -0.20), 0.045, QUIET)
        self.bars, self.bar_labels = [], []
        for i, label in enumerate(CLASSES):
            y = -0.62 - i * 0.065
            name = text(top_right, (-0.94, y), 0.038)
            name.setText(CLASSES_FR[label])
            DirectFrame(parent=top_right, frameColor=(0.9, 0.9, 0.88, 1), frameSize=(-0.68, -0.08, y - 0.012, y + 0.035))
            bar = DirectFrame(parent=top_right, frameColor=(*CLASS_COLORS[label], 1), frameSize=(0, 0.6, y - 0.012, y + 0.035))
            bar.setPos(-0.68, 0, 0)
            self.bars.append(bar)

        DirectFrame(parent=base.a2dBottomLeft, frameColor=PANEL, frameSize=(0.02, 3.4, 0.015, 0.145))
        self.help = text(base.a2dBottomLeft, (0.04, 0.10), 0.038, QUIET)
        self.help.setText(HELP)
        self.credits = text(base.a2dBottomLeft, (0.04, 0.04), 0.028, QUIET)
        self.credits.setText(CREDITS)
        self._detection: Detection | None = None

    def update(self, episode: Episode, view: str, time_factor: float, paused: bool, street: str) -> None:
        r = episode.result
        name = episode.classifier.name
        self.banner.setText(name)
        self.banner["fg"] = (0.80, 0.42, 0.05, 1) if "SIMULATION" in name.upper() else TEXT
        state = r.status or ("pause" if paused else "en route")
        self.status.setText(
            f"Épisode {episode.episode} · pas {episode.step_count} · {episode.speed_mps * 3.6:.0f} km/h · "
            f"×{time_factor:g} · {state}\n{latin(street)}\n"
            f"Obstacles classés {len(r.encounters)} / {len(episode.obstacles)} · collisions {r.collisions} · "
            f"recalculs {r.replans} · vue {view}"
        )
        if episode.last_detection is not self._detection:
            self._show_detection(episode.last_detection)

    def _show_detection(self, detection: Detection | None) -> None:
        from direct.gui.OnscreenImage import OnscreenImage

        self._detection = detection
        if self.image is not None:
            self.image.destroy()
            self.image = None
        if detection is None:
            self.waiting.setText("En attente d'un obstacle…")
            self.prediction.setText("")
            self.details.setText("")
            for bar in self.bars:
                bar.setScale(0.001, 1, 1)
            return
        self.waiting.setText("")
        image = detection.image_rgb
        x, y, side = self.image_box
        h, w = image.shape[:2]
        sx, sy = (side / 2, side / 2 * h / w) if w >= h else (side / 2 * w / h, side / 2)
        self.image = OnscreenImage(image=image_texture(image), parent=self.base.a2dTopRight,
                                   pos=(x + side / 2, 0, y - side / 2), scale=(sx, 1, sy))
        pred, true_label = detection.prediction, detection.obstacle.true_label
        word, color = verdict(pred.label, true_label)
        self.prediction.setText(f"{pred.label_fr} {pred.score:.2f}")
        self.prediction["fg"] = (*color, 1)
        self.details.setText(f"{word}\nVraie classe : {CLASSES_FR[true_label]}\nDécision : {detection.action.value}")
        for bar, p in zip(self.bars, pred.probs):
            bar.setScale(max(float(p), 0.001), 1, 1)
