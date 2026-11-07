"""Panneau latéral de la vue 3D : classifieur, état de l'épisode, vue caméra (entrée du modèle), minicarte, crédits.

Le panneau occupe une colonne à droite de la fenêtre : la vue 3D est réduite à la zone restante (``app.py``),
rien ne la recouvre. Tout est placé dans ``pixel2d`` (unités : pixels ; z vers le haut, donc négatif vers le bas).

La vue caméra montre l'image **réellement donnée au classifieur** (banque d'obstacles), pas le rendu 3D :
les modèles 3D ne servent qu'à l'affichage (skill ``sim3d-architecture``).
"""

from __future__ import annotations

import re

import numpy as np

from simulator.contracts import CLASSES, CLASSES_FR
from simulator.engine import Detection, Episode
from simulator.render3d.actors import verdict

PANEL_W = 400
MARGIN = 16
CREDITS = ("Carte et bâtiments © contributeurs OpenStreetMap (ODbL). Modèles 3D : Kenney, Quaternius (CC0) ; "
           "« Bicycle », « Motorcycle » par Poly by Google, « Truckk » par KolosStudios (CC-BY 3.0, poly.pizza).")
HELP = ("Espace pause · → pas à pas · ↑/↓ vitesse · V vue · S scénario suivant · 1 à 4 scénarios Go/NoGo · "
        "0 tirage · R · N épisode suivant · Échap")

# Couleurs par classe : celles de ui/theme.py, de 0 à 1.
CLASS_COLORS = {
    "person": (0.84, 0.28, 0.25), "bicycle": (0.18, 0.55, 0.29), "car": (0.15, 0.39, 0.78),
    "motorcycle": (0.55, 0.32, 0.75), "bus": (0.87, 0.53, 0.10), "truck": (0.47, 0.35, 0.24),
    "unknown": (0.51, 0.51, 0.51),
}
TEXT, QUIET, BACKGROUND = (0.12, 0.12, 0.12, 1), (0.40, 0.40, 0.40, 1), (0.965, 0.955, 0.935, 1)
SIMULATION_COLOR = (0.80, 0.42, 0.05, 1)
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
        font = loader.loadFont(Filename.fromOsSpecific(path).getFullpath())
        if font.getNumPages() == 0:                 # police déjà utilisée (FontPool) : réglage impossible
            font.setPixelsPerUnit(48)
        return font
    except (ImportError, ValueError):
        return None


def image_texture(image_rgb: np.ndarray):
    from panda3d.core import Texture

    h, w = image_rgb.shape[:2]
    tex = Texture("entree_du_modele")
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
    tex.setRamImageAs(np.ascontiguousarray(image_rgb[::-1]).tobytes(), "RGB")   # Panda3D : origine en bas
    return tex


class EventBanner:
    """Bandeau en haut de la vue 3D : ce que la voiture vient de faire (obstacle classé, demi-tour, arrêt…).

    Il reste affiché quelques pas après l'événement (``Episode.recent_event``) : on voit ainsi chaque étape
    d'un scénario au moment où elle se produit, sans lire le panneau."""

    def __init__(self, base, font=None) -> None:
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self.text = OnscreenText(parent=base.pixel2d, text="", scale=19, fg=(1, 1, 1, 1), align=TextNode.ACenter,
                                 font=font, mayChange=True)
        node = self.text.textNode
        node.setCardColor(0.08, 0.10, 0.14, 0.82)
        node.setCardAsMargin(0.8, 0.8, 0.45, 0.35)
        node.setCardDecal(True)
        self._shown = None

    def layout(self, scene_w: int) -> None:
        self.text.setPos(scene_w / 2, -46)

    def update(self, episode: Episode) -> None:
        event = episode.recent_event()
        if event != self._shown:
            self._shown = event
            self.text.setText(event or "")
            (self.text.show if event else self.text.hide)()


class SidePanel:
    def __init__(self, base, font=None) -> None:
        from direct.gui.DirectGui import DirectFrame
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self.base, self.font = base, font
        self.root = base.pixel2d.attachNewNode("panneau")
        self.background = DirectFrame(parent=self.root, frameColor=BACKGROUND, frameSize=(0, PANEL_W, -1, 0))
        DirectFrame(parent=self.root, frameColor=(0.80, 0.79, 0.76, 1), frameSize=(0, 1, -4000, 0))   # séparation
        inner = PANEL_W - 2 * MARGIN

        def text(y, scale, fg=TEXT, x=MARGIN, width=inner):
            return OnscreenText(parent=self.root, text="", pos=(x, -y), scale=scale, fg=fg, align=TextNode.ALeft,
                                font=font, mayChange=True, wordwrap=width / scale)

        self.banner = text(30, 16)
        self.status = text(78, 13.5, QUIET)
        text(172, 14, QUIET).setText("Vue caméra · entrée du modèle")
        self.image = None
        self.image_box = (MARGIN, 184, 136)                   # x, y du coin haut gauche, côté (pixels)
        right = MARGIN + 136 + 14
        self.waiting = text(214, 14, QUIET, x=right, width=PANEL_W - right - MARGIN)
        self.prediction = text(204, 20, x=right, width=PANEL_W - right - MARGIN)
        self.details = text(228, 13, QUIET, x=right, width=PANEL_W - right - MARGIN)
        self.bars = []
        for i, label in enumerate(CLASSES):
            y = 344 + i * 21
            text(y, 13, x=MARGIN, width=110).setText(CLASSES_FR[label])
            DirectFrame(parent=self.root, frameColor=(0.88, 0.87, 0.84, 1), frameSize=(110, inner + MARGIN, -y - 4, -y + 11))
            bar = DirectFrame(parent=self.root, frameColor=(*CLASS_COLORS[label], 1),
                              frameSize=(0, inner + MARGIN - 110, -y - 4, -y + 11))
            bar.setX(110)
            self.bars.append(bar)
        self.minimap_title = text(486, 14, QUIET)
        self.minimap_title.setText("Minicarte · le haut = sens de marche")
        self.minimap_top = 496
        self.help = text(0, 11.5, QUIET)
        self.help.setText(HELP)
        self.credits = text(0, 10, QUIET)
        self.credits.setText(CREDITS)
        self._detection: Detection | None = None
        self._show_detection(None)

    def layout(self, window_w: int, window_h: int) -> tuple[int, int, int]:
        """Place le panneau à droite. Renvoie la zone de la minicarte : (x, y du haut, côté), en pixels."""
        self.root.setPos(window_w - PANEL_W, 0, 0)
        self.background["frameSize"] = (0, PANEL_W, -window_h, 0)
        self.help.setPos(MARGIN, -(window_h - 98))          # aide sur trois lignes au plus
        self.credits.setPos(MARGIN, -(window_h - 50))
        side = max(80, min(PANEL_W - 2 * MARGIN, window_h - 118 - self.minimap_top))
        return window_w - PANEL_W + MARGIN, self.minimap_top, side

    def update(self, episode: Episode, view: str, time_factor: float, paused: bool, street: str,
               traffic: str = "", scenario: str = "") -> None:
        r = episode.result
        name = episode.classifier.name
        self.banner.setText(name)
        self.banner["fg"] = SIMULATION_COLOR if "SIMULATION" in name.upper() else TEXT
        state = r.status or ("pause" if paused else "en route")
        self.status.setText(
            f"Épisode {episode.episode} · pas {episode.step_count} · {episode.speed_mps * 3.6:.0f} km/h · "
            f"×{time_factor:g} · {state}\n{latin(street)}\n"
            f"Obstacles classés {len(r.encounters)} / {len(episode.obstacles)} · collisions {r.collisions} · "
            f"recalculs {r.replans}\nVue {view}{traffic}" + (f"\n{scenario}" if scenario else "")
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
                bar.setSx(0.001)
            return
        self.waiting.setText("")
        image = detection.image_rgb
        x, y, side = self.image_box
        h, w = image.shape[:2]
        sx, sz = (side / 2, side / 2 * h / w) if w >= h else (side / 2 * w / h, side / 2)
        self.image = OnscreenImage(image=image_texture(image), parent=self.root,
                                   pos=(x + side / 2, 0, -(y + side / 2)), scale=(sx, 1, sz))
        pred, true_label = detection.prediction, detection.obstacle.true_label
        word, color = verdict(pred.label, true_label)
        self.prediction.setText(f"{pred.label_fr} {pred.score:.2f}")
        self.prediction["fg"] = (*color, 1)
        self.details.setText(f"{word}\nVraie classe : {CLASSES_FR[true_label]}\nDécision : {detection.action.value}")
        for bar, p in zip(self.bars, pred.probs):
            bar.setSx(max(float(p), 0.001))
