"""Vue caméra (équipe B) : image de l'obstacle, classe prédite, vraie classe, score, décision et 6 probabilités.

Interface fixée à la section 3 du plan, utilisée par ``main.py`` (équipe A) :

    camera = CameraPanel(pygame.Rect(x, y, 408, 230))
    camera.draw(screen, episode.last_detection)   # Detection (engine.py) ou None

Le panneau ne dessine que dans son rectangle. Le résultat est écrit en toutes lettres (« juste », « faux »,
« inconnu ») : la couleur seule ne suffit pas. Couleurs et polices : ``simulator/ui/theme.py``.

Aperçu sans le reste du simulateur (mode simulation, images de test ou de remplacement) :
    python -m simulator.ui.camera_panel                       # fenêtre ; Espace = obstacle suivant
    python -m simulator.ui.camera_panel --save                # image reports/simulator/apercu_camera.png
    python -m simulator.ui.camera_panel --save autre.png      # nom simple : écrit aussi dans reports/simulator/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame

from simulator.config import REPORTS_DIR
from simulator.contracts import CLASSES, CLASSES_FR, UNKNOWN, Prediction
from simulator.perception.images import check_image_rgb
from simulator.ui import theme

if TYPE_CHECKING:
    from simulator.engine import Detection

PAD = 16
IMAGE_BG = (236, 234, 228)
BAR_TRACK = (236, 234, 228)
LIGHTEN = 0.55  # barres des classes non prédites : couleur de la classe éclaircie
DEFAULT_SAVE = REPORTS_DIR / "apercu_camera.png"


def verdict(prediction: Prediction, true_label: str) -> tuple[str, tuple[int, int, int]]:
    """Résultat affiché : (texte, couleur)."""
    if prediction.label == UNKNOWN:
        return "inconnu", theme.WARNING
    if prediction.label == true_label:
        return "juste", theme.GOOD
    return "faux", theme.BLOCKED


def _lighten(color: tuple[int, int, int], amount: float = LIGHTEN) -> tuple[int, int, int]:
    return tuple(round(c + (255 - c) * amount) for c in color)  # type: ignore[return-value]


def _image_surface(image_rgb: np.ndarray, box: pygame.Rect) -> tuple[pygame.Surface, pygame.Rect]:
    """Convertit l'image (H, W, 3) en surface pygame, agrandie sans déformation et centrée dans ``box``."""
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(image_rgb.swapaxes(0, 1)))  # pygame : (W, H, 3)
    h, w = image_rgb.shape[:2]
    scale = min(box.width / w, box.height / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    scaled = pygame.transform.smoothscale(surf, size)
    return scaled, scaled.get_rect(center=box.center)


class CameraPanel:
    """Panneau « Vue caméra » : ce que la voiture voit et ce que le classifieur en a conclu."""

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)
        if not pygame.font.get_init():  # main.py appelle pygame.init(), mais pas forcément les autres appelants
            pygame.font.init()

    def draw(self, surface: pygame.Surface, detection: Detection | None) -> None:
        """Dessine le dernier obstacle classé (``None`` : aucun obstacle vu pour l'instant)."""
        if detection is not None:
            check_image_rgb(detection.image_rgb)
        previous_clip = surface.get_clip()
        surface.set_clip(self.rect)
        try:
            self._draw(surface, detection)
        finally:
            surface.set_clip(previous_clip)

    # ------------------------------------------------------------------ dessin
    def _draw(self, surface: pygame.Surface, detection: Detection | None) -> None:
        r = self.rect
        pygame.draw.rect(surface, theme.PANEL, r, border_radius=8)
        pygame.draw.rect(surface, theme.BORDER, r, 1, border_radius=8)
        x, y = r.x + PAD, r.y + 12
        theme.draw_text(surface, "Vue caméra", (x, y), 14, theme.QUIET, bold=True)

        top = y + 26
        side = max(1, min(r.h - (top - r.y) - 14, (r.w - 2 * PAD) // 2))
        box = pygame.Rect(x, top, side, side)
        pygame.draw.rect(surface, IMAGE_BG, box, border_radius=4)

        if detection is None:
            text = "aucun obstacle"
            w, h = theme.font(13).size(text)
            theme.draw_text(surface, text, (box.centerx - w // 2, box.centery - h // 2), 13, theme.QUIET)
            theme.draw_text(surface, "En attente d'un obstacle…", (box.right + 14, top), 14, theme.QUIET,
                            max_width=r.right - PAD - box.right - 14)
            return

        pred, true_label = detection.prediction, detection.obstacle.true_label
        scaled, where = _image_surface(detection.image_rgb, box)
        surface.blit(scaled, where)

        # Temps d'inférence en haut à droite : mesuré par le classifieur, jamais estimé.
        ms = f"{pred.infer_ms:.2f} ms" if pred.infer_ms < 1 else f"{pred.infer_ms:.1f} ms"
        theme.draw_text(surface, ms, (r.right - PAD - theme.font(12).size(ms)[0], y + 1), 12, theme.QUIET)

        self._draw_text_column(surface, pygame.Rect(box.right + 14, top, r.right - PAD - box.right - 14,
                                                    r.bottom - 12 - top), detection, pred, true_label)

    def _draw_text_column(self, surface: pygame.Surface, col: pygame.Rect, detection: Detection,
                          pred: Prediction, true_label: str) -> None:
        x, y, width = col.x, col.y, col.w
        y += theme.draw_text(surface, "Prédiction", (x, y), 12, theme.QUIET)
        name_h = theme.draw_text(surface, pred.label_fr, (x, y), 19, theme.CLASS_COLORS[pred.label], bold=True,
                                 max_width=width - 70)
        score = f"score {pred.score:.2f}"
        theme.draw_text(surface, score, (col.right - theme.font(12).size(score)[0], y + 6), 12, theme.QUIET)
        y += name_h

        text, color = verdict(pred, true_label)
        y += theme.draw_text(surface, f"Vraie classe : {CLASSES_FR[true_label]}", (x, y), 13, max_width=width)
        y += theme.draw_text(surface, f"Résultat : {text}", (x, y), 13, color, bold=True, max_width=width)
        y += theme.draw_text(surface, f"Décision : {detection.action.value}", (x, y), 13, max_width=width) + 6

        self._draw_bars(surface, pygame.Rect(x, y, width, col.bottom - y), pred, true_label)

    @staticmethod
    def _draw_bars(surface: pygame.Surface, area: pygame.Rect, pred: Prediction, true_label: str) -> None:
        """Une barre par classe, dans l'ordre de CLASSES. La vraie classe est écrite en gras."""
        if area.h < len(CLASSES) * 8:
            return
        row = min(16, area.h // len(CLASSES))
        size = max(9, min(12, row - 3))
        label_w = min(62, area.w // 3)
        value_w = theme.font(size).size("100 %")[0] + 4
        bar_w = max(1, area.w - label_w - value_w)
        for i, (label, prob) in enumerate(zip(CLASSES, pred.probs)):
            top = area.y + i * row
            is_true = label == true_label
            theme.draw_text(surface, CLASSES_FR[label], (area.x, top), size,
                            theme.TEXT if is_true else theme.QUIET, bold=is_true, max_width=label_w - 4)
            track = pygame.Rect(area.x + label_w, top + 3, bar_w, max(3, row - 7))
            pygame.draw.rect(surface, BAR_TRACK, track, border_radius=2)
            color = theme.CLASS_COLORS[label]
            fill = color if label == pred.label else _lighten(color)
            if prob > 0:
                pygame.draw.rect(surface, fill, (track.x, track.y, max(1, round(bar_w * prob)), track.h),
                                 border_radius=2)
            pct = f"{prob * 100:.0f} %"
            theme.draw_text(surface, pct, (area.right - theme.font(size).size(pct)[0], top), size, theme.QUIET)


def _demo(save: Path | None) -> None:
    """Aperçu : obstacles tirés au hasard, classés par le mode simulation."""
    from dataclasses import dataclass

    from simulator.config import SimConfig
    from simulator.contracts import Action, Obstacle
    from simulator.perception import make_classifier, make_image_provider

    @dataclass(frozen=True)
    class _Detection:  # mêmes champs que engine.Detection
        obstacle: Obstacle
        image_rgb: np.ndarray
        prediction: Prediction
        action: Action

    pygame.init()
    size = (440, 262)
    screen = pygame.display.set_mode(size) if save is None else pygame.Surface(size)
    if save is None:
        pygame.display.set_caption("AutoMaze AI — vue caméra")
    config = SimConfig()
    classifier, images = make_classifier("simulation", config), make_image_provider(config)
    panel = CameraPanel(pygame.Rect(16, 16, 408, 230))
    rng = np.random.default_rng(0)

    def next_obstacle() -> None:
        label = CLASSES[int(rng.integers(len(CLASSES)))]
        obstacle = Obstacle(edge=(int(rng.integers(1_000_000)), 0, 0), true_label=label, image_path=None)
        image = images(obstacle)
        pred = classifier.predict(image, true_label=label)
        screen.fill(theme.BACKGROUND)
        panel.draw(screen, _Detection(obstacle, image, pred, config.decision_rules[pred.label]))

    next_obstacle()
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, str(save))
        print("Aperçu enregistré :", save)
        return
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                next_obstacle()
        pygame.display.flip()
        clock.tick(30)


def save_path(arg: str) -> Path:
    """Un nom simple (``apercu.png``) va dans reports/simulator/ (ignoré par Git), jamais à la racine du dépôt."""
    path = Path(arg)
    return REPORTS_DIR / path if not path.is_absolute() and path.parent == Path(".") else path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aperçu de la vue caméra (mode simulation)")
    parser.add_argument("--save", nargs="?", const=DEFAULT_SAVE.name, type=save_path,
                        help=f"enregistre l'aperçu dans une image au lieu d'ouvrir une fenêtre (défaut : {DEFAULT_SAVE})")
    _demo(parser.parse_args().save)
