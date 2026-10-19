"""Vue caméra (équipe B) : image de l'obstacle, classe prédite, vraie classe, score et 6 probabilités.

Appelée par l'interface de l'équipe A à chaque image (signature proposée dans l'issue #1, point 4) :

    draw_camera_panel(screen, pygame.Rect(x, y, 300, 520), image_rgb, prediction, true_label, classifier.name)

Le panneau ne dessine que dans ``rect``. Sans obstacle en vue, ``prediction=None`` affiche « en attente ».
Le résultat est écrit en toutes lettres (« juste », « faux », « inconnu ») : la couleur seule ne suffit pas.

Aperçu sans le reste du simulateur :
    python -m simulator.ui.camera_panel                       # fenêtre, touche Espace = obstacle suivant
    python -m simulator.ui.camera_panel --save apercu.png     # image, sans fenêtre
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pygame

from simulator.contracts import CLASSES, CLASSES_FR, UNKNOWN, Prediction
from simulator.perception.images import check_image_rgb

BACKGROUND = (24, 26, 31)
FRAME = (70, 74, 84)
TEXT = (235, 236, 240)
MUTED = (150, 154, 165)
BAR_TRACK = (48, 51, 59)
BAR_FILL = (96, 140, 200)
BAR_PREDICTED = (120, 180, 255)
CORRECT = (80, 190, 120)
WRONG = (230, 85, 75)
UNSURE = (240, 170, 60)

PADDING = 12


@lru_cache(maxsize=8)
def _font(size: int, bold: bool = False) -> pygame.font.Font:
    """Police par défaut de pygame (accents compris), mise en cache par taille."""
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, size)
    font.set_bold(bold)
    return font


def _text(surface: pygame.Surface, text: str, pos: tuple[int, int], size: int,
          color: tuple[int, int, int] = TEXT, bold: bool = False, max_width: int | None = None) -> int:
    """Écrit une ligne (tronquée à ``max_width`` avec « … ») et renvoie sa hauteur."""
    font = _font(size, bold)
    if max_width is not None and font.size(text)[0] > max_width:
        while text and font.size(text + "…")[0] > max_width:
            text = text[:-1]
        text += "…"
    rendered = font.render(text, True, color)
    surface.blit(rendered, pos)
    return rendered.get_height()


def _image_surface(image_rgb: np.ndarray, box: pygame.Rect) -> tuple[pygame.Surface, pygame.Rect]:
    """Convertit l'image (H, W, 3) en surface pygame, agrandie sans déformation et centrée dans ``box``."""
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(image_rgb.transpose(1, 0, 2)))  # pygame : (W, H, 3)
    h, w = image_rgb.shape[:2]
    scale = min(box.width / w, box.height / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    scaled = pygame.transform.smoothscale(surf, size)
    return scaled, scaled.get_rect(center=box.center)


def verdict(prediction: Prediction, true_label: str) -> tuple[str, tuple[int, int, int]]:
    """Résultat affiché sous l'image : (texte, couleur)."""
    if prediction.label == UNKNOWN:
        return "inconnu : arrêt prudent", UNSURE
    if prediction.label == true_label:
        return "juste", CORRECT
    return "faux", WRONG


def draw_camera_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    image_rgb: np.ndarray | None,
    prediction: Prediction | None,
    true_label: str | None,
    classifier_name: str | None = None,
) -> None:
    """Dessine le panneau caméra dans ``rect``. Ne modifie aucun pixel hors de ``rect``."""
    if image_rgb is not None:
        check_image_rgb(image_rgb)
    if true_label is not None and true_label not in CLASSES:
        raise ValueError(f"true_label doit être une classe entraînée, reçu {true_label!r}")

    previous_clip = surface.get_clip()
    surface.set_clip(rect)
    try:
        _draw(surface, rect, image_rgb, prediction, true_label, classifier_name)
    finally:
        surface.set_clip(previous_clip)


def _draw(surface: pygame.Surface, rect: pygame.Rect, image_rgb: np.ndarray | None,
          prediction: Prediction | None, true_label: str | None, classifier_name: str | None) -> None:
    pygame.draw.rect(surface, BACKGROUND, rect)
    pygame.draw.rect(surface, FRAME, rect, width=1)
    inner_w = rect.width - 2 * PADDING
    x = rect.x + PADDING
    y = rect.y + PADDING

    y += _text(surface, "Caméra", (x, y), 26, bold=True) + 2
    if classifier_name:
        y += _text(surface, classifier_name, (x, y), 18, MUTED, max_width=inner_w)
    y += 8

    # Image de l'obstacle : un carré qui occupe au plus 40 % de la hauteur du panneau.
    side = max(1, min(inner_w, int(rect.height * 0.40)))
    box = pygame.Rect(x + (inner_w - side) // 2, y, side, side)
    pygame.draw.rect(surface, BAR_TRACK, box)
    if image_rgb is not None:
        scaled, where = _image_surface(image_rgb, box)
        surface.blit(scaled, where)
    else:
        w, h = _font(20).size("aucun obstacle")
        _text(surface, "aucun obstacle", (box.centerx - w // 2, box.centery - h // 2), 20, MUTED)
    y = box.bottom + 10

    if prediction is None:
        _text(surface, "En attente d'un obstacle…", (x, y), 22, MUTED, max_width=inner_w)
        return

    predicted = CLASSES_FR.get(prediction.label, "Inconnu")
    y += _text(surface, f"Prédit : {predicted}", (x, y), 24, bold=True, max_width=inner_w) + 4
    if true_label is not None:
        y += _text(surface, f"Vraie classe : {CLASSES_FR[true_label]}", (x, y), 20, max_width=inner_w) + 3
        text, color = verdict(prediction, true_label)
        y += _text(surface, f"Résultat : {text}", (x, y), 20, color, bold=True, max_width=inner_w) + 3
    y += _text(surface, f"Score : {prediction.score:.2f}   ·   {prediction.infer_ms:.1f} ms",
               (x, y), 18, MUTED, max_width=inner_w) + 8

    _draw_bars(surface, pygame.Rect(x, y, inner_w, rect.bottom - PADDING - y), prediction, true_label)


def _draw_bars(surface: pygame.Surface, area: pygame.Rect, prediction: Prediction, true_label: str | None) -> None:
    """Une barre par classe, dans l'ordre de CLASSES. La vraie classe est marquée d'un trait à gauche."""
    if area.height <= 0:
        return
    row_h = max(12, min(26, area.height // len(CLASSES)))
    label_w = min(80, area.width // 3)
    value_w = 44
    bar_w = max(1, area.width - label_w - value_w - 8)
    for i, (label, prob) in enumerate(zip(CLASSES, prediction.probs)):
        top = area.y + i * row_h
        if top + row_h > area.bottom + 1:
            break
        if label == true_label:
            pygame.draw.rect(surface, CORRECT, (area.x, top + 2, 3, row_h - 6))
        _text(surface, CLASSES_FR[label], (area.x + 6, top + 2), 18, max_width=label_w - 6)
        track = pygame.Rect(area.x + label_w, top + 3, bar_w, row_h - 8)
        pygame.draw.rect(surface, BAR_TRACK, track)
        fill_color = BAR_PREDICTED if label == prediction.label else BAR_FILL
        pygame.draw.rect(surface, fill_color, (track.x, track.y, round(bar_w * prob), track.height))
        _text(surface, f"{prob * 100:.0f} %", (track.right + 6, top + 2), 18, MUTED)


def _demo(save: Path | None) -> None:
    """Aperçu : obstacles tirés au hasard, classés par le mode simulation, sur images de remplacement ou réelles."""
    from simulator.contracts import Obstacle
    from simulator.perception.obstacle_bank import ObstacleBank
    from simulator.perception.simulated import SimulatedClassifier

    pygame.init()
    size = (320, 560)
    screen = pygame.display.set_mode(size) if save is None else pygame.Surface(size)
    if save is None:
        pygame.display.set_caption("AutoMaze AI — vue caméra")
    bank = ObstacleBank()
    classifier = SimulatedClassifier(error_rate=0.3, seed=0)
    rng = np.random.default_rng(0)

    def next_obstacle() -> None:
        label = CLASSES[int(rng.integers(len(CLASSES)))]
        obstacle = Obstacle(edge=(0, 0, 0), true_label=label, image_path=bank.sample_image_path(label, rng))
        image = bank.load_image(obstacle)
        pred = classifier.predict(image, true_label=label)
        draw_camera_panel(screen, screen.get_rect(), image, pred, label, classifier.name)

    next_obstacle()
    if save is not None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aperçu de la vue caméra (mode simulation)")
    parser.add_argument("--save", type=Path, help="enregistre l'aperçu dans une image au lieu d'ouvrir une fenêtre")
    _demo(parser.parse_args().save)
