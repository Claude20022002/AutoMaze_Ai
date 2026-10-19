"""Tests de la vue caméra, sans fenêtre (pilote vidéo « dummy »)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402

from simulator.contracts import CLASSES, UNKNOWN, Prediction  # noqa: E402
from simulator.perception.images import placeholder_image  # noqa: E402
from simulator.ui.camera_panel import BACKGROUND, CORRECT, UNSURE, WRONG, draw_camera_panel, verdict  # noqa: E402

CAR_80 = Prediction(label="car", score=0.8, probs=(0.04, 0.04, 0.8, 0.04, 0.04, 0.04), infer_ms=1.2)
UNSURE_30 = Prediction(label=UNKNOWN, score=0.3, probs=(0.3, 0.14, 0.14, 0.14, 0.14, 0.14), infer_ms=1.0)
OUTSIDE = (1, 2, 3)
PANEL = pygame.Rect(40, 20, 300, 520)


@pytest.fixture
def screen() -> pygame.Surface:
    surface = pygame.Surface((400, 600))
    surface.fill(OUTSIDE)
    return surface


def _pixels(surface: pygame.Surface) -> np.ndarray:
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)  # (H, W, 3)


@pytest.mark.parametrize(
    "image, prediction, true_label",
    [
        (placeholder_image("car"), CAR_80, "car"),
        (placeholder_image("bus"), CAR_80, "bus"),
        (placeholder_image("person"), UNSURE_30, "person"),
        (None, None, None),                                    # aucun obstacle en vue
        (np.zeros((40, 200, 3), dtype=np.uint8), CAR_80, None),  # image très allongée, vraie classe absente
    ],
)
def test_draws_only_inside_rect(screen, image, prediction, true_label):
    draw_camera_panel(screen, PANEL, image, prediction, true_label, "SIMULATION — modèle réel non chargé")
    pixels = _pixels(screen)
    inside = np.zeros(pixels.shape[:2], dtype=bool)
    inside[PANEL.top:PANEL.bottom, PANEL.left:PANEL.right] = True
    assert (pixels[~inside] == OUTSIDE).all()               # rien hors du panneau
    assert (pixels[inside] == BACKGROUND).all(axis=1).any()  # le panneau est bien dessiné
    assert screen.get_clip() == screen.get_rect()           # la zone de découpe est restaurée


def test_obstacle_image_is_shown(screen):
    draw_camera_panel(screen, PANEL, placeholder_image("truck"), CAR_80, "truck")
    from simulator.perception.images import PLACEHOLDER_COLORS

    assert (_pixels(screen) == PLACEHOLDER_COLORS["truck"]).all(axis=2).sum() > 1000


@pytest.mark.parametrize("size", [(120, 200), (300, 260), (60, 60)])
def test_small_panels_do_not_crash(screen, size):
    draw_camera_panel(screen, pygame.Rect(0, 0, *size), placeholder_image("car"), CAR_80, "car")


def test_verdict():
    assert verdict(CAR_80, "car") == ("juste", CORRECT)
    assert verdict(CAR_80, "person") == ("faux", WRONG)
    assert verdict(UNSURE_30, "person")[1] == UNSURE


def test_rejects_invalid_inputs(screen):
    with pytest.raises(ValueError):
        draw_camera_panel(screen, PANEL, np.zeros((10, 10), dtype=np.uint8), CAR_80, "car")
    with pytest.raises(ValueError):
        draw_camera_panel(screen, PANEL, placeholder_image("car"), CAR_80, UNKNOWN)


def test_every_class_can_be_drawn(screen):
    for label in CLASSES:
        probs = tuple(0.75 if c == label else 0.05 for c in CLASSES)
        pred = Prediction(label=label, score=0.75, probs=probs, infer_ms=0.5)
        draw_camera_panel(screen, PANEL, placeholder_image(label), pred, label)
