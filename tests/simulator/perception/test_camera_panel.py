"""Tests de la vue caméra, sans fenêtre (pilote vidéo « dummy »).

La vue caméra utilise ``simulator/ui/theme.py`` (équipe A) : ces tests sont ignorés tant que ce fichier
n'est pas fusionné dans ``main``.
"""

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402

theme = pytest.importorskip("simulator.ui.theme", reason="simulator/ui/theme.py (équipe A) pas encore dans main")

from simulator.contracts import CLASSES, UNKNOWN, Action, Obstacle, Prediction  # noqa: E402
from simulator.perception.images import PLACEHOLDER_COLORS, placeholder_image  # noqa: E402
from simulator.ui.camera_panel import CameraPanel, verdict  # noqa: E402

CAR_80 = Prediction(label="car", score=0.8, probs=(0.04, 0.04, 0.8, 0.04, 0.04, 0.04), infer_ms=1.2)
UNSURE_30 = Prediction(label=UNKNOWN, score=0.3, probs=(0.3, 0.14, 0.14, 0.14, 0.14, 0.14), infer_ms=1.0)
OUTSIDE = (1, 2, 3)
RECT = pygame.Rect(40, 20, 408, 230)


@dataclass(frozen=True)
class FakeDetection:
    """Mêmes champs que ``simulator.engine.Detection``."""

    obstacle: Obstacle
    image_rgb: np.ndarray
    prediction: Prediction
    action: Action


def _detection(true_label: str, prediction: Prediction, image: np.ndarray | None = None) -> FakeDetection:
    image = placeholder_image(true_label) if image is None else image
    return FakeDetection(Obstacle((1, 2, 0), true_label, None), image, prediction, Action.REROUTE)


@pytest.fixture
def screen() -> pygame.Surface:
    surface = pygame.Surface((500, 300))
    surface.fill(OUTSIDE)
    return surface


def _pixels(surface: pygame.Surface) -> np.ndarray:
    return pygame.surfarray.array3d(surface).swapaxes(0, 1)  # (H, W, 3)


@pytest.mark.parametrize(
    "detection",
    [
        _detection("car", CAR_80),
        _detection("bus", CAR_80),                                          # erreur
        _detection("person", UNSURE_30),                                    # inconnu
        _detection("car", CAR_80, np.zeros((40, 200, 3), dtype=np.uint8)),  # image très allongée
        None,                                                               # aucun obstacle vu
    ],
)
def test_draws_only_inside_rect(screen, detection):
    CameraPanel(RECT).draw(screen, detection)
    pixels = _pixels(screen)
    inside = np.zeros(pixels.shape[:2], dtype=bool)
    inside[RECT.top:RECT.bottom, RECT.left:RECT.right] = True
    assert (pixels[~inside] == OUTSIDE).all()                    # rien hors du panneau
    assert (pixels[inside] == theme.PANEL).all(axis=1).any()      # le panneau est bien dessiné
    assert screen.get_clip() == screen.get_rect()                # la zone de découpe est restaurée


def test_obstacle_image_is_shown(screen):
    CameraPanel(RECT).draw(screen, _detection("truck", CAR_80))
    assert (_pixels(screen) == PLACEHOLDER_COLORS["truck"]).all(axis=2).sum() > 1000


@pytest.mark.parametrize("size", [(408, 230), (300, 160), (200, 120), (60, 40)])
def test_other_sizes_do_not_crash(screen, size):
    CameraPanel(pygame.Rect(0, 0, *size)).draw(screen, _detection("car", CAR_80))


def test_every_class_can_be_drawn(screen):
    for label in CLASSES:
        probs = tuple(0.75 if c == label else 0.05 for c in CLASSES)
        pred = Prediction(label=label, score=0.75, probs=probs, infer_ms=0.5)
        CameraPanel(RECT).draw(screen, _detection(label, pred))


def test_verdict():
    assert verdict(CAR_80, "car") == ("juste", theme.GOOD)
    assert verdict(CAR_80, "person") == ("faux", theme.BLOCKED)
    assert verdict(UNSURE_30, "person") == ("inconnu", theme.WARNING)


def test_rejects_invalid_image(screen):
    with pytest.raises(ValueError):
        CameraPanel(RECT).draw(screen, _detection("car", CAR_80, np.zeros((10, 10), dtype=np.uint8)))


def test_save_path_defaults_to_reports(tmp_path):
    from simulator.config import REPORTS_DIR
    from simulator.ui.camera_panel import DEFAULT_SAVE, save_path

    assert DEFAULT_SAVE.parent == REPORTS_DIR
    assert save_path("apercu.png") == REPORTS_DIR / "apercu.png"   # jamais à la racine du dépôt
    assert save_path(str(tmp_path / "x.png")) == tmp_path / "x.png"
    assert save_path("captures/x.png") == __import__("pathlib").Path("captures/x.png")


def test_placeholder_colors_match_theme():
    """Les images de remplacement reprennent la couleur de chaque classe sur la carte et dans le journal."""
    assert PLACEHOLDER_COLORS == {label: theme.CLASS_COLORS[label] for label in CLASSES}
