"""Lance le simulateur AutoMaze AI dans une fenêtre pygame.

    python -m simulator.main                                   # classifieur « simulation »
    python -m simulator.main --classifier ours --episode 3     # notre CNN, épisode n° 3
    python -m simulator.main --screenshot capture.png          # image après 150 pas, sans fenêtre

Commandes : Espace pause · → pas à pas · ↑/↓ vitesse · R recommencer · N épisode suivant · Échap quitter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from simulator.config import SimConfig
from simulator.contracts import CLASSES, Classifier, Prediction
from simulator.engine import Detection, Episode, gray_placeholder

WINDOW = (1280, 800)
MAP_WIDTH = 840


class _DemoClassifier(Classifier):
    """Utilisé seulement tant que l'équipe B n'a pas livré simulator.perception : renvoie la vraie classe."""

    name = "SIMULATION — démo sans modèle (vraie classe)"

    def predict(self, image_rgb, true_label=None):
        probs = tuple(1.0 if c == true_label else 0.0 for c in CLASSES)
        return Prediction(label=true_label, score=1.0, probs=probs, infer_ms=0.0)


def load_perception(name: str, config: SimConfig):
    """Classifieur et images de l'équipe B, ou la démo si son module n'existe pas encore."""
    try:
        from simulator.perception import make_classifier, make_image_provider
    except ImportError:
        print("simulator.perception absent (équipe B) : classifieur de démonstration, images grises.")
        return _DemoClassifier(), gray_placeholder
    return make_classifier(name, config), make_image_provider(config)


class _PlaceholderCamera:
    """Vue caméra provisoire, remplacée par simulator.ui.camera_panel.CameraPanel (équipe B)."""

    def __init__(self, rect) -> None:
        self.rect = rect

    def draw(self, surface, detection: Detection | None) -> None:
        import pygame

        from simulator.contracts import CLASSES_FR
        from simulator.ui import theme

        pygame.draw.rect(surface, theme.PANEL, self.rect, border_radius=8)
        pygame.draw.rect(surface, theme.BORDER, self.rect, 1, border_radius=8)
        x, y = self.rect.x + 16, self.rect.y + 12
        theme.draw_text(surface, "Vue caméra (provisoire, équipe B)", (x, y), 14, theme.QUIET, bold=True)
        if detection is None:
            theme.draw_text(surface, "En attente d'un obstacle…", (x, y + 30), 14, theme.QUIET)
            return
        image = pygame.surfarray.make_surface(detection.image_rgb.swapaxes(0, 1))
        side = self.rect.h - 60
        surface.blit(pygame.transform.smoothscale(image, (side, side)), (x, y + 30))
        tx = x + side + 16
        pred = detection.prediction
        theme.draw_text(surface, "Prédiction", (tx, y + 30), 13, theme.QUIET)
        theme.draw_text(surface, pred.label_fr, (tx, y + 48), 20, theme.CLASS_COLORS[pred.label], bold=True)
        theme.draw_text(surface, f"score {pred.score:.2f}", (tx, y + 76), 13, theme.QUIET)
        theme.draw_text(surface, "Vraie classe", (tx, y + 104), 13, theme.QUIET)
        theme.draw_text(surface, CLASSES_FR[detection.obstacle.true_label], (tx, y + 122), 16, theme.TEXT, bold=True)
        theme.draw_text(surface, f"Décision : {detection.action.value}", (tx, y + 152), 13, theme.TEXT)


def load_camera_panel(rect):
    try:
        from simulator.ui.camera_panel import CameraPanel
    except ImportError:
        return _PlaceholderCamera(rect)
    return CameraPanel(rect)


def run(args: argparse.Namespace) -> Path | None:
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # rendu sans fenêtre
    import pygame

    from simulator.ui.hud import Hud
    from simulator.ui.map_view import MapView
    from simulator.world.road_map import load_map

    config = SimConfig(seed=args.seed)
    graph = load_map()
    classifier, images = load_perception(args.classifier, config)

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    pygame.display.set_caption("AutoMaze AI — simulateur (Casablanca)")
    from simulator.world.road_map import landmark_node

    from simulator.ui.driver_view import DriverView
    from simulator.ui.minimap import MiniMap

    focus = [landmark_node(graph, config.start), landmark_node(graph, config.goal)]
    map_view = MapView(graph, pygame.Rect(0, 0, MAP_WIDTH, WINDOW[1]), focus_nodes=focus)
    driver_view = DriverView(graph, pygame.Rect(0, 0, MAP_WIDTH, WINDOW[1]))
    minimap = MiniMap(graph, center=(MAP_WIDTH - 135, 135))
    view_mode = args.view
    side_x, side_w = MAP_WIDTH + 16, WINDOW[0] - MAP_WIDTH - 32
    camera = load_camera_panel(pygame.Rect(side_x, 16, side_w, 230))
    hud = Hud(pygame.Rect(side_x, 262, side_w, WINDOW[1] - 278))

    def new_episode(number: int) -> Episode:
        return Episode(graph, classifier, config, number, image_for=images)

    episode_number = args.episode
    episode = new_episode(episode_number)
    clock = pygame.time.Clock()
    steps_per_second, paused, budget = args.speed, False, 0.0
    previous = (episode.car.position_xy(), episode.car.heading())  # pour lisser la caméra entre deux pas

    def advance() -> None:
        nonlocal previous
        previous = (episode.car.position_xy(), episode.car.heading())
        episode.step()

    if args.screenshot:  # avance de N pas puis enregistre l'image
        for _ in range(args.steps):
            advance()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    advance()
                elif event.key == pygame.K_v:
                    view_mode = "carte" if view_mode == "conducteur" else "conducteur"
                elif event.key == pygame.K_UP:
                    steps_per_second = min(steps_per_second * 2, 160)
                elif event.key == pygame.K_DOWN:
                    steps_per_second = max(steps_per_second / 2, 1.25)
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
                        episode_number += 1
                    episode = new_episode(episode_number)
                    previous = (episode.car.position_xy(), episode.car.heading())

        dt = clock.tick(args.fps) / 1000
        if not paused and not episode.done and not args.screenshot:
            budget += dt * steps_per_second
            while budget >= 1 and not episode.done:
                advance()
                budget -= 1

        # Position affichée : interpolée entre le pas précédent et le pas actuel (mouvement fluide).
        alpha = 1.0 if (paused or episode.done or args.screenshot) else min(budget, 1.0)
        (px, py), (hx, hy) = previous
        cx, cy = episode.car.position_xy()
        if abs(cx - px) + abs(cy - py) > 60:  # demi-tour ou nouvel épisode : pas d'interpolation
            alpha = 1.0
        position = (px + (cx - px) * alpha, py + (cy - py) * alpha)
        heading = episode.car.heading() if alpha >= 1.0 else (hx + (episode.car.heading()[0] - hx) * alpha,
                                                              hy + (episode.car.heading()[1] - hy) * alpha)

        screen.fill((236, 234, 228))
        if view_mode == "conducteur":
            driver_view.draw(screen, episode, images, position, heading)
            minimap.draw(screen, episode, position, heading)
        else:
            map_view.draw(screen, episode, config.start, config.goal)
        camera.draw(screen, episode.last_detection)
        hud.draw(screen, episode, paused, steps_per_second)
        pygame.display.flip()

        if args.screenshot:
            out = Path(args.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(out))
            pygame.quit()
            return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="simulation", help="simulation, reference ou ours (équipe B)")
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=10.0, help="pas simulés par seconde")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--view", choices=("conducteur", "carte"), default="conducteur", help="vue au démarrage (touche V)")
    parser.add_argument("--screenshot", help="enregistre une image au lieu d'ouvrir la fenêtre")
    parser.add_argument("--steps", type=int, default=150, help="pas simulés avant la capture")
    out = run(parser.parse_args(argv))
    if out:
        print(f"Capture enregistrée : {out}")


if __name__ == "__main__":
    sys.exit(main())
