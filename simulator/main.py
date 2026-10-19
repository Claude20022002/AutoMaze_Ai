"""Lance le simulateur AutoMaze AI dans une fenêtre pygame.

    python -m simulator.main                                   # classifieur « simulation »
    python -m simulator.main --classifier ours --episode 3     # notre CNN, épisode n° 3
    python -m simulator.main --screenshot capture.png          # image après 150 pas, sans fenêtre

Commandes : Espace pause · → pas à pas · ↑/↓ vitesse · V vue conducteur/carte · F11 plein écran ·
R recommencer · N épisode suivant · Échap quitter. La fenêtre se redimensionne à la souris.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from simulator.config import SimConfig
from simulator.contracts import CLASSES, Classifier, Prediction
from simulator.engine import Detection, Episode, gray_placeholder



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

    from simulator.ui.driver_view import DriverView
    from simulator.ui.hud import Hud
    from simulator.ui.map_view import MapView
    from simulator.ui.minimap import MiniMap
    from simulator.world.road_map import landmark_node, load_map

    config = SimConfig(seed=args.seed)
    graph = load_map()
    classifier, images = load_perception(args.classifier, config)

    pygame.init()
    size = args.size or default_window_size()
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.set_caption("AutoMaze AI — simulateur (Casablanca)")
    focus = [landmark_node(graph, config.start), landmark_node(graph, config.goal)]
    driver_view = DriverView(graph, pygame.Rect(0, 0, *size))  # la géométrie de la ville est préparée une fois
    views: dict = {}
    view_mode = args.view

    def layout(width: int, height: int) -> None:
        """Place les zones selon la taille de la fenêtre : grande vue à gauche, panneau à droite."""
        side_w = min(max(int(width * 0.30), 380), 460)
        main_area = pygame.Rect(0, 0, width - side_w, height)
        x, w = main_area.right + 16, side_w - 32
        driver_view.resize(main_area)
        views.update(
            map=MapView(graph, main_area, focus_nodes=focus),
            minimap=MiniMap(graph, center=(main_area.right - 135, 135)),
            camera=load_camera_panel(pygame.Rect(x, 16, w, 230)),
            hud=Hud(pygame.Rect(x, 262, w, height - 278)),
        )

    layout(*size)

    def new_episode(number: int) -> Episode:
        return Episode(graph, classifier, config, number, image_for=images)

    episode_number = args.episode
    episode = new_episode(episode_number)
    clock = pygame.time.Clock()
    time_factor, paused, budget, fullscreen = args.time_factor, False, 0.0, False
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
            if event.type == pygame.VIDEORESIZE and not fullscreen:
                width, height = max(event.w, 1000), max(event.h, 640)
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                layout(width, height)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    advance()
                elif event.key == pygame.K_v:
                    view_mode = "carte" if view_mode == "conducteur" else "conducteur"
                elif event.key == pygame.K_UP:
                    time_factor = min(time_factor * 2, 32)
                elif event.key == pygame.K_DOWN:
                    time_factor = max(time_factor / 2, 0.25)
                elif event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    screen = pygame.display.set_mode((0, 0) if fullscreen else size,
                                                     pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
                    layout(*screen.get_size())
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
                        episode_number += 1
                    episode = new_episode(episode_number)
                    previous = (episode.car.position_xy(), episode.car.heading())

        dt = clock.tick(args.fps) / 1000
        if not paused and not episode.done and not args.screenshot:
            # Temps réel × time_factor : un pas représente config.step_seconds secondes.
            budget += dt * time_factor / config.step_seconds
            while budget >= 1 and not episode.done:
                advance()
                budget -= 1

        # Position affichée : interpolée entre le pas précédent et le pas actuel (mouvement fluide).
        alpha = 1.0 if (paused or episode.done or args.screenshot) else min(budget, 1.0)
        (px, py), (hx, hy) = previous
        cx, cy = episode.car.position_xy()
        if abs(cx - px) + abs(cy - py) > 60:  # demi-tour ou nouvel épisode : pas d'interpolation
            alpha = 1.0
        hx2, hy2 = episode.car.heading()
        position = (px + (cx - px) * alpha, py + (cy - py) * alpha)
        heading = (hx + (hx2 - hx) * alpha, hy + (hy2 - hy) * alpha)

        screen.fill((236, 234, 228))
        if view_mode == "conducteur":
            driver_view.draw(screen, episode, images, position, heading)
            views["minimap"].draw(screen, episode, position, heading)
        else:
            views["map"].draw(screen, episode, config.start, config.goal)
        views["camera"].draw(screen, episode.last_detection)
        views["hud"].draw(screen, episode, paused, time_factor)
        pygame.display.flip()

        if args.screenshot:
            out = Path(args.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(out))
            pygame.quit()
            return out


def default_window_size() -> tuple[int, int]:
    """90 % de l'écran, entre 1100 × 700 et 1920 × 1080 (1280 × 800 si l'écran est inconnu)."""
    import pygame

    info = pygame.display.Info()
    if not info.current_w or not info.current_h or info.current_w < 0:
        return 1280, 800
    return min(max(int(info.current_w * 0.9), 1100), 1920), min(max(int(info.current_h * 0.85), 700), 1080)


def _size(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="simulation", help="simulation, reference ou ours (équipe B)")
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--time-factor", type=float, default=1.0, help="1 = temps réel, 2 = deux fois plus vite (↑/↓)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=_size, help="taille de la fenêtre, par exemple 1600x900 (défaut : selon l'écran)")
    parser.add_argument("--view", choices=("conducteur", "carte"), default="conducteur", help="vue au démarrage (touche V)")
    parser.add_argument("--screenshot", help="enregistre une image au lieu d'ouvrir la fenêtre")
    parser.add_argument("--steps", type=int, default=150, help="pas simulés avant la capture")
    out = run(parser.parse_args(argv))
    if out:
        print(f"Capture enregistrée : {out}")


if __name__ == "__main__":
    sys.exit(main())
