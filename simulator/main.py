"""Lance le simulateur AutoMaze AI dans une fenêtre pygame.

    python -m simulator.main                                   # classifieur « simulation »
    python -m simulator.main --classifier ours --episode 3     # notre CNN, épisode n° 3
    python -m simulator.main --screenshot capture.png          # image après 150 pas, sans fenêtre
    python -m simulator.main --scenario pieton                 # scénario (liste : python -m simulator.scenarios -h)

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


def episode_images(config: SimConfig):
    """Fonction ``épisode -> images des obstacles`` : les images changent d'un épisode à l'autre, comme dans
    ``compare.py`` (sinon un itinéraire fixe ferait revenir les mêmes images)."""
    try:
        from simulator.perception import make_image_provider
    except ImportError:
        return lambda episode: gray_placeholder
    return lambda episode: make_image_provider(config, episode=episode)


def make_episode(graph, classifier: Classifier, config: SimConfig, number: int, images, scenario=None) -> Episode:
    """Épisode tiré au hasard (comportement historique) ou épisode d'un scénario (``simulator.scenarios``).
    ``images`` : fonction ``épisode -> images des obstacles`` (``episode_images``)."""
    if scenario is None:
        return Episode(graph, classifier, config, number, image_for=images(number))
    from simulator.scenarios import build_episode

    return build_episode(scenario, graph, classifier, config, number, image_for=images(number))


def scenario_status(scenario, episode: Episode) -> str:
    """Une ligne pour l'interface : le scénario joué et, à la fin de l'épisode, s'il s'est réalisé."""
    if scenario is None:
        return ""
    from simulator.scenarios import check

    if not episode.done:
        return f"{scenario.title} · en cours"
    failed = [c.label for c in check(scenario, episode) if not c.ok]
    return f"{scenario.title} · " + ("réalisé" if not failed else "non réalisé : " + "; ".join(failed).lower())


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

    from simulator.scenarios import SCENARIOS

    config = SimConfig(seed=args.seed)
    graph = load_map()
    classifier, images = load_perception(args.classifier, config)[0], episode_images(config)
    scenario = SCENARIOS[args.scenario] if args.scenario else None

    def places(scenario) -> tuple[str, str, list[int]]:
        """Noms du départ et de l'arrivée, et intersections à garder dans le cadre de la vue carte."""
        if scenario is None:
            return config.start, config.goal, [landmark_node(graph, config.start), landmark_node(graph, config.goal)]
        names = [scenario.start, *(s.place for s in scenario.stops), scenario.goal]
        return scenario.start, scenario.goal, [landmark_node(graph, name) for name in names]

    start_name, goal_name, focus = places(scenario)

    pygame.init()
    size = args.size or default_window_size()
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.set_caption("AutoMaze AI — simulateur (Casablanca)")
    driver_view = DriverView(graph, pygame.Rect(0, 0, *size))  # la géométrie de la ville est préparée une fois
    views: dict = {}
    view_mode = args.view
    SCENARIO_KEYS = {pygame.K_0: None, pygame.K_1: "route_libre", pygame.K_2: "pieton",   # noqa: N806
                     pygame.K_3: "route_bloquee", pygame.K_4: "contournement"}

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
        return make_episode(graph, classifier, config, number, images, scenario)

    episode_number = args.episode
    episode = new_episode(episode_number)
    clock = pygame.time.Clock()
    time_factor, paused, budget, fullscreen = args.time_factor, False, 0.0, False
    episode.car.record_path = True     # trace du chemin : la voiture le suit entre deux pas (virages, demi-tours)
    previous = 0.0                     # distance parcourue au pas précédent

    def advance() -> None:
        nonlocal previous
        previous = episode.car.distance_m
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
                elif event.key in (pygame.K_s, *SCENARIO_KEYS):     # S : scénario suivant ; 0 à 4 : choix direct
                    keys = [None, *SCENARIOS]
                    if event.key == pygame.K_s:
                        chosen = keys[(keys.index(scenario.key if scenario else None) + 1) % len(keys)]
                    else:
                        chosen = SCENARIO_KEYS[event.key]
                    scenario = SCENARIOS[chosen] if chosen else None
                    start_name, goal_name, focus = places(scenario)
                    layout(*screen.get_size())
                    episode_number = 0
                    episode = new_episode(episode_number)
                    episode.car.record_path, previous = True, 0.0
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
                        episode_number += 1
                    episode = new_episode(episode_number)
                    episode.car.record_path, previous = True, 0.0

        dt = clock.tick(args.fps) / 1000
        if not paused and not episode.done and not args.screenshot:
            # Temps réel × time_factor : un pas représente config.step_seconds secondes.
            budget += dt * time_factor / config.step_seconds
            while budget >= 1 and not episode.done:
                advance()
                budget -= 1

        # Position affichée : le long du chemin réellement parcouru pendant le pas (mouvement fluide, sans saut).
        alpha = 1.0 if (paused or episode.done or args.screenshot) else min(budget, 1.0)
        distance = previous + (episode.car.distance_m - previous) * alpha
        position, heading = episode.car.pose_at(distance, CAR_BODY_M)

        screen.fill((236, 234, 228))
        if view_mode == "conducteur":
            driver_view.draw(screen, episode, episode.image_for, position, heading)
            views["minimap"].draw(screen, episode, position, heading)
        else:
            views["map"].draw(screen, episode, start_name, goal_name)
        views["camera"].draw(screen, episode.last_detection)
        views["hud"].draw(screen, episode, paused, time_factor, scenario_status(scenario, episode))
        draw_event_banner(screen, episode, views["map"].rect)
        pygame.display.flip()

        if args.screenshot:
            out = Path(args.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(out))
            pygame.quit()
            return out


CAR_BODY_M = 4.4    # longueur de la voiture : son cap suit le chemin entre l'avant et l'arrière


def draw_event_banner(surface, episode: Episode, area) -> None:
    """Bandeau en haut de la grande vue : ce que la voiture vient de faire (obstacle classé, demi-tour, arrêt…)."""
    import pygame

    from simulator.ui import theme

    event = episode.recent_event()
    if not event:
        return
    font = theme.font(17, bold=True)
    width, height = font.size(event)
    box = pygame.Rect(0, 0, min(width + 32, area.w - 32), height + 14)
    box.midtop = (area.centerx, area.y + 14)
    pygame.draw.rect(surface, (24, 28, 36), box, border_radius=8)
    theme.draw_text(surface, event, (box.x + 16, box.y + 7), 17, theme.PANEL, bold=True, max_width=box.w - 32)


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


def scenario_keys() -> list[str]:
    from simulator.scenarios import SCENARIOS

    return list(SCENARIOS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="simulation", help="simulation, reference ou ours (équipe B)")
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--scenario", choices=scenario_keys(), help="scénario Go/NoGo ou trajet quotidien (défaut : tirage)")
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
