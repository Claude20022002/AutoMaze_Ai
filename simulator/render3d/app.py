"""Vue 3D du simulateur AutoMaze AI (Panda3D), sur la carte réelle de Casablanca.

    python -m simulator.render3d.app                                  # classifieur « simulation »
    python -m simulator.render3d.app --classifier ours --episode 3    # notre CNN, épisode n° 3
    python -m simulator.render3d.app --view conducteur                # poste de conduite et tableau de bord
    python -m simulator.render3d.app --traffic 0 --pedestrians 0      # sans trafic d'ambiance
    python -m simulator.render3d.app --screenshot capture.png         # image après 60 pas, sans fenêtre
    python -m simulator.render3d.app --scenario contournement         # scénario Go/NoGo ou trajet quotidien

Commandes : Espace pause · → pas à pas · ↑/↓ vitesse · V vue (poursuite, conducteur, dessus) ·
R recommencer · N épisode suivant · Échap quitter.

Mise en page : la vue 3D occupe la partie gauche de la fenêtre, le panneau d'information la colonne de droite
(rien ne recouvre la scène). En vue conducteur, un tableau de bord occupe le bas de la zone 3D.

La logique est celle de ``simulator.engine`` : mêmes épisodes, mêmes décisions et mêmes journaux qu'en mode
sans affichage. La 3D ne fait que lire l'état. Le classifieur reçoit les images de la banque d'obstacles,
jamais le rendu 3D. Premier lancement : la ville est construite (une trentaine de secondes) puis mise en cache.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from simulator.config import MAP_FILE, SimConfig
from simulator.engine import Episode

VIEWS = ("poursuite", "conducteur", "dessus")
DASHBOARD_H = 190            # hauteur du tableau de bord (vue conducteur), en pixels


def _size(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


@dataclass(frozen=True)
class Layout:
    """Zones de la fenêtre, en fractions (gauche, droite, bas, haut) comme ``DisplayRegion.setDimensions``."""

    scene: tuple[float, float, float, float]
    scene_px: tuple[int, int]               # largeur, hauteur de la zone 3D en pixels
    dashboard_px: tuple[int, int, int, int]  # x, y du haut, largeur, hauteur (pixels depuis le haut) ; hauteur 0 = aucun


def compute_layout(width: int, height: int, panel_w: int, driver_view: bool) -> Layout:
    """Vue 3D à gauche du panneau ; en vue conducteur, tableau de bord sous la vue 3D."""
    scene_w = max(1, width - panel_w)
    dash_h = DASHBOARD_H if driver_view else 0
    scene_h = max(1, height - dash_h)
    return Layout(scene=(0.0, scene_w / width, dash_h / height, 1.0), scene_px=(scene_w, scene_h),
                  dashboard_px=(0, scene_h, scene_w, dash_h))


class Simulator3D:
    def __init__(self, args: argparse.Namespace) -> None:
        from panda3d.core import loadPrcFileData

        width, height = args.size
        offscreen = args.screenshot is not None
        loadPrcFileData("", "\n".join([
            f"window-type {'offscreen' if offscreen else 'onscreen'}",
            f"win-size {width} {height}",
            "window-title AutoMaze AI — vue 3D (Casablanca)",
            "audio-library-name null",
            "sync-video true",
            "hardware-animated-vertices true",              # piétons animés par le GPU
            "framebuffer-multisample 1" if args.msaa else "",
            "multisamples 4" if args.msaa else "",
        ]))
        from direct.showbase.ShowBase import ShowBase

        from simulator.main import load_perception
        from simulator.render3d import buildings
        from simulator.render3d.actors import EgoCar, RouteLine
        from simulator.render3d.catalog import load_catalog
        from simulator.render3d.dashboard import Dashboard
        from simulator.render3d.environment import Environment
        from simulator.render3d.hud import SidePanel, accented_font
        from simulator.render3d.minimap import MiniMap
        from simulator.render3d.world_mesh import CACHE_DIR, WorldFrame, build_world, cache_key
        from simulator.world.road_map import load_map

        self.base = base = ShowBase()
        self.args = args
        self.config = SimConfig(seed=args.seed)
        self.graph = load_map()
        from simulator.main import episode_images
        from simulator.scenarios import SCENARIOS

        self.classifier, self.images = load_perception(args.classifier, self.config)[0], episode_images(self.config)

        self.cli_ambiance = (args.traffic, args.pedestrians)     # None = selon le scénario
        self.set_scenario(args.scenario, start=False)
        self.catalog = load_catalog()
        self.frame = WorldFrame.from_graph(self.graph)
        self.font = accented_font(base.loader)

        base.disableMouse()
        base.camLens.setFov(60)
        base.camLens.setNearFar(0.3, 1200)
        base.win.setClearColor((0.10, 0.11, 0.13, 1))            # fond des zones d'interface
        self.environment = Environment(base, shadows=not args.no_shadows)
        cache = CACHE_DIR / f"ville_{cache_key(MAP_FILE, buildings.BUILDINGS_FILE)}.bam"
        if not cache.exists():
            print("Construction de la ville 3D (une seule fois, mise en cache)…")
            for stale in CACHE_DIR.glob("ville_*.bam"):   # caches d'une ancienne carte ou d'une ancienne géométrie
                stale.unlink()
        world = build_world(self.graph, self.frame, base.loader, self.catalog, buildings.load(), cache_file=cache)
        world.reparentTo(base.render)

        self.dynamic = base.render.attachNewNode("dynamique")
        # Les objets mobiles projettent des ombres mais n'en reçoivent pas : sur leurs petites facettes, l'ombrage
        # par pixel produit des rayures (« shadow acne »). Ils gardent l'éclairage classique.
        self.dynamic.setShaderOff(1)
        self.ego = EgoCar(self.dynamic, self.catalog, base.loader, self.frame)
        self.route_line = RouteLine(self.dynamic, self.graph, self.frame)
        self.panel = SidePanel(base, self.font)
        from simulator.render3d.hud import EventBanner

        self.banner = EventBanner(base, self.font)
        self.dashboard = Dashboard(base, self.font, self.config.sensor_range_m)
        self.minimap = MiniMap(base, self.ego.node)
        self.view = args.view
        self.time_factor, self.paused, self.budget = args.time_factor, False, 0.0
        self.episode_number = args.episode
        self.obstacle_views = []
        self.traffic = self.traffic_view = None
        self._layout_key = None
        self.new_episode()
        self.apply_layout()

        if not offscreen:
            for key, action in {"space": self.toggle_pause, "arrow_right": self.single_step,
                                "arrow_up": lambda: self.set_speed(2), "arrow_down": lambda: self.set_speed(0.5),
                                "v": self.next_view, "r": lambda: self.new_episode(),
                                "n": lambda: self.new_episode(self.episode_number + 1), "escape": sys.exit,
                                "s": self.next_scenario, "0": lambda: self.set_scenario(None)}.items():
                base.accept(key, action)
            for number, key in enumerate(("route_libre", "pieton", "route_bloquee", "contournement"), start=1):
                base.accept(str(number), self.set_scenario, [key])
            base.taskMgr.add(self.tick, "sim_tick")

    # ------------------------------------------------------------------ mise en page
    def apply_layout(self, force: bool = False) -> None:
        """Zones d'affichage selon la taille de la fenêtre et la vue. Rappelé à chaque image (redimensionnement)."""
        from simulator.render3d.hud import PANEL_W

        width, height = self.base.win.getXSize(), self.base.win.getYSize()
        driver = self.view == "conducteur"
        key = (width, height, driver)
        layout = compute_layout(width, height, PANEL_W, driver)
        aspect = layout.scene_px[0] / layout.scene_px[1]
        # ShowBase remet le format de la fenêtre entière à chaque redimensionnement : on le corrige ici.
        if key == self._layout_key and not force and abs(self.base.camLens.getAspectRatio() - aspect) < 1e-3:
            return
        self._layout_key = key
        self.base.camNode.getDisplayRegion(0).setDimensions(*layout.scene)
        self.base.camLens.setAspectRatio(aspect)
        x, top, side = self.panel.layout(width, height)
        self.minimap.set_region(x / width, (x + side) / width, 1 - (top + side) / height, 1 - top / height)
        dx, dy, dw, dh = layout.dashboard_px
        self.dashboard.layout(dx, dy, dw, dh)
        self.dashboard.show(driver)

    # ------------------------------------------------------------------ épisodes
    def set_scenario(self, key: str | None, start: bool = True) -> None:
        """Change de scénario (touches S, 1 à 4, 0) ; l'ambiance suit le scénario sauf si la ligne de commande la fixe."""
        from simulator.scenarios import SCENARIOS

        self.scenario = SCENARIOS[key] if key else None
        traffic, pedestrians = self.cli_ambiance
        self.args.traffic = traffic if traffic is not None else (self.scenario.traffic if self.scenario else 80)
        self.args.pedestrians = pedestrians if pedestrians is not None else (
            self.scenario.pedestrians if self.scenario else 40)
        if start:
            self.new_episode(0)

    def next_scenario(self) -> None:
        """Tirage au hasard, puis chaque scénario dans l'ordre (Go/NoGo, puis trajets quotidiens)."""
        from simulator.scenarios import SCENARIOS

        keys = [None, *SCENARIOS]
        self.set_scenario(keys[(keys.index(self.scenario.key if self.scenario else None) + 1) % len(keys)])

    def new_episode(self, number: int | None = None) -> None:
        from simulator.render3d.actors import ObstacleView

        if number is not None:
            self.episode_number = number
        for view in self.obstacle_views:
            view.destroy()
        if self.traffic_view is not None:
            self.traffic_view.destroy()
            self.traffic = self.traffic_view = None
        from simulator.main import make_episode

        self.episode = make_episode(self.graph, self.classifier, self.config, self.episode_number, self.images,
                                    self.scenario)
        if self.args.traffic > 0 or self.args.pedestrians > 0:
            from simulator.render3d.traffic import TrafficSim
            from simulator.render3d.traffic_view import TrafficView

            # Trafic d'ambiance : sa propre graine ; il évite les tronçons des obstacles de l'épisode.
            self.traffic = TrafficSim(self.graph, seed=self.config.seed * 1000 + self.episode_number,
                                      n_vehicles=self.args.traffic, n_pedestrians=self.args.pedestrians,
                                      blocked_edges={state.obstacle.edge for state in self.episode.obstacles})
            self.traffic.set_ego(self._ego_state())
            self.traffic.populate()
            self.traffic_view = TrafficView(self.dynamic, self.traffic, self.graph, self.frame, self.catalog,
                                            self.base.loader, self.minimap)
        seed = self.config.seed * 100_003 + self.episode_number
        self.obstacle_views = [ObstacleView(self.dynamic, state, self.graph, self.catalog, self.base.loader,
                                            self.frame, seed, self.font, self.minimap)
                               for state in self.episode.obstacles]
        self.episode.car.record_path = True       # trace du chemin : la carrosserie le suit entre deux pas
        self.previous_distance = 0.0
        self.budget = 0.0

    def advance(self) -> None:
        self.previous_distance = self.episode.car.distance_m
        self.episode.step()

    def _ego_state(self):
        from simulator.render3d.traffic import ego_state

        return ego_state(self.episode, self.catalog.ego.length_m)

    def step_traffic(self, seconds: float) -> None:
        """Avance le trafic d'ambiance de ``seconds`` secondes simulées, par pas de 0,1 s au plus."""
        if self.traffic is None:
            return
        steps = max(1, math.ceil(seconds / 0.1))
        for _ in range(steps):
            self.traffic.set_ego(self._ego_state())
            self.traffic.step(seconds / steps)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def single_step(self) -> None:
        if self.paused and not self.episode.done:
            self.advance()
            self.budget = 1.0

    def set_speed(self, factor: float) -> None:
        self.time_factor = min(max(self.time_factor * factor, 0.25), 32)

    def next_view(self) -> None:
        self.view = VIEWS[(VIEWS.index(self.view) + 1) % len(VIEWS)]

    # ------------------------------------------------------------------ boucle
    def tick(self, task):
        from direct.task import Task
        from panda3d.core import ClockObject

        dt = min(ClockObject.getGlobalClock().getDt(), 0.1)
        if not self.paused:
            self.step_traffic(dt * self.time_factor)       # le trafic continue après l'arrivée de l'ego
        if not self.paused and not self.episode.done:
            # Temps réel × time_factor : un pas du moteur représente config.step_seconds secondes.
            self.budget += dt * self.time_factor / self.config.step_seconds
            while self.budget >= 1 and not self.episode.done:
                self.advance()
                self.budget -= 1
        self.render_state()
        return Task.cont

    def render_state(self) -> None:
        from simulator.render3d.masks import MAIN_MASK

        self.apply_layout()
        episode = self.episode
        alpha = 1.0 if (self.paused or episode.done) else min(self.budget, 1.0)
        # La voiture avance le long du chemin réellement parcouru pendant le pas (virages, demi-tours compris),
        # et non en ligne droite entre deux positions : plus de saut ni de coupe de trottoir.
        from simulator.render3d.actors import ego_body

        start, end = self.previous_distance, episode.car.distance_m
        center, heading = ego_body(self.graph, episode.car, start + (end - start) * alpha, self.ego.length)
        self.ego.place(center, heading)
        # Vue conducteur : la carrosserie de l'ego masquerait la route (elle reste visible sur la minicarte).
        (self.ego.node.hide if self.view == "conducteur" else self.ego.node.show)(MAIN_MASK)
        self._place_camera()
        self.environment.follow(self.ego.node, self.base.camera)
        self.minimap.update()

        t = episode.step_count - 1 + alpha if not episode.done else float(episode.step_count)
        camera_pos = self.base.camera.getPos(self.base.render)
        for view in self.obstacle_views:
            view.update(t, camera_pos)
        self.route_line.update(episode)
        traffic = ""
        if self.traffic_view is not None:
            self.traffic_view.update(camera_pos)
            traffic = f" · trafic {len(self.traffic.vehicles)} véhicules, {len(self.traffic.pedestrians)} piétons"
        street = self._street()
        from simulator.main import scenario_status

        self.panel.update(episode, self.view, self.time_factor, self.paused, street, traffic,
                          scenario_status(self.scenario, episode))
        if self.view == "conducteur":
            from simulator.render3d.hud import latin

            self.dashboard.update(episode, latin(street))

    def _place_camera(self) -> None:
        cam, car = self.base.camera, self.ego.node
        half = self.catalog.ego.length_m / 2
        if self.view == "poursuite":
            cam.setPos(car, 0, -11, 4.5)
            cam.lookAt(car, 0, 8, 1.2)
        elif self.view == "conducteur":
            cam.setPos(car, -0.35, half - 1.9, 1.2)     # place du conducteur (on roule à droite : volant à gauche)
            cam.lookAt(car, -0.35, half + 30, 0.9)
        else:
            cam.setPos(car, 0, -30, 110)
            cam.lookAt(car, 0, 12, 0)

    def _street(self) -> str:
        from simulator.world.road_map import street_name

        edge = self.episode.car.current_edge
        if edge is None:
            return f"Arrêt prévu : {self.episode.at_stop}" if self.episode.at_stop else "Arrivée"
        return street_name(self.graph, edge)

    # ------------------------------------------------------------------ capture sans fenêtre
    def screenshot(self, path: Path, steps: int) -> Path:
        from PIL import Image

        from simulator.render3d.contact_sheet import _capture

        for _ in range(steps):
            if self.episode.done:
                break
            self.advance()
            self.step_traffic(self.config.step_seconds)
        self.budget = 1.0
        self.render_state()
        self.base.taskMgr.step()                       # anime les personnages (Actor) une fois
        self.render_state()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_capture(self.base)).save(path)
        return path


def main(argv: list[str] | None = None) -> None:
    from simulator.main import scenario_keys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="simulation", help="simulation, reference ou ours")
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--time-factor", type=float, default=1.0, help="1 = temps réel (↑/↓ pour changer)")
    parser.add_argument("--size", type=_size, default=(1440, 860), help="taille de la fenêtre, par exemple 1600x900")
    parser.add_argument("--view", choices=VIEWS, default="poursuite", help="vue au démarrage (touche V)")
    parser.add_argument("--no-shadows", action="store_true", help="sans ombres portées (si l'affichage saccade)")
    parser.add_argument("--scenario", choices=scenario_keys(), help="scénario Go/NoGo ou trajet quotidien (défaut : tirage)")
    parser.add_argument("--traffic", type=int, help="véhicules du trafic d'ambiance (0 = aucun ; défaut : 80 ou selon le scénario)")
    parser.add_argument("--pedestrians", type=int, help="piétons sur les trottoirs (0 = aucun ; défaut : 40 ou selon le scénario)")
    parser.add_argument("--msaa", action="store_true", help="anticrénelage (plus joli, plus lent sur GPU intégré)")
    parser.add_argument("--screenshot", type=Path, help="enregistre une image au lieu d'ouvrir la fenêtre")
    parser.add_argument("--steps", type=int, default=60, help="pas simulés avant la capture")
    args = parser.parse_args(argv)

    simulator = Simulator3D(args)
    if args.screenshot is not None:
        out = simulator.screenshot(args.screenshot, args.steps)
        print(f"Capture enregistrée : {out}")
        simulator.base.destroy()
        return
    simulator.base.run()


if __name__ == "__main__":
    main()
