"""Vue 3D du simulateur AutoMaze AI (Panda3D), sur la carte réelle de Casablanca.

    python -m simulator.render3d.app                                  # classifieur « simulation »
    python -m simulator.render3d.app --classifier ours --episode 3    # notre CNN, épisode n° 3
    python -m simulator.render3d.app --screenshot capture.png         # image après 60 pas, sans fenêtre

Commandes : Espace pause · → pas à pas · ↑/↓ vitesse · V vue (poursuite, conducteur, dessus) ·
R recommencer · N épisode suivant · Échap quitter.

La logique est celle de ``simulator.engine`` : mêmes épisodes, mêmes décisions et mêmes journaux qu'en mode
sans affichage. La 3D ne fait que lire l'état. Le classifieur reçoit les images de la banque d'obstacles,
jamais le rendu 3D. Premier lancement : la ville est construite (environ 15 s) puis mise en cache.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from simulator.config import MAP_FILE, SimConfig
from simulator.engine import Episode

VIEWS = ("poursuite", "conducteur", "dessus")
SKY = (0.74, 0.84, 0.94)


def _size(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


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
            "framebuffer-multisample 1" if args.msaa else "",
            "multisamples 4" if args.msaa else "",
        ]))
        from direct.showbase.ShowBase import ShowBase

        from simulator.main import load_perception
        from simulator.render3d import buildings
        from simulator.render3d.actors import EgoCar, RouteLine
        from simulator.render3d.catalog import load_catalog
        from simulator.render3d.hud import Hud, accented_font
        from simulator.render3d.world_mesh import CACHE_DIR, WorldFrame, build_world, cache_key
        from simulator.world.road_map import load_map

        self.base = base = ShowBase()
        self.args = args
        self.config = SimConfig(seed=args.seed)
        self.graph = load_map()
        self.classifier, self.images = load_perception(args.classifier, self.config)
        self.catalog = load_catalog()
        self.frame = WorldFrame.from_graph(self.graph)
        self.font = accented_font(base.loader)

        self._setup_scene()
        cache = CACHE_DIR / f"ville_{cache_key(MAP_FILE, buildings.BUILDINGS_FILE)}.bam"
        if not cache.exists():
            print("Construction de la ville 3D (une seule fois, mise en cache)…")
            for stale in CACHE_DIR.glob("ville_*.bam"):   # caches d'une ancienne carte ou d'une ancienne géométrie
                stale.unlink()
        world = build_world(self.graph, self.frame, base.loader, self.catalog, buildings.load(), cache_file=cache)
        world.reparentTo(base.render)

        self.dynamic = base.render.attachNewNode("dynamique")
        self.ego = EgoCar(self.dynamic, self.catalog, base.loader, self.frame)
        self.route_line = RouteLine(self.dynamic, self.graph, self.frame)
        self.hud = Hud(base, self.font)
        self.view = args.view
        self.time_factor, self.paused, self.budget = args.time_factor, False, 0.0
        self.episode_number = args.episode
        self.obstacle_views = []
        self.new_episode()

        if not offscreen:
            for key, action in {"space": self.toggle_pause, "arrow_right": self.single_step,
                                "arrow_up": lambda: self.set_speed(2), "arrow_down": lambda: self.set_speed(0.5),
                                "v": self.next_view, "r": lambda: self.new_episode(),
                                "n": lambda: self.new_episode(self.episode_number + 1), "escape": sys.exit}.items():
                base.accept(key, action)
            base.taskMgr.add(self.tick, "sim_tick")

    # ------------------------------------------------------------------ scène
    def _setup_scene(self) -> None:
        from panda3d.core import AmbientLight, DirectionalLight, Fog

        base = self.base
        base.disableMouse()
        base.setBackgroundColor(*SKY, 1)
        base.camLens.setFov(60)
        base.camLens.setNearFar(0.3, 1200)
        sun = base.render.attachNewNode(DirectionalLight("soleil"))
        sun.node().setColor((0.85, 0.82, 0.74, 1))
        sun.setHpr(-35, -55, 0)
        ambient = base.render.attachNewNode(AmbientLight("ambiante"))
        ambient.node().setColor((0.55, 0.56, 0.62, 1))
        base.render.setLight(sun)
        base.render.setLight(ambient)
        fog = Fog("brume")
        fog.setColor(*SKY)
        fog.setExpDensity(0.0022)
        base.render.setFog(fog)

    # ------------------------------------------------------------------ épisodes
    def new_episode(self, number: int | None = None) -> None:
        from simulator.render3d.actors import ObstacleView

        if number is not None:
            self.episode_number = number
        for view in self.obstacle_views:
            view.destroy()
        self.episode = Episode(self.graph, self.classifier, self.config, self.episode_number, image_for=self.images)
        seed = self.config.seed * 100_003 + self.episode_number
        self.obstacle_views = [ObstacleView(self.dynamic, state, self.graph, self.catalog, self.base.loader,
                                            self.frame, seed, self.font) for state in self.episode.obstacles]
        self.previous = (self.episode.car.position_xy(), self.episode.car.heading())
        self.budget = 0.0

    def advance(self) -> None:
        self.previous = (self.episode.car.position_xy(), self.episode.car.heading())
        self.episode.step()

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

        dt = ClockObject.getGlobalClock().getDt()
        if not self.paused and not self.episode.done:
            # Temps réel × time_factor : un pas du moteur représente config.step_seconds secondes.
            self.budget += dt * self.time_factor / self.config.step_seconds
            while self.budget >= 1 and not self.episode.done:
                self.advance()
                self.budget -= 1
        self.render_state()
        return Task.cont

    def render_state(self) -> None:
        episode = self.episode
        alpha = 1.0 if (self.paused or episode.done) else min(self.budget, 1.0)
        (px, py), (hx, hy) = self.previous
        cx, cy = episode.car.position_xy()
        if abs(cx - px) + abs(cy - py) > 60:          # demi-tour ou nouvel épisode : pas d'interpolation
            alpha = 1.0
        hx2, hy2 = episode.car.heading()
        position = (px + (cx - px) * alpha, py + (cy - py) * alpha)
        heading = (hx + (hx2 - hx) * alpha, hy + (hy2 - hy) * alpha)
        self.ego.place(position, heading)
        self._place_camera()

        t = episode.step_count - 1 + alpha if not episode.done else float(episode.step_count)
        camera_pos = self.base.camera.getPos(self.base.render)
        for view in self.obstacle_views:
            view.update(t, camera_pos)
        self.route_line.update(episode)
        self.hud.update(episode, self.view, self.time_factor, self.paused, self._street())

    def _place_camera(self) -> None:
        cam, car = self.base.camera, self.ego.node
        if self.view == "poursuite":
            cam.setPos(car, 0, -11, 4.5)
            cam.lookAt(car, 0, 8, 1.2)
        elif self.view == "conducteur":
            cam.setPos(car, -0.35, 0.2, 1.25)
            cam.lookAt(car, -0.35, 25, 1.0)
        else:
            cam.setPos(car, 0, -30, 110)
            cam.lookAt(car, 0, 12, 0)

    def _street(self) -> str:
        from simulator.world.road_map import street_name

        edge = self.episode.car.current_edge
        return street_name(self.graph, edge) if edge is not None else "Arrivée"

    # ------------------------------------------------------------------ capture sans fenêtre
    def screenshot(self, path: Path, steps: int) -> Path:
        from PIL import Image

        from simulator.render3d.contact_sheet import _capture

        for _ in range(steps):
            if self.episode.done:
                break
            self.advance()
        self.budget = 1.0
        self.render_state()
        self.base.taskMgr.step()                       # anime les personnages (Actor) une fois
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_capture(self.base)).save(path)
        return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="simulation", help="simulation, reference ou ours")
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--time-factor", type=float, default=1.0, help="1 = temps réel (↑/↓ pour changer)")
    parser.add_argument("--size", type=_size, default=(1440, 860), help="taille de la fenêtre, par exemple 1600x900")
    parser.add_argument("--view", choices=VIEWS, default="poursuite", help="vue au démarrage (touche V)")
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
