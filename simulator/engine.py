"""Boucle d'un épisode : la voiture roule, détecte les obstacles, les fait classer et réagit.

Un épisode avance pas à pas (``Episode.step``) : l'interface pygame appelle ``step`` à chaque image,
le mode sans affichage appelle ``run`` jusqu'à la fin. À chaque obstacle rencontré, une ``EncounterLog``
est enregistrée ; ``EpisodeResult`` résume l'épisode (statut, collisions, recalculs…).

Les obstacles se comportent selon leur VRAIE classe (``SimConfig.clear_steps``) : une personne finit
de traverser, un camion reste là. Une décision prise sur une mauvaise classe peut donc mener à une collision.

Sans affichage :
    python -m simulator.engine --classifier simulation --episodes 50
"""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from simulator.config import REPORTS_DIR, SimConfig
from simulator.contracts import Action, Classifier, EncounterLog, Obstacle, Prediction
from simulator.world.car import Car
from simulator.world.decisions import apply_action
from simulator.world.obstacles import ObstacleState, place_obstacles
from simulator.world.pathfinding import CostModel, NoPathError, Route, dijkstra

ImageProvider = Callable[[Obstacle], np.ndarray]

ARRIVED, BLOCKED, TIMEOUT = "arrivé", "bloqué", "temps dépassé"


def gray_placeholder(obstacle: Obstacle) -> np.ndarray:
    """Image neutre utilisée tant qu'aucune banque d'obstacles n'est branchée (tests de l'équipe A)."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@dataclass(frozen=True)
class Detection:
    """Dernier obstacle classé : ce que la vue caméra affiche."""

    obstacle: Obstacle
    image_rgb: np.ndarray
    prediction: Prediction
    action: Action


@dataclass
class EpisodeResult:
    episode: int
    classifier: str
    status: str | None                 # ARRIVED, BLOCKED, TIMEOUT ; None tant que l'épisode tourne
    steps: int
    distance_m: float                  # distance réellement parcourue
    planned_length_m: float            # longueur de l'itinéraire initial, sans obstacle
    replans: int
    collisions: int
    waiting_steps: int
    encounters: list[EncounterLog] = field(default_factory=list)


class Episode:
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        classifier: Classifier,
        config: SimConfig,
        episode: int = 0,
        start: int | None = None,
        goal: int | None = None,
        image_for: ImageProvider = gray_placeholder,
    ) -> None:
        from simulator.world.road_map import landmark_node

        self.graph, self.classifier, self.config, self.episode = graph, classifier, config, episode
        self.image_for = image_for
        self.start = start if start is not None else landmark_node(graph, config.start)
        self.goal = goal if goal is not None else landmark_node(graph, config.goal)

        # Même graine pour tous les classifieurs : ils affrontent exactement les mêmes obstacles.
        rng = random.Random(config.seed * 100_003 + episode)
        self.planned: Route = dijkstra(graph, self.start, self.goal)
        self.obstacles = [ObstacleState(o) for o in place_obstacles(graph, self.planned, config.n_obstacles, rng)]

        self.costs = CostModel(graph)
        self.car = Car(graph, self.planned)
        self.step_count = 0
        self.wait = 0
        self.pending_replan = False
        self.no_path_since: int | None = None
        self.slow_edges: set = set()
        self.collided: set[int] = set()
        self.last_detection: Detection | None = None
        self.last_collision_step: int | None = None
        self.result = EpisodeResult(episode, classifier.name, None, 0, 0.0, self.planned.length_m, 0, 0, 0)

    # ------------------------------------------------------------------ boucle
    @property
    def done(self) -> bool:
        return self.result.status is not None

    @property
    def speed_mps(self) -> float:
        """Vitesse actuelle de la voiture (0 à l'arrêt, réduite sur un tronçon où elle ralentit)."""
        if self.done or self.wait > 0 or self.pending_replan:
            return 0.0
        speed = self.config.car_speed_mps
        return speed / self.config.slow_factor if self._in_slow_zone() else speed

    def _in_slow_zone(self) -> bool:
        """Vrai si un obstacle pour lequel on a décidé de ralentir se trouve sur la portée du capteur."""
        for _, state, ahead in self._remaining_obstacles():
            if state.obstacle.edge in self.slow_edges and state.present(self.step_count)                     and ahead <= self.config.sensor_range_m:
                return True
        return self.car.current_edge in self.slow_edges

    def run(self) -> EpisodeResult:
        while not self.done:
            self.step()
        return self.result

    def step(self) -> None:
        if self.done:
            return
        self.step_count += 1
        self.costs.release_expired(self.step_count)

        if self.wait > 0:
            self.wait -= 1
            self.result.waiting_steps += 1
        else:
            self._detect()
            if self.wait == 0 and (not self.pending_replan or self._replan()):
                self._move()

        self._finish_if_needed()

    # ------------------------------------------------------------------ étapes
    def _remaining_obstacles(self):
        """Obstacles situés sur la partie restante de l'itinéraire, avec leur distance devant la voiture."""
        for index in range(self.car.edge_index, len(self.car.route.edges)):
            edge = self.car.route.edges[index]
            for i, state in enumerate(self.obstacles):
                if state.obstacle.edge == edge:
                    distance = self.car.distance_to(index, state.obstacle.offset_m)
                    if distance >= 0:
                        yield i, state, distance

    def _detect(self) -> None:
        for _, state, distance in list(self._remaining_obstacles()):
            if state.handled or distance > self.config.sensor_range_m:
                continue
            obstacle = state.obstacle
            image = self.image_for(obstacle)
            prediction = self.classifier.predict(image, true_label=obstacle.true_label)
            action = self.config.decision_rules[prediction.label]
            self.last_detection = Detection(obstacle, image, prediction, action)  # pour la vue caméra
            state.handled, state.detected_step = True, self.step_count
            state.predicted_label, state.predicted_score = prediction.label, prediction.score
            clear = self.config.clear_steps[obstacle.true_label]
            if clear is not None:
                state.cleared_step = self.step_count + clear

            reaction = apply_action(action, obstacle.edge, self.graph, self.costs, self.config)
            self.wait = max(self.wait, reaction.wait_steps)
            self.pending_replan = self.pending_replan or reaction.replan
            if reaction.slow_edge:
                self.slow_edges.add(obstacle.edge)
            self.result.encounters.append(EncounterLog(
                episode=self.episode, step=self.step_count, classifier=self.classifier.name,
                true_label=obstacle.true_label, predicted_label=prediction.label, score=prediction.score,
                action=action.name, infer_ms=prediction.infer_ms,
            ))

    def _replan(self) -> bool:
        """Recalcule l'itinéraire. Renvoie False s'il n'existe aucun chemin (la voiture attend)."""
        current = self.car.current_edge
        if current is None:
            return True
        # On cherche d'abord un chemin depuis le bout du tronçon actuel. Si l'obstacle est sur ce tronçon,
        # ou si aucune rue ne repart de son extrémité (une seule sortie, bloquée), la voiture fait demi-tour.
        options = [(current[0], True)] if self.costs.is_blocked(current) else [(self.car.next_node, False), (current[0], True)]
        for start, u_turn in options:
            try:
                route = dijkstra(self.graph, start, self.goal, self.costs)
                break
            except NoPathError:
                continue
        else:
            self.no_path_since = self.no_path_since or self.step_count
            self.result.waiting_steps += 1
            return False
        self.car.follow(route, keep_current_edge=not u_turn)
        self.pending_replan, self.no_path_since = False, None
        self.result.replans += 1
        return True

    def _move(self) -> None:
        distance = self.config.car_speed_mps * self.config.step_seconds
        if self._in_slow_zone():
            distance /= self.config.slow_factor

        # Collision : un obstacle encore présent se trouve sur la distance à parcourir pendant ce pas.
        for i, state, ahead in self._remaining_obstacles():
            if i not in self.collided and state.present(self.step_count) and ahead <= distance:
                self.car.advance(max(ahead - 1.0, 0.0))  # la voiture s'arrête contre l'obstacle
                self.collided.add(i)
                self.result.collisions += 1
                self.last_collision_step = self.step_count
                if state.cleared_step is not None:   # obstacle mobile : on attend qu'il libère la voie
                    self.wait = max(self.wait, state.cleared_step - self.step_count)
                else:                                 # obstacle immobile : on le contourne
                    self.costs.block(state.obstacle.edge)
                    self.pending_replan = True
                return
        self.car.advance(distance)

    def _finish_if_needed(self) -> None:
        r = self.result
        r.steps, r.distance_m = self.step_count, self.car.distance_m
        if self.car.arrived:
            r.status = ARRIVED
        elif self.no_path_since is not None and self.step_count - self.no_path_since >= self.config.patience_steps:
            r.status = BLOCKED
        elif self.step_count >= self.config.max_steps:
            r.status = TIMEOUT


def save_encounters_csv(results: list[EpisodeResult], path: Path) -> Path:
    """Écrit toutes les ``EncounterLog`` dans un CSV (lu par metrics.py de l'équipe B)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(log) for r in results for log in r.encounters]
    fields = list(EncounterLog.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="simulation", help="simulation, reference ou ours")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    args = parser.parse_args()

    from simulator.perception import make_classifier, make_image_provider  # fournis par l'équipe B
    from simulator.world.road_map import load_map

    config = SimConfig(seed=args.seed)
    graph = load_map()
    classifier, images = make_classifier(args.classifier, config), make_image_provider(config)
    results = [Episode(graph, classifier, config, e, image_for=images).run() for e in range(args.episodes)]
    out = save_encounters_csv(results, REPORTS_DIR / f"encounters_{args.classifier}_seed{args.seed}.csv")
    arrived = sum(r.status == ARRIVED for r in results)
    print(f"{classifier.name} : {arrived}/{len(results)} arrivés, "
          f"{sum(r.collisions for r in results)} collisions, {sum(r.replans for r in results)} recalculs -> {out}")


if __name__ == "__main__":
    main()
