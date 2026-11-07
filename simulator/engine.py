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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from simulator.config import REPORTS_DIR, SimConfig
from simulator.contracts import CLASSES_FR, Action, Classifier, EncounterLog, Obstacle, Prediction
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


@dataclass(frozen=True)
class Stop:
    """Arrêt prévu sur le trajet (déposer quelqu'un, faire une course) : la voiture y reste ``dwell_steps`` pas."""

    node: int
    dwell_steps: int
    name: str = "Arrêt"


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
    waiting_steps: int                 # attente due aux obstacles ou à l'absence de chemin (hors arrêts prévus)
    encounters: list[EncounterLog] = field(default_factory=list)
    u_turns: int = 0                   # demi-tours (obstacle sur la rue où roule la voiture)
    stops_reached: list[tuple[str, int]] = field(default_factory=list)  # (arrêt prévu, pas d'arrivée)


def join_routes(legs: Sequence[Route]) -> Route:
    """Met bout à bout les étapes d'un trajet (chaque étape part de l'arrivée de la précédente)."""
    nodes: tuple[int, ...] = legs[0].nodes
    for leg in legs[1:]:
        nodes += leg.nodes[1:]
    return Route(nodes=nodes, edges=tuple(e for leg in legs for e in leg.edges),
                 cost=sum(leg.cost for leg in legs), length_m=sum(leg.length_m for leg in legs))


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
        obstacles: Sequence[Obstacle] | None = None,
        stops: Sequence[Stop] = (),
    ) -> None:
        """``obstacles`` impose les obstacles (scénarios) ; sinon ils sont tirés au hasard sur l'itinéraire.
        ``stops`` : arrêts prévus, dans l'ordre, entre le départ et l'arrivée (trajets quotidiens)."""
        from simulator.world.road_map import landmark_node

        self.graph, self.classifier, self.config, self.episode = graph, classifier, config, episode
        self.image_for = image_for
        self.start = start if start is not None else landmark_node(graph, config.start)
        self.goal = goal if goal is not None else landmark_node(graph, config.goal)
        self.stops: tuple[Stop, ...] = tuple(stops)
        self.next_stop = 0                    # indice du prochain arrêt prévu
        self.dwell = 0                        # pas restants à l'arrêt prévu en cours
        self.leg_pending = False              # arrêt terminé : l'étape suivante reste à calculer

        # Même graine pour tous les classifieurs : ils affrontent exactement les mêmes obstacles.
        rng = random.Random(config.seed * 100_003 + episode)
        targets = [s.node for s in self.stops] + [self.goal]
        legs = [dijkstra(graph, a, b) for a, b in zip([self.start, *targets[:-1]], targets)]
        self.planned: Route = join_routes(legs)
        self.planned_legs: tuple[Route, ...] = tuple(legs)   # étapes prévues (distance restante affichée)
        if obstacles is None:
            obstacles = place_obstacles(graph, self.planned, config.n_obstacles, rng)
        self.obstacles = [ObstacleState(o) for o in obstacles]

        self.costs = CostModel(graph)
        self.car = Car(graph, legs[0])
        self.step_count = 0
        self.wait = 0
        self.pending_replan = False
        self.no_path_since: int | None = None
        self.slow_edges: set = set()
        self.collided: set[int] = set()
        self.last_detection: Detection | None = None
        self.last_collision_step: int | None = None
        self.events: list[tuple[int, str]] = []   # (pas, ce qui vient de se passer) : affiché par les vues
        self.result = EpisodeResult(episode, classifier.name, None, 0, 0.0, self.planned.length_m, 0, 0, 0)

    # ------------------------------------------------------------------ boucle
    @property
    def done(self) -> bool:
        return self.result.status is not None

    @property
    def speed_mps(self) -> float:
        """Vitesse actuelle de la voiture (0 à l'arrêt, réduite sur un tronçon où elle ralentit)."""
        if self.done or self.wait > 0 or self.pending_replan or self.dwell > 0 or self.leg_pending:
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
        elif self.dwell > 0:                  # arrêt prévu (dépose, course) : ce n'est pas une attente subie
            self.dwell -= 1
        elif not self.leg_pending or self._start_next_leg():
            self._detect()
            if self.wait == 0 and (not self.pending_replan or self._replan()):
                self._move()

        self._finish_if_needed()

    def _event(self, text: str) -> None:
        self.events.append((self.step_count, text))

    @property
    def at_stop(self) -> str | None:
        """Nom de l'arrêt prévu où la voiture se trouve en ce moment, sinon None."""
        return self.stops[self.next_stop - 1].name if self.leg_pending and not self.done else None

    @property
    def target(self) -> int:
        """Destination de l'étape en cours : le prochain arrêt prévu, ou l'arrivée."""
        return self.stops[self.next_stop].node if self.next_stop < len(self.stops) else self.goal

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
            self._event(f"Obstacle à {distance:.0f} m classé « {prediction.label_fr} » : {action.value}")
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
                route = dijkstra(self.graph, start, self.target, self.costs)
                break
            except NoPathError:
                continue
        else:
            if self.no_path_since is None:
                self._event("Aucun chemin libre : la voiture attend")
            self.no_path_since = self.no_path_since or self.step_count
            self.result.waiting_steps += 1
            return False
        self.car.follow(route, keep_current_edge=not u_turn)
        self.pending_replan, self.no_path_since = False, None
        self.result.replans += 1
        self.result.u_turns += u_turn
        self._event("Demi-tour : retour au carrefour, puis nouveau chemin" if u_turn
                    else "Nouveau chemin calculé par Dijkstra")
        return True

    def _start_next_leg(self) -> bool:
        """Après un arrêt prévu : calcule l'étape suivante, avec les blocages connus. False s'il n'y a aucun chemin."""
        here = self.car.route.nodes[-1]
        try:
            route = dijkstra(self.graph, here, self.target, self.costs)
        except NoPathError:
            self.no_path_since = self.no_path_since or self.step_count
            self.result.waiting_steps += 1
            return False
        self.car.start_leg(route)
        self.leg_pending, self.no_path_since = False, None
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
                self._event(f"Collision : « {CLASSES_FR[state.obstacle.true_label]} » mal classé")
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
        if self.car.arrived and not self.leg_pending and self.next_stop < len(self.stops):
            stop = self.stops[self.next_stop]                  # arrêt prévu atteint : la voiture y reste un moment
            r.stops_reached.append((stop.name, self.step_count))
            self.next_stop += 1
            self.dwell, self.leg_pending = stop.dwell_steps, True
            self.pending_replan = False                        # l'étape suivante sera calculée au départ
            self._event(f"Arrêt prévu : {stop.name}")
        elif self.car.arrived and not self.leg_pending:
            r.status = ARRIVED
            self._event("Arrivée à destination")
        elif self.no_path_since is not None and self.step_count - self.no_path_since >= self.config.patience_steps:
            r.status = BLOCKED
            self._event("Bloqué : aucun chemin vers la destination")
        elif self.step_count >= self.config.max_steps:
            r.status = TIMEOUT
            self._event("Temps dépassé")

    def recent_event(self, within_steps: int = 4) -> str | None:
        """Dernier événement s'il date de moins de ``within_steps`` pas (bandeau des vues)."""
        if self.events and self.step_count - self.events[-1][0] <= within_steps:
            return self.events[-1][1]
        return None


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
    classifier = make_classifier(args.classifier, config)
    results = [
        Episode(graph, classifier, config, e, image_for=make_image_provider(config, episode=e)).run()
        for e in range(args.episodes)
    ]
    out = save_encounters_csv(results, REPORTS_DIR / f"encounters_{args.classifier}_seed{args.seed}.csv")
    arrived = sum(r.status == ARRIVED for r in results)
    print(f"{classifier.name} : {arrived}/{len(results)} arrivés, "
          f"{sum(r.collisions for r in results)} collisions, {sum(r.replans for r in results)} recalculs -> {out}")


if __name__ == "__main__":
    main()
