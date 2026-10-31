"""Trafic d'ambiance (équivalent du Traffic Manager de CARLA) : véhicules, deux-roues, piétons et feux tricolores.

Le trafic est un DÉCOR : il n'a aucun effet sur l'épisode (``engine.py``), ses décisions ni ses métriques.
Le moteur ne connaît pas le trafic ; c'est donc le trafic qui s'adapte à la voiture de l'ego :
- aucun véhicule ne s'engage sur l'itinéraire à venir de l'ego (dans le même sens), ni sur un tronçon qui porte un
  obstacle de l'épisode ;
- l'ego réserve les carrefours qu'il va traverser : le trafic lui cède le passage ;
- un feu passe au vert pour l'approche de l'ego quand il arrive (priorité, comme un véhicule d'essai) ;
- un véhicule qui se retrouve devant l'ego après un recalcul d'itinéraire est replacé ailleurs, hors de vue.

Modèles :
- suivi du véhicule de devant : Intelligent Driver Model (IDM, Treiber et al., 2000) ;
- carrefours sans feu : réservation par le premier arrivé, avec délai maximal (pas de blocage) ;
- feux : deux groupes d'approches (axes perpendiculaires) en alternance vert, orange, rouge intégral ;
- « bulle » autour de l'ego : au-delà de ``radius_m``, un véhicule ou un piéton réapparaît plus près, hors de vue
  (derrière l'ego, ou loin devant), de préférence sur un tronçon proche.

Tout est en Python pur et déterministe pour une graine donnée : testable sans affichage.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

EdgeKey = tuple[int, int, int]

# type : (longueur m, vitesse désirée m/s, poids dans le tirage)
VEHICLE_TYPES: dict[str, tuple[float, float, float]] = {
    "car": (4.4, 11.0, 0.60),
    "bus": (12.0, 8.5, 0.07),
    "truck": (7.5, 9.0, 0.07),
    "motorcycle": (2.1, 12.0, 0.15),
    "bicycle": (1.8, 4.5, 0.11),
}
# IDM : accélération max, décélération confortable, temps inter-véhicules, distance minimale à l'arrêt
IDM_A, IDM_B, IDM_T, IDM_S0, IDM_DELTA = 1.4, 2.5, 1.3, 2.0, 4
LOOK_AHEAD_M = 70.0
JUNCTION_HOLD_S = 3.0          # durée d'une réservation de carrefour
MAX_WAIT_S = 8.0               # au-delà, un véhicule passe quand même (évite tout blocage)
STOP_BEFORE_NODE_M = 5.0       # ligne d'arrêt avant un carrefour (feu ou cédez-le-passage)
EGO_RESERVE_M = 45.0           # l'ego réserve les carrefours qu'il atteindra dans cette distance
EGO_ROUTE_HORIZON_M = 300.0    # le trafic évite l'itinéraire de l'ego sur cette distance
SPAWN_AHEAD_M = 110.0          # devant l'ego, distance minimale pour apparaître (sinon on le verrait surgir)
SPAWN_BEHIND_M = 25.0          # derrière l'ego (hors du champ de la caméra), on peut apparaître bien plus près

GREEN_S, YELLOW_S, ALL_RED_S = 18.0, 3.0, 2.0
PEDESTRIAN_PHASE_S = 12.0
SIGNAL_PRIORITY_M = 75.0

WALK_SPEED = (1.1, 1.5)
SIDEWALK_OFFSET_M = 1.3        # piéton : au milieu du trottoir, au-delà du bord de la chaussée


@dataclass
class EgoState:
    """Ce que le trafic sait de la voiture de l'ego (lu dans l'épisode à chaque pas)."""

    edge: EdgeKey | None
    s: float                       # position de l'avant de l'ego sur ``edge`` (m)
    speed: float
    length: float
    xy: tuple[float, float]
    upcoming: list[tuple[EdgeKey, float]]   # (tronçon, distance depuis l'ego jusqu'au début du tronçon)
    prev_edge: EdgeKey | None = None        # tronçon précédent : l'arrière de l'ego peut encore s'y trouver
    heading: tuple[float, float] = (0.0, 1.0)   # direction de marche (pour apparaître hors du champ de la caméra)


@dataclass
class Vehicle:
    vid: int
    kind: str
    length: float
    v0: float
    edge: EdgeKey
    s: float                       # position de l'AVANT du véhicule sur ``edge``
    v: float
    route: list[EdgeKey] = field(default_factory=list)   # tronçons suivants
    prev_edge: EdgeKey | None = None
    waiting: float = 0.0
    variant: int = 0               # modèle 3D (indice dans les variantes du catalogue)


@dataclass
class Pedestrian:
    pid: int
    edge: EdgeKey                  # rue suivie, parcourue dans le sens u -> v
    s: float
    side: int                      # +1 : trottoir de droite (sens u -> v), -1 : trottoir de gauche
    speed: float
    variant: int = 0
    crossing: float | None = None  # progression 0 -> 1 d'une traversée, sinon None
    cross_duration: float = 0.0


@dataclass
class SignalController:
    """Feu d'un carrefour : deux groupes d'approches en alternance (ou une phase piétons s'il n'y a qu'un axe)."""

    node: int
    groups: tuple[frozenset[EdgeKey], frozenset[EdgeKey]]   # groupe B vide : feu de passage piéton
    phase: int = 0                 # 0 vert A, 1 orange A, 2 rouge intégral, 3 vert B, 4 orange B, 5 rouge intégral
    timer: float = 0.0
    requested: int | None = None   # groupe demandé par l'ego

    def durations(self) -> tuple[float, ...]:
        b_green = GREEN_S if self.groups[1] else PEDESTRIAN_PHASE_S
        return GREEN_S, YELLOW_S, ALL_RED_S, b_green, (YELLOW_S if self.groups[1] else 0.0), ALL_RED_S

    def state(self, edge: EdgeKey) -> str:
        """« green », « yellow » ou « red » pour un tronçon qui arrive au carrefour."""
        group = 0 if edge in self.groups[0] else 1
        if self.phase in (0, 1) and group == 0 or self.phase in (3, 4) and group == 1:
            return "green" if self.phase in (0, 3) else "yellow"
        return "red"

    def step(self, dt: float) -> None:
        self.timer += dt
        durations = self.durations()
        # Priorité à l'ego : le vert de l'autre groupe est écourté.
        if self.requested is not None:
            green_phase = 0 if self.requested == 0 else 3
            other_green = 3 if self.requested == 0 else 0
            if self.phase == other_green and self.timer > 3.0:
                self.timer = durations[self.phase]
            if self.phase == green_phase:
                self.timer = min(self.timer, 1.0)          # le vert est prolongé tant que l'ego approche
        while self.timer >= durations[self.phase]:
            self.timer -= durations[self.phase]
            self.phase = (self.phase + 1) % 6
        self.requested = None


def _first(value):
    return value[0] if isinstance(value, list) and value else value


def _approach_direction(graph: nx.MultiDiGraph, edge: EdgeKey) -> tuple[float, float]:
    from simulator.render3d.world_mesh import edge_line

    u, v, _ = edge
    line = edge_line(graph, u, v, graph.edges[edge])
    end = line.interpolate(line.length)
    before = line.interpolate(max(line.length - 6.0, 0.0))
    dx, dy = end.x - before.x, end.y - before.y
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


def build_signals(graph: nx.MultiDiGraph) -> dict[int, SignalController]:
    """Un contrôleur par nœud OSM « traffic_signals ». Groupes : approches alignées avec la première, puis les autres."""
    signals = {}
    for node, data in graph.nodes(data=True):
        if _first(data.get("highway")) != "traffic_signals":
            continue
        incoming = [(u, v, k) for u, v, k in graph.in_edges(node, keys=True)]
        if not incoming:
            continue
        ref = _approach_direction(graph, incoming[0])
        a, b = [], []
        for edge in incoming:
            d = _approach_direction(graph, edge)
            (a if abs(d[0] * ref[0] + d[1] * ref[1]) > math.cos(math.radians(45)) else b).append(edge)
        signals[node] = SignalController(node, (frozenset(a), frozenset(b)))
    return signals


class TrafficSim:
    def __init__(self, graph: nx.MultiDiGraph, seed: int = 7, n_vehicles: int = 80, n_pedestrians: int = 40,
                 radius_m: float = 200.0, blocked_edges: set[EdgeKey] | None = None) -> None:
        from simulator.render3d.world_mesh import road_width

        self.graph, self.radius_m = graph, radius_m
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.blocked = set(blocked_edges or ())
        self.edges: list[EdgeKey] = [e for e in graph.edges(keys=True) if float(graph.edges[e]["length"]) >= 8.0]
        self.length = {e: float(graph.edges[e]["length"]) for e in graph.edges(keys=True)}
        self.width = {e: road_width(graph.edges[e]) for e in graph.edges(keys=True)}
        mid = np.array([[(graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2, (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2]
                        for u, v, _ in self.edges])
        self.edge_mid = mid
        # Les grands axes ont plus de trafic : poids du tirage proportionnel à la largeur.
        self.edge_weight = np.array([self.width[e] for e in self.edges])
        self.junction = {n for n in graph.nodes if len(set(graph.predecessors(n)) | set(graph.successors(n))) >= 3}
        self.signals = build_signals(graph)
        self.crossing_nodes = {n for n, d in graph.nodes(data=True) if _first(d.get("highway")) == "crossing"}
        self.reservations: dict[int, tuple[object, float]] = {}
        self.ego: EgoState | None = None
        self._ego_edges: set[EdgeKey] = set()
        self.vehicles: list[Vehicle] = []
        self.pedestrians: list[Pedestrian] = []
        self._next_id = 0
        self.n_vehicles, self.n_pedestrians = n_vehicles, n_pedestrians
        self._kinds = list(VEHICLE_TYPES)
        weights = np.array([VEHICLE_TYPES[k][2] for k in self._kinds])
        self._kind_p = weights / weights.sum()

    # ------------------------------------------------------------------ apparition
    def _candidate_edges(self, center: tuple[float, float], min_m: float, max_m: float) -> np.ndarray:
        """Tronçons à moins de ``max_m`` : au moins ``min_m`` devant l'ego, ``SPAWN_BEHIND_M`` derrière lui."""
        rel = self.edge_mid - np.asarray(center)
        d = np.hypot(rel[:, 0], rel[:, 1])
        if self.ego is None:
            return np.flatnonzero((d >= min_m) & (d <= max_m))
        hx, hy = self.ego.heading
        behind = rel[:, 0] * hx + rel[:, 1] * hy < -0.5 * d
        near_enough = np.where(behind, d >= min(min_m, SPAWN_BEHIND_M), d >= min_m)
        return np.flatnonzero(near_enough & (d <= max_m))

    def _proximity(self, indices, center) -> np.ndarray:
        d = np.hypot(self.edge_mid[indices, 0] - center[0], self.edge_mid[indices, 1] - center[1])
        return 1.0 / (1.0 + d / 80.0)

    def _allowed(self, edge: EdgeKey) -> bool:
        return edge not in self.blocked and edge not in self._ego_edges

    def _spawn_vehicle(self, min_m: float) -> Vehicle | None:
        center = self.ego.xy if self.ego else tuple(self.edge_mid.mean(axis=0))
        candidates = [i for i in self._candidate_edges(center, min_m, self.radius_m) if self._allowed(self.edges[i])]
        if not candidates:
            return None
        weights = self.edge_weight[candidates] * self._proximity(candidates, center)
        for _ in range(8):
            edge = self.edges[int(self.rng.choice(candidates, p=weights / weights.sum()))]
            kind = self._kinds[int(self.rng.choice(len(self._kinds), p=self._kind_p))]
            length, v0, _ = VEHICLE_TYPES[kind]
            if self.length[edge] < length + 2.0:                # tronçon trop court pour ce véhicule
                continue
            s = float(self.rng.uniform(length, self.length[edge]))
            if all(o.edge != edge or abs(o.s - s) > 14 for o in self.vehicles):
                vehicle = Vehicle(self._new_id(), kind, length, v0 * float(self.rng.uniform(0.85, 1.1)), edge, s,
                                  v0 * 0.6, variant=int(self.rng.integers(1000)))
                self._extend_route(vehicle)
                return vehicle
        return None

    def _spawn_pedestrian(self, min_m: float) -> Pedestrian | None:
        center = self.ego.xy if self.ego else tuple(self.edge_mid.mean(axis=0))
        candidates = self._candidate_edges(center, min_m, min(self.radius_m, 150.0))
        if not len(candidates):
            return None
        weights = self._proximity(candidates, center)
        edge = self.edges[int(self.rng.choice(candidates, p=weights / weights.sum()))]
        return Pedestrian(self._new_id(), edge, float(self.rng.uniform(0, self.length[edge])),
                          int(self.rng.choice([-1, 1])), float(self.rng.uniform(*WALK_SPEED)),
                          variant=int(self.rng.integers(1000)))

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def populate(self) -> None:
        """Remplit la bulle au démarrage (les véhicules peuvent apparaître plus près, hors de l'itinéraire de l'ego)."""
        for _ in range(self.n_vehicles * 6):                  # plusieurs essais : un tirage peut échouer
            if len(self.vehicles) >= self.n_vehicles:
                break
            vehicle = self._spawn_vehicle(20.0)
            if vehicle is not None:
                self.vehicles.append(vehicle)
        for _ in range(self.n_pedestrians * 6):
            if len(self.pedestrians) >= self.n_pedestrians:
                break
            pedestrian = self._spawn_pedestrian(15.0)
            if pedestrian is not None:
                self.pedestrians.append(pedestrian)

    # ------------------------------------------------------------------ itinéraires
    def _next_edges(self, edge: EdgeKey) -> list[EdgeKey]:
        u, v, _ = edge
        options = [(v, w, k) for w, keys in self.graph[v].items() for k in keys if w != u]
        if not options:                                        # impasse : demi-tour
            options = [(v, w, k) for w, keys in self.graph[v].items() for k in keys]
        return options

    def _extend_route(self, vehicle: Vehicle, horizon: int = 3) -> None:
        last = vehicle.route[-1] if vehicle.route else vehicle.edge
        while len(vehicle.route) < horizon:
            options = [e for e in self._next_edges(last) if self._allowed(e)] or self._next_edges(last)
            weights = np.array([self.width[e] for e in options])
            last = options[int(self.rng.choice(len(options), p=weights / weights.sum()))]
            vehicle.route.append(last)

    # ------------------------------------------------------------------ pas de simulation
    def set_ego(self, ego: EgoState | None) -> None:
        self.ego = ego
        self._ego_edges = set()
        if ego is None:
            return
        if ego.edge is not None:
            self._ego_edges.add(ego.edge)
        for edge, ahead in ego.upcoming:
            if ahead <= EGO_ROUTE_HORIZON_M:
                self._ego_edges.add(edge)
            if ahead <= EGO_RESERVE_M:
                self.reservations[edge[0]] = ("ego", self.time + 1.0)
        # Feu à l'approche de l'ego : demande de vert pour son tronçon d'arrivée.
        approach = [(ego.edge, 0.0)] if ego.edge is not None else []
        for edge, ahead in approach + list(ego.upcoming):
            node = edge[1]
            if node in self.signals and ahead + self.length[edge] - (ego.s if edge == ego.edge else 0) <= SIGNAL_PRIORITY_M:
                controller = self.signals[node]
                controller.requested = 0 if edge in controller.groups[0] else 1
                break

    def step(self, dt: float) -> None:
        self.time += dt
        for controller in self.signals.values():
            controller.step(dt)
        self._clear_vehicles_ahead_of_ego()
        crossings = self._crossing_points()
        by_edge: dict[EdgeKey, list[Vehicle]] = defaultdict(list)
        for vehicle in self.vehicles:
            by_edge[vehicle.edge].append(vehicle)
        for vehicles in by_edge.values():
            vehicles.sort(key=lambda o: o.s)
        accelerations = [self._idm(vehicle, by_edge, crossings) for vehicle in self.vehicles]
        for vehicle, acc in zip(self.vehicles, accelerations):
            vehicle.v = max(0.0, vehicle.v + acc * dt)
            vehicle.waiting = vehicle.waiting + dt if vehicle.v < 0.3 else 0.0
            self._advance(vehicle, vehicle.v * dt)
            # Le carrefour reste réservé tant que le véhicule ne l'a pas dégagé (arrière sorti du carrefour).
            node = vehicle.edge[0]
            if vehicle.prev_edge is not None and node in self.junction and vehicle.s < vehicle.length + 4.0:
                owner, _ = self.reservations.get(node, (None, 0.0))
                if owner != "ego":
                    self.reservations[node] = (vehicle.vid, self.time + 0.5)
        for pedestrian in self.pedestrians:
            self._walk(pedestrian, dt)
        self._recycle()

    def _clear_vehicles_ahead_of_ego(self) -> None:
        """Après un recalcul de l'ego, un véhicule peut se trouver sur son nouvel itinéraire : il est replacé ailleurs."""
        if self.ego is None:
            return
        for i, vehicle in enumerate(self.vehicles):
            ahead = self._ahead_of_ego(vehicle)
            if ahead is not None and 0 <= ahead <= 150.0:
                replacement = self._spawn_vehicle(SPAWN_AHEAD_M)
                if replacement is not None:
                    self.vehicles[i] = replacement

    def _ahead_of_ego(self, vehicle: Vehicle) -> float | None:
        """Distance de l'ego à l'arrière du véhicule, s'il est sur l'itinéraire de l'ego (même sens) ; sinon None."""
        ego = self.ego
        if ego.edge is not None and vehicle.edge == ego.edge:
            return vehicle.s - vehicle.length - ego.s
        for edge, distance in ego.upcoming:
            if edge == vehicle.edge:
                return distance + vehicle.s - vehicle.length
        return None

    def _crossing_points(self) -> dict[EdgeKey, list[float]]:
        """Piétons en train de traverser : points d'arrêt pour les véhicules des deux sens."""
        points: dict[EdgeKey, list[float]] = defaultdict(list)
        for p in self.pedestrians:
            if p.crossing is None:
                continue
            u, v, k = p.edge
            points[p.edge].append(self.length[p.edge] - 3.0)
            for key in self.graph[v].get(u, {}):
                points[(v, u, key)].append(3.0)
        return points

    def _gap(self, vehicle: Vehicle, by_edge, crossings) -> tuple[float, float]:
        """(distance libre devant, vitesse de ce qui est devant) sur ``LOOK_AHEAD_M`` mètres."""
        best = (math.inf, vehicle.v0)

        def consider(gap: float, speed: float) -> None:
            nonlocal best
            if gap < best[0]:
                best = (gap, speed)

        offset = -vehicle.s                     # distance de l'avant du véhicule au début du tronçon examiné
        edges = [vehicle.edge] + vehicle.route
        for index, edge in enumerate(edges):
            if offset > LOOK_AHEAD_M:
                break
            for other in by_edge.get(edge, ()):
                if other is vehicle or (index == 0 and other.s <= vehicle.s):
                    continue
                consider(offset + other.s - other.length, other.v)
                break                                           # le plus proche suffit (liste triée)
            for s in crossings.get(edge, ()):
                if index > 0 or s > vehicle.s:
                    consider(offset + s - 2.0, 0.0)
            ego = self.ego
            if ego is not None and ego.edge == edge and (index > 0 or ego.s > vehicle.s):
                consider(offset + ego.s - ego.length, ego.speed)
            if ego is not None and ego.prev_edge == edge and ego.s - ego.length < 0:
                rear = self.length[edge] + ego.s - ego.length
                if index > 0 or rear > vehicle.s:
                    consider(offset + rear, ego.speed)
            end = offset + self.length[edge]
            node = edge[1]
            if self._must_stop_at(vehicle, edge, node, end):
                consider(end - STOP_BEFORE_NODE_M, 0.0)
                break
            offset = end
        return best

    def _must_stop_at(self, vehicle: Vehicle, edge: EdgeKey, node: int, distance: float) -> bool:
        """Feu rouge (ou orange si on peut encore s'arrêter), ou carrefour réservé par un autre."""
        if node in self.signals:
            state = self.signals[node].state(edge)
            if state == "red":
                return vehicle.waiting < MAX_WAIT_S * 4
            if state == "yellow":
                return distance - STOP_BEFORE_NODE_M > vehicle.v ** 2 / (2 * IDM_B)
            return False
        if node not in self.junction or distance > 25.0:
            return False
        owner, until = self.reservations.get(node, (None, 0.0))
        free = owner is None or owner == vehicle.vid or until < self.time
        if free or (owner != "ego" and vehicle.waiting > 3 * MAX_WAIT_S):   # anti-blocage, très rare
            if distance < 12.0:
                self.reservations[node] = (vehicle.vid, self.time + JUNCTION_HOLD_S)
            return False
        return True

    def _idm(self, vehicle: Vehicle, by_edge, crossings) -> float:
        gap, leader_speed = self._gap(vehicle, by_edge, crossings)
        v, v0 = vehicle.v, vehicle.v0
        free = 1 - (v / v0) ** IDM_DELTA
        if math.isinf(gap):
            return IDM_A * free
        dv = v - leader_speed
        s_star = IDM_S0 + max(0.0, v * IDM_T + v * dv / (2 * math.sqrt(IDM_A * IDM_B)))
        return max(-9.0, IDM_A * (free - (s_star / max(gap, 0.1)) ** 2))

    def _advance(self, vehicle: Vehicle, distance: float) -> None:
        vehicle.s += distance
        while vehicle.s > self.length[vehicle.edge]:
            vehicle.s -= self.length[vehicle.edge]
            vehicle.prev_edge = vehicle.edge
            vehicle.edge = vehicle.route.pop(0)
            self._extend_route(vehicle)

    def _walk(self, p: Pedestrian, dt: float) -> None:
        if p.crossing is not None:
            p.crossing += dt / p.cross_duration
            if p.crossing >= 1.0:                            # arrivé en face : il repart par une autre rue
                p.crossing, p.side = None, -p.side
                p.s = self.length[p.edge]
            return
        p.s += p.speed * dt
        length = self.length[p.edge]
        if p.s < length - 3.0 or (p.s < length and p.edge[1] not in self.crossing_nodes):
            return
        if p.edge[1] in self.crossing_nodes and p.s < length and self.rng.random() < 0.6:
            p.s = length - 3.0                               # traverse au passage piéton, au bout de la rue
            p.crossing = 0.0
            p.cross_duration = (self.width[p.edge] + 2 * SIDEWALK_OFFSET_M) / p.speed
            return
        if p.s < length:
            return
        options = self._next_edges(p.edge)
        p.edge = options[int(self.rng.integers(len(options)))]
        p.s = 0.0

    def _recycle(self) -> None:
        if self.ego is None:
            return
        ex, ey = self.ego.xy
        for i, vehicle in enumerate(self.vehicles):
            if self._distance(vehicle.edge, ex, ey) > self.radius_m * 1.2:
                replacement = self._spawn_vehicle(SPAWN_AHEAD_M)
                if replacement is not None:
                    self.vehicles[i] = replacement
        for i, pedestrian in enumerate(self.pedestrians):
            if self._distance(pedestrian.edge, ex, ey) > 180.0:
                replacement = self._spawn_pedestrian(70.0)
                if replacement is not None:
                    self.pedestrians[i] = replacement

    def _distance(self, edge: EdgeKey, x: float, y: float) -> float:
        u = self.graph.nodes[edge[0]]
        return math.hypot(u["x"] - x, u["y"] - y)

    # ------------------------------------------------------------------ lecture pour l'affichage
    def signal_state(self, edge: EdgeKey) -> str | None:
        controller = self.signals.get(edge[1])
        return controller.state(edge) if controller is not None else None


def ego_state(episode, ego_length: float) -> EgoState:
    """État de l'ego lu dans un ``Episode`` (sans le modifier).

    L'ego affiché est interpolé entre deux pas du moteur : il a jusqu'à un pas de retard sur l'ego du moteur.
    Sa longueur est donc augmentée de la distance d'un pas, pour que le trafic ne rattrape jamais l'ego visible.
    """
    car = episode.car
    length = ego_length + episode.config.car_speed_mps * episode.config.step_seconds
    upcoming, ahead = [], 0.0
    if not car.arrived:
        ahead = car.edge_length(car.route.edges[car.edge_index]) - car.progress_m
        for edge in car.route.edges[car.edge_index + 1:]:
            upcoming.append((edge, ahead))
            ahead += car.edge_length(edge)
            if ahead > EGO_ROUTE_HORIZON_M:
                break
    previous = car.route.edges[car.edge_index - 1] if 0 < car.edge_index <= len(car.route.edges) else (
        car.driven_edges[-1] if car.driven_edges else None)
    return EgoState(car.current_edge, car.progress_m, episode.speed_mps, length, car.position_xy(), upcoming, previous,
                    car.heading())
