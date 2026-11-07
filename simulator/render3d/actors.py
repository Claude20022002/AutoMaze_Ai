"""Acteurs 3D synchronisés avec un ``Episode`` : voiture de l'ego, obstacles, étiquettes et itinéraire.

Les acteurs LISENT l'état de l'épisode et ne le modifient jamais. Le comportement affiché suit le moteur :
- une personne traverse la chaussée entre sa détection et le moment où elle libère la voie (``cleared_step``) ;
- un vélo ou une moto repart quand il libère la voie ;
- une voiture, un bus ou un camion reste immobile (le moteur le fait contourner).

Le moteur traite la voiture et les obstacles comme des points. À l'affichage, le point de la voiture est son
pare-chocs avant, et le point d'un véhicule obstacle est son arrière : quand le moteur arrête la voiture 1 m
avant l'obstacle, les deux modèles ne se chevauchent pas.
"""

from __future__ import annotations

import math

import networkx as nx

from simulator.contracts import CLASSES_FR, UNKNOWN
from simulator.engine import Episode
from simulator.render3d.catalog import Catalog, load_model
from simulator.render3d.masks import no_shadow, only_on_main_view
from simulator.render3d.world_mesh import (DRAW_LAG_M, ROAD_Z, WorldFrame, heading_deg, is_two_way, lane_offset,
                                           road_width)
from simulator.world.car import Car, point_on_edge
from simulator.world.obstacles import ObstacleState

LEAVE_SPEED_MPS = {"bicycle": 4.0, "motorcycle": 8.0}   # vitesse à laquelle un deux-roues repart
LABEL_DISTANCE_M = 90.0                                  # au-delà, l'étiquette d'un obstacle est masquée

# Mêmes couleurs que ui/theme.py (recopiées : la vue 3D n'importe pas pygame).
GOOD, WRONG, WARNING, QUIET = (0.18, 0.55, 0.29), (0.81, 0.24, 0.19), (0.87, 0.53, 0.10), (0.35, 0.35, 0.35)


def verdict(predicted: str, true_label: str) -> tuple[str, tuple[float, float, float]]:
    """Résultat écrit en toutes lettres (la couleur seule ne suffit pas), comme la vue caméra 2D."""
    if predicted == UNKNOWN:
        return "inconnu", WARNING
    if predicted == true_label:
        return "juste", GOOD
    return f"faux · vrai : {CLASSES_FR[true_label]}", WRONG


def edge_pose(graph: nx.MultiDiGraph, edge, offset_m: float) -> tuple[float, float, float, float]:
    """Point (x, y) en coordonnées de la carte et direction unitaire (dx, dy) du tronçon à ``offset_m``."""
    length = float(graph.edges[edge]["length"])
    x, y = point_on_edge(graph, edge, offset_m)
    ax, ay = point_on_edge(graph, edge, max(offset_m - 2.0, 0.0))
    bx, by = point_on_edge(graph, edge, min(offset_m + 2.0, length))
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy) or 1.0
    return x, y, dx / norm, dy / norm


def lane_point(frame: WorldFrame, x: float, y: float, dx: float, dy: float, lateral_m: float) -> tuple[float, float]:
    """Point local décalé de ``lateral_m`` vers la droite du sens de marche (dx, dy)."""
    return frame.to_local(x + dy * lateral_m, y - dx * lateral_m)


def driving_lateral(graph: nx.MultiDiGraph, edge, backwards: bool) -> float:
    """Décalage vers la droite du sens de marche sur un tronçon. En marche à contresens d'un sens unique
    (retour après un demi-tour), la voiture roule au milieu de la chaussée."""
    if edge is None:
        return 0.0
    data = graph.edges[edge]
    return 0.0 if backwards and not is_two_way(data) else lane_offset(data)


def ego_body(graph: nx.MultiDiGraph, car: Car, distance_m: float, length_m: float,
             smooth_m: float = DRAW_LAG_M) -> tuple[tuple[float, float], tuple[float, float]]:
    """Centre (x, y) et cap unitaire de la carrosserie quand le moteur a parcouru ``distance_m`` mètres.

    L'avant et l'arrière sont posés sur la trace réelle de la voiture (``Car.trail``), chacun décalé sur sa voie :
    la carrosserie suit les virages sans couper les trottoirs et tourne pendant un demi-tour au lieu de sauter.
    Le cap et le décalage de voie sont moyennés sur ± ``smooth_m`` mètres. L'avant est donc dessiné ``DRAW_LAG_M``
    derrière le moteur : ces moyennes ne lisent que du chemin déjà connu, sinon un demi-tour décidé au pas suivant
    ferait sauter la voiture de côté.
    """
    distance_m -= DRAW_LAG_M
    lanes: dict = {}

    def lateral_at(index: int) -> float:
        """Décalage du segment de trace qui finit au point ``index`` (tronçon et sens de ce point)."""
        point = car.trail[index]
        key = (point[3], point[4])
        if key not in lanes:
            lanes[key] = driving_lateral(graph, point[3], point[4])
        return lanes[key]

    def mean_lateral(s: float) -> float:
        """Moyenne exacte du décalage sur [s - smooth_m, s + smooth_m] : il varie continûment, sans marche."""
        trail = car.trail
        lo, hi = s - smooth_m, s + smooth_m
        total = weight = 0.0
        for i in range(1, len(trail)):
            a, b = max(trail[i - 1][0], lo), min(trail[i][0], hi)
            if b > a:
                total += lateral_at(i) * (b - a)
                weight += b - a
        if weight < 1e-6:                      # hors de la trace (tout début ou bout) : point le plus proche
            return lateral_at(len(trail) - 1 if s > trail[-1][0] else min(1, len(trail) - 1))
        return total / weight

    if not car.trail:
        car.trail_point(distance_m)            # premier point de la trace

    # Cap de chaque segment de la trace, « déroulé » : un retournement (demi-tour, virage en épingle) compte
    # pour +180° (on tourne par la gauche), jamais pour un saut de -180° à +180°.
    trail = car.trail
    hx0, hy0 = car.heading()
    angles = [math.atan2(hy0, hx0)]
    for i in range(1, len(trail)):
        dx, dy = trail[i][1] - trail[i - 1][1], trail[i][2] - trail[i - 1][2]
        if math.hypot(dx, dy) < 1e-6:
            angles.append(angles[-1])
            continue
        delta = (math.atan2(dy, dx) - angles[-1] + math.pi) % (2 * math.pi) - math.pi
        if delta < -0.9 * math.pi:
            delta += 2 * math.pi
        angles.append(angles[-1] + delta)

    def mean_angle(s: float, window: float = DRAW_LAG_M) -> float:
        """Cap moyen sur [s - window, s + window] : il tourne continûment, même pendant un demi-tour."""
        total = weight = 0.0
        for i in range(1, len(trail)):
            a, b = max(trail[i - 1][0], s - window), min(trail[i][0], s + window)
            if b > a:
                total += angles[i] * (b - a)
                weight += b - a
        if weight < 1e-6:
            return angles[-1] if s > trail[-1][0] else angles[min(1, len(angles) - 1)]
        return total / weight

    def lane_side(s: float) -> tuple[float, float]:
        _, x, y, _, _ = car.trail_point(s)
        angle = mean_angle(s)
        dx, dy = math.cos(angle), math.sin(angle)
        lateral = mean_lateral(s)
        return x + dy * lateral, y - dx * lateral

    fx, fy = lane_side(distance_m)
    first = car.trail[0][0]
    rx, ry = lane_side(max(distance_m - length_m, first))
    missing = first - (distance_m - length_m)
    if missing > 0:                            # début d'épisode : l'arrière est avant le début de la trace,
        ax, ay = lane_side(first + 2.0)        # on le prolonge dans l'axe du départ
        dx, dy = ax - rx, ay - ry
        norm = math.hypot(dx, dy)
        dx, dy = (dx / norm, dy / norm) if norm > 1e-6 else car.heading()
        rx, ry = rx - dx * missing, ry - dy * missing
    dx, dy = fx - rx, fy - ry
    norm = math.hypot(dx, dy) or 1.0
    return ((fx + rx) / 2, (fy + ry) / 2), (dx / norm, dy / norm)


class EgoCar:
    def __init__(self, parent, catalog: Catalog, loader, frame: WorldFrame) -> None:
        self.frame = frame
        self.length = catalog.ego.length_m
        self.node = load_model(catalog.ego, loader)
        self.node.reparentTo(parent)

    def place(self, center: tuple[float, float], heading: tuple[float, float]) -> None:
        """``center`` : centre de la carrosserie déjà posé sur sa voie (``ego_body``), coordonnées de la carte."""
        (x, y), (hx, hy) = center, heading
        lx, ly = self.frame.to_local(x, y)
        self.node.setPos(lx, ly, ROAD_Z)
        self.node.setH(heading_deg(hx, hy))


class ObstacleView:
    """Un obstacle : son modèle 3D (choisi avec sa graine) et son étiquette de classification."""

    def __init__(self, parent, state: ObstacleState, graph: nx.MultiDiGraph, catalog: Catalog, loader,
                 frame: WorldFrame, seed: int, font=None, minimap=None) -> None:
        from panda3d.core import TextNode

        obstacle = state.obstacle
        self.state, self.graph, self.frame = state, graph, frame
        self.label = obstacle.true_label
        u, v, key = obstacle.edge
        self.entry = catalog.pick(self.label, [seed, u, v, key, round(obstacle.offset_m * 1000)])
        self.root = parent.attachNewNode(f"obstacle_{self.label}")
        self.model = load_model(self.entry, loader)
        self.model.reparentTo(self.root)
        self.actor = self.model.getPythonTag("actor")
        self.animation: str | None = None
        self.half_width = road_width(graph.edges[obstacle.edge]) / 2
        self.lane = lane_offset(graph.edges[obstacle.edge])     # même voie que l'ego, qui s'arrête derrière
        self.edge_length = float(graph.edges[obstacle.edge]["length"])

        text = TextNode("etiquette")
        text.setAlign(TextNode.ACenter)
        text.setCardColor(1, 1, 1, 0.85)
        text.setCardAsMargin(0.3, 0.3, 0.15, 0.1)
        text.setCardDecal(True)
        if font is not None:
            text.setFont(font)
        self.text = text
        top = self.model.getTightBounds()[1].z + 0.9      # au-dessus du modèle, quelle que soit sa hauteur
        self.tag = self.root.attachNewNode(text)
        self.tag.setPos(0, 0, top)
        self.tag.setScale(0.55)
        self.tag.setBillboardPointEye()
        self.tag.setLightOff()
        self.tag.setDepthTest(False)
        self.tag.setBin("fixed", 10)
        no_shadow(self.tag)
        self._shown_text = None
        self.marker = minimap.obstacle_marker(self.root, QUIET) if minimap is not None else None
        if minimap is not None:
            only_on_main_view(self.tag)

    def _play(self, name: str) -> None:
        if self.actor is not None and self.animation != name:
            self.actor.loop(self.entry.animations[name])
            self.animation = name

    def update(self, t: float, camera_pos) -> None:
        """``t`` : temps de l'épisode en pas (fractionnaire, pour un mouvement fluide)."""
        state, obstacle = self.state, self.state.obstacle
        offset = obstacle.offset_m
        if self.entry.length_m is not None:
            offset += self.entry.length_m / 2               # le point du moteur = arrière du véhicule
        lateral, extra_heading = self.lane, 0.0
        detected, cleared = state.detected_step, state.cleared_step
        visible = True
        if self.label == "person":
            extra_heading = 90.0                         # traverse vers la gauche du sens de circulation
            far_side = -(self.half_width + 1.2)
            if detected is not None and cleared is not None and cleared > detected:
                fraction = min(max((t - detected) / (cleared - detected), 0.0), 1.0)
                lateral = self.lane + (far_side - self.lane) * fraction
                self._play("walk" if 0.0 < fraction < 1.0 else "idle")
            else:
                self._play("idle")
        elif self.label in LEAVE_SPEED_MPS and cleared is not None and t > cleared:
            offset += (t - cleared) * LEAVE_SPEED_MPS[self.label]
            visible = offset < self.edge_length

        if not visible:
            self.root.hide()
            return
        self.root.show()
        x, y, dx, dy = edge_pose(self.graph, obstacle.edge, min(offset, self.edge_length))
        lx, ly = lane_point(self.frame, x, y, dx, dy, lateral)
        self.root.setPos(lx, ly, ROAD_Z)
        self.model.setH(heading_deg(dx, dy) + extra_heading)
        self._update_tag(camera_pos)

    def _update_tag(self, camera_pos) -> None:
        state = self.state
        if (self.root.getPos() - camera_pos).length() > LABEL_DISTANCE_M:
            self.tag.hide()
        else:
            self.tag.show()
        if state.predicted_label is None:
            content, color = "?", QUIET
        else:
            name = CLASSES_FR.get(state.predicted_label, "Inconnu")
            word, color = verdict(state.predicted_label, self.label)
            content = f"{name} {state.predicted_score:.2f}\n{word}"
        if content != self._shown_text:
            self.text.setText(content)
            self.text.setTextColor(*color, 1)
            self._shown_text = content
            if self.marker is not None:
                self.marker.setColor(*color, 1)

    def destroy(self) -> None:
        if self.actor is not None:
            self.actor.cleanup()
        self.root.removeNode()


class RouteLine:
    """Itinéraire restant, dessiné au sol (comme le débogage de CARLA). Redessiné quand l'itinéraire change."""

    def __init__(self, parent, graph: nx.MultiDiGraph, frame: WorldFrame) -> None:
        self.parent, self.graph, self.frame = parent, graph, frame
        self.node = None
        self._key = None

    def update(self, episode: Episode) -> None:
        from panda3d.core import LineSegs

        car = episode.car
        key = (id(car.route), car.edge_index)
        if key == self._key:
            return
        self._key = key
        if self.node is not None:
            self.node.removeNode()
        lines = LineSegs("itineraire")
        lines.setThickness(4)
        lines.setColor(0.2, 0.55, 0.95, 1)
        for edge in car.route.edges[car.edge_index:]:
            length = float(self.graph.edges[edge]["length"])
            steps = max(2, int(length / 5) + 1)
            lane = lane_offset(self.graph.edges[edge])
            for i in range(steps):
                x, y, dx, dy = edge_pose(self.graph, edge, length * i / (steps - 1))
                lx, ly = lane_point(self.frame, x, y, dx, dy, lane)
                (lines.moveTo if i == 0 else lines.drawTo)(lx, ly, ROAD_Z + 0.08)
        self.node = self.parent.attachNewNode(lines.create())
        self.node.setLightOff()
        self.node.setDepthOffset(3)
        no_shadow(self.node)
