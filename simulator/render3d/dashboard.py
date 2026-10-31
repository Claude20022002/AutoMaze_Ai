"""Tableau de bord de la vue conducteur : compteur, décision en cours, objet devant, itinéraire.

Inspiré des combinés d'instruments et des affichages tête haute des simulateurs de conduite (compteur analogique,
distance au véhicule devant, consigne en cours). Les fonctions ``driving_state``, ``object_ahead`` et
``remaining_m`` ne lisent que l'épisode : elles se testent sans affichage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from simulator.contracts import CLASSES_FR, Action
from simulator.engine import ARRIVED, BLOCKED, TIMEOUT, Episode

MAX_SPEED_KMH = 60
LOOK_AHEAD_M = 80.0

GREEN, ORANGE, RED, GREY = (0.30, 0.78, 0.42), (0.98, 0.66, 0.20), (0.94, 0.33, 0.27), (0.62, 0.64, 0.66)


@dataclass(frozen=True)
class DrivingState:
    text: str
    color: tuple[float, float, float]


def driving_state(episode: Episode) -> DrivingState:
    """Ce que fait la voiture en ce moment, en une ligne (voyant du tableau de bord)."""
    status = episode.result.status
    if status == ARRIVED:
        return DrivingState("Arrivée", GREEN)
    if status in (BLOCKED, TIMEOUT):
        return DrivingState(f"Épisode terminé : {status}", RED)
    detection = episode.last_detection
    if episode.wait > 0:
        if detection is not None and detection.action == Action.STOP_WAIT:
            return DrivingState(f"Arrêt : piéton ({episode.wait} s)", RED)
        if detection is not None and detection.action == Action.CAUTIOUS_STOP:
            return DrivingState(f"Arrêt prudent : objet inconnu ({episode.wait} s)", ORANGE)
        return DrivingState(f"Arrêt ({episode.wait} s)", RED)
    if episode.pending_replan:
        return DrivingState("Recalcul de l'itinéraire", ORANGE)
    if episode.speed_mps < episode.config.car_speed_mps - 1e-6:
        return DrivingState("Ralentissement", ORANGE)
    return DrivingState("Voie libre", GREEN)


@dataclass(frozen=True)
class ObjectAhead:
    distance_m: float
    label: str | None        # classe prédite, None tant que l'objet n'est pas classé
    score: float | None


def object_ahead(episode: Episode, look_ahead_m: float = LOOK_AHEAD_M) -> ObjectAhead | None:
    """Obstacle encore présent le plus proche devant la voiture, sur son itinéraire."""
    car, step = episode.car, episode.step_count
    best: ObjectAhead | None = None
    for index in range(car.edge_index, len(car.route.edges)):
        edge = car.route.edges[index]
        for state in episode.obstacles:
            if state.obstacle.edge != edge or not state.present(step):
                continue
            distance = car.distance_to(index, state.obstacle.offset_m)
            if 0 <= distance <= look_ahead_m and (best is None or distance < best.distance_m):
                best = ObjectAhead(distance, state.predicted_label, state.predicted_score)
    return best


def remaining_m(episode: Episode) -> float:
    car = episode.car
    if car.arrived:
        return 0.0
    edges = car.route.edges[car.edge_index:]
    return sum(car.edge_length(e) for e in edges) - car.progress_m


def object_text(ahead: ObjectAhead | None) -> str:
    if ahead is None:
        return "Aucun objet devant"
    if ahead.label is None:
        return f"Objet à {ahead.distance_m:.0f} m · non classé"
    return f"{CLASSES_FR.get(ahead.label, 'Inconnu')} {ahead.score:.2f} à {ahead.distance_m:.0f} m"


class Dashboard:
    """Bandeau du bas de la vue conducteur, dessiné dans ``pixel2d`` (unités : pixels, z vers le haut)."""

    def __init__(self, base, font, sensor_range_m: float) -> None:
        from direct.gui.DirectGui import DirectFrame
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import LineSegs, TextNode

        self.base, self.sensor_range_m = base, sensor_range_m
        self.root = base.pixel2d.attachNewNode("tableau_de_bord")
        self.frame = DirectFrame(parent=self.root, frameColor=(0.10, 0.11, 0.13, 1), frameSize=(0, 1, 0, 1))

        def text(pos, scale, fg=(0.92, 0.93, 0.95, 1), align=TextNode.ALeft):
            return OnscreenText(parent=self.root, text="", pos=pos, scale=scale, fg=fg, align=align, font=font,
                                mayChange=True)

        # Compteur : arc de 0 à 60 km/h, graduations tous les 10 km/h
        self.gauge = self.root.attachNewNode("compteur")
        arc = LineSegs("cadran")
        arc.setThickness(3)
        arc.setColor(0.75, 0.77, 0.80, 1)
        for i in range(61):
            angle = self._angle(i)
            x, z = 70 * math.sin(math.radians(angle)), 70 * math.cos(math.radians(angle))
            (arc.moveTo if i == 0 else arc.drawTo)(x, 0, z)
        for kmh in range(0, MAX_SPEED_KMH + 1, 10):
            a = math.radians(self._angle(kmh))
            arc.moveTo(62 * math.sin(a), 0, 62 * math.cos(a))
            arc.drawTo(72 * math.sin(a), 0, 72 * math.cos(a))
        self.gauge.attachNewNode(arc.create())
        for kmh in range(0, MAX_SPEED_KMH + 1, 20):
            a = math.radians(self._angle(kmh))
            OnscreenText(parent=self.gauge, text=str(kmh), pos=(48 * math.sin(a), 48 * math.cos(a) - 4), scale=12,
                         fg=(0.7, 0.72, 0.75, 1), font=font)
        needle = LineSegs("aiguille")
        needle.setThickness(4)
        needle.setColor(0.98, 0.45, 0.20, 1)
        needle.moveTo(0, 0, 0)
        needle.drawTo(0, 0, 60)
        self.needle = self.gauge.attachNewNode(needle.create())
        self.speed = text((0, -30), 24, align=TextNode.ACenter)
        self.speed.reparentTo(self.gauge)
        self.unit = text((0, -46), 12, fg=(0.6, 0.62, 0.66, 1), align=TextNode.ACenter)
        self.unit.reparentTo(self.gauge)
        self.unit.setText("km/h")

        # Voyant de décision et objet devant
        self.lamp = DirectFrame(parent=self.root, frameColor=(*GREEN, 1), frameSize=(0, 14, -14, 0))
        self.state = text((0, 0), 22)
        self.ahead = text((0, 0), 17)
        self.bar_back = DirectFrame(parent=self.root, frameColor=(0.22, 0.23, 0.26, 1), frameSize=(0, 1, -8, 0))
        self.bar = DirectFrame(parent=self.root, frameColor=(*GREY, 1), frameSize=(0, 1, -8, 0))
        self.sensor_mark = DirectFrame(parent=self.root, frameColor=(0.9, 0.9, 0.9, 1), frameSize=(-1, 1, -14, 6))
        self.bar_caption = text((0, 0), 12, fg=(0.6, 0.62, 0.66, 1))
        self.route = text((0, 0), 15, fg=(0.80, 0.82, 0.85, 1), align=TextNode.ARight)
        self.layout(0, 0, 800, 200)

    @staticmethod
    def _angle(kmh: float) -> float:
        return -120 + 240 * min(max(kmh, 0), MAX_SPEED_KMH) / MAX_SPEED_KMH

    def layout(self, x: float, y_top: float, width: float, height: float) -> None:
        """Place le bandeau : (x, y_top) coin haut gauche en pixels depuis le haut de la fenêtre."""
        self.root.setPos(x, 0, -y_top)
        self.frame["frameSize"] = (0, width, -height, 0)
        mid = -height / 2
        self.gauge.setPos(110, 0, mid + 6)
        left = 240
        self.lamp.setPos(left, 0, mid + 52)
        self.state.setPos(left + 26, mid + 39)
        self.ahead.setPos(left, mid + 2)
        self.bar_width = max(120, min(420, width - left - 260))
        self.bar_back.setPos(left, 0, mid - 22)
        self.bar_back["frameSize"] = (0, self.bar_width, -8, 0)
        self.bar.setPos(left, 0, mid - 22)
        self.sensor_mark.setPos(left + self.bar_width * self.sensor_range_m / LOOK_AHEAD_M, 0, mid - 22)
        self.bar_caption.setPos(left, mid - 48)
        self.route.setPos(width - 24, mid + 39)
        self.width = width

    def show(self, visible: bool) -> None:
        (self.root.show if visible else self.root.hide)()

    def update(self, episode: Episode, street: str) -> None:
        kmh = episode.speed_mps * 3.6
        self.needle.setR(self._angle(kmh))
        self.speed.setText(f"{kmh:.0f}")
        state = driving_state(episode)
        self.lamp["frameColor"] = (*state.color, 1)
        self.state.setText(state.text)
        self.state["fg"] = (*state.color, 1)
        ahead = object_ahead(episode)
        self.ahead.setText(object_text(ahead))
        if ahead is None:
            self.bar.hide()
        else:
            self.bar.show()
            fraction = ahead.distance_m / LOOK_AHEAD_M            # longueur de la barre = distance à l'objet
            color = RED if ahead.distance_m <= self.sensor_range_m else GREY
            self.bar["frameColor"] = (*color, 1)
            self.bar["frameSize"] = (0, max(2.0, self.bar_width * fraction), -8, 0)
        self.bar_caption.setText(f"distance (0 à {LOOK_AHEAD_M:.0f} m) · trait blanc : portée du capteur "
                                 f"{self.sensor_range_m:.0f} m")
        km = remaining_m(episode) / 1000
        self.route.setText(f"{street}\nArrivée dans {km:.1f} km".replace(".", ","))
