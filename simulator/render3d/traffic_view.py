"""Affichage du trafic d'ambiance (``traffic.py``) : véhicules, deux-roues, piétons et feux tricolores.

Les modèles viennent du catalogue (mêmes véhicules et piétons que les obstacles). Un véhicule est posé avec
son avant et son arrière sur la voie (il suit les courbes) ; un deux-roues roule près du trottoir.
Les feux sont procéduraux : poteau, boîtier et trois lampes dont la couleur suit le contrôleur du carrefour.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import shapely

from simulator.render3d.actors import edge_pose, lane_point
from simulator.render3d.catalog import Catalog, load_model
from simulator.render3d.masks import no_shadow
from simulator.render3d.traffic import SIDEWALK_OFFSET_M, TrafficSim, Vehicle
from simulator.render3d.world_mesh import LANE_OFFSET_M, ROAD_Z, SIDEWALK_Z, WorldFrame, heading_deg, road_width

LAMP_ON = {"red": (1.0, 0.16, 0.10), "yellow": (1.0, 0.72, 0.08), "green": (0.20, 1.0, 0.40)}
LAMP_OFF = (0.16, 0.16, 0.17)
LAMP_ORDER = ("red", "yellow", "green")


def _point_along(graph: nx.MultiDiGraph, vehicle: Vehicle, back_m: float) -> tuple[float, float, float, float]:
    """Point situé ``back_m`` mètres derrière l'avant du véhicule (sur le tronçon précédent si besoin)."""
    s = vehicle.s - back_m
    if s >= 0 or vehicle.prev_edge is None:
        return edge_pose(graph, vehicle.edge, max(s, 0.0))
    previous = float(graph.edges[vehicle.prev_edge]["length"])
    return edge_pose(graph, vehicle.prev_edge, max(previous + s, 0.0))


def vehicle_pose(graph: nx.MultiDiGraph, frame: WorldFrame, vehicle: Vehicle) -> tuple[float, float, float]:
    """(x, y, cap) locaux du centre du véhicule, sur sa voie (deux-roues : près du trottoir)."""
    width = road_width(graph.edges[vehicle.edge])
    lateral = max(LANE_OFFSET_M, width / 2 - 1.0) if vehicle.kind in ("bicycle", "motorcycle") else LANE_OFFSET_M
    fx, fy, fdx, fdy = _point_along(graph, vehicle, 0.0)
    rx, ry, rdx, rdy = _point_along(graph, vehicle, vehicle.length)
    front = lane_point(frame, fx, fy, fdx, fdy, lateral)
    rear = lane_point(frame, rx, ry, rdx, rdy, lateral)
    dx, dy = front[0] - rear[0], front[1] - rear[1]
    if math.hypot(dx, dy) < 1e-3:
        dx, dy = fdx, fdy
    return (front[0] + rear[0]) / 2, (front[1] + rear[1]) / 2, heading_deg(dx, dy)


def signal_head():
    """Feu procédural : poteau, boîtier, et les trois lampes (rouge en haut) sur la face -Y (vers les véhicules)."""
    from simulator.render3d.street_furniture import _Builder

    pole = _Builder()
    pole.prism(6, 0.0, 3.3, 0.08, 0.07, (0.25, 0.26, 0.28))
    head = _Builder()
    w, d, z0, z1 = 0.20, 0.16, 3.0, 4.05
    corners = {(sx, sy, sz): (sx * w, sy * d, z0 if sz < 0 else z1) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
    faces = [((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)), ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)),
             ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)), ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)),
             ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
    for a, b, c, e in faces:
        head.tri(corners[a], corners[b], corners[c], (0.10, 0.10, 0.11))
        head.tri(corners[a], corners[c], corners[e], (0.10, 0.10, 0.11))
    lamps = []
    for i in range(3):
        lamp = _Builder()
        cz = z1 - 0.18 - i * 0.34
        ring = [(0.12 * math.cos(a), -d - 0.01, cz + 0.12 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)[:-1]]
        for p, q in zip(ring, ring[1:] + ring[:1]):
            lamp.tri((0.0, -d - 0.01, cz), q, p, (1, 1, 1))
        lamps.append(lamp.mesh())
    return pole.mesh(), head.mesh(), lamps


class TrafficView:
    def __init__(self, parent, sim: TrafficSim, graph: nx.MultiDiGraph, frame: WorldFrame, catalog: Catalog, loader,
                 minimap=None) -> None:
        from panda3d.core import NodePath

        self.sim, self.graph, self.frame, self.catalog, self.loader = sim, graph, frame, catalog, loader
        self.root = parent.attachNewNode("trafic")
        self.minimap = minimap
        self._templates: dict[str, NodePath] = {}
        self._vehicles: dict[int, tuple[str, NodePath]] = {}
        self._people: dict[int, tuple[NodePath, object, str, str | None]] = {}
        self.lights = self._build_lights()

    # ------------------------------------------------------------------ modèles
    def _template(self, entry) -> NodePath:
        if entry.model not in self._templates:
            self._templates[entry.model] = load_model(entry, self.loader)
        return self._templates[entry.model]

    def _vehicle_node(self, vehicle: Vehicle):
        variants = self.catalog.obstacles[vehicle.kind]
        entry = variants[vehicle.variant % len(variants)]
        node = self._template(entry).copyTo(self.root)
        if self.minimap is not None:
            marker = self.minimap.obstacle_marker(node, (0.42, 0.43, 0.46))
            marker.setScale(0.55)
        return entry.model, node

    def _person_node(self, variant: int):
        from direct.actor.Actor import Actor
        from panda3d.core import NodePath

        variants = self.catalog.obstacles["person"]
        entry = variants[variant % len(variants)]
        template = self._template(entry).getPythonTag("actor")
        root = NodePath("pieton")
        actor = Actor(other=template)
        actor.reparentTo(root)
        actor.setTransform(template.getTransform())
        actor.loop(entry.animations["walk"], fromFrame=variant % 20)
        root.reparentTo(self.root)
        return root, actor, entry.animations["walk"], entry.animations["idle"]

    # ------------------------------------------------------------------ feux
    def _build_lights(self) -> list[tuple[tuple, list]]:
        from panda3d.core import NodePath

        from simulator.render3d.world_mesh import _geom_node, edge_line, push_off_road

        pole_mesh, head_mesh, lamp_meshes = signal_head()
        out = []
        for node_id, controller in self.sim.signals.items():
            near = [self.frame.geometry(edge_line(self.graph, u, v, self.graph.edges[(u, v, k)])).buffer(
                road_width(self.graph.edges[(u, v, k)]) / 2, cap_style="flat")
                for u, v, k in list(self.graph.in_edges(node_id, keys=True)) + list(self.graph.out_edges(node_id, keys=True))]
            roads = shapely.union_all(near)
            for edge in controller.groups[0] | controller.groups[1]:
                u, v, _ = edge
                line = edge_line(self.graph, u, v, self.graph.edges[edge])
                end, before = line.interpolate(line.length), line.interpolate(max(line.length - 4.0, 0.0))
                dx, dy = end.x - before.x, end.y - before.y
                norm = math.hypot(dx, dy) or 1.0
                dx, dy = dx / norm, dy / norm
                side = road_width(self.graph.edges[edge]) / 2 + 1.0
                x, y = self.frame.to_local(end.x - dx * 3.0 + dy * side, end.y - dy * 3.0 - dx * side)
                x, y = push_off_road(x, y, roads, roads.boundary)
                root = self.root.attachNewNode("feu")
                root.setPos(x, y, SIDEWALK_Z)
                root.setH(heading_deg(dx, dy))
                for mesh in (pole_mesh, head_mesh):
                    root.attachNewNode(_geom_node("feu", *mesh)).setTwoSided(True)
                lamps = []
                for name, mesh in zip(LAMP_ORDER, lamp_meshes):
                    lamp = root.attachNewNode(_geom_node(f"lampe_{name}", *mesh))
                    lamp.setLightOff(1)
                    lamp.setTwoSided(True)
                    no_shadow(lamp)
                    lamps.append((name, lamp))
                out.append((edge, lamps))
        return out

    # ------------------------------------------------------------------ mise à jour
    def update(self) -> None:
        sim, graph, frame = self.sim, self.graph, self.frame
        alive = set()
        for vehicle in sim.vehicles:
            alive.add(vehicle.vid)
            if vehicle.vid not in self._vehicles:
                self._vehicles[vehicle.vid] = self._vehicle_node(vehicle)
            _, node = self._vehicles[vehicle.vid]
            x, y, h = vehicle_pose(graph, frame, vehicle)
            node.setPos(x, y, ROAD_Z)
            node.setH(h)
        for vid in [v for v in self._vehicles if v not in alive]:
            self._vehicles.pop(vid)[1].removeNode()

        alive = set()
        for p in sim.pedestrians:
            alive.add(p.pid)
            if p.pid not in self._people:
                self._people[p.pid] = self._person_node(p.variant)
            root, actor, walk, _ = self._people[p.pid]
            x, y, dx, dy = edge_pose(graph, p.edge, min(p.s, sim.length[p.edge]))
            offset = sim.width[p.edge] / 2 + SIDEWALK_OFFSET_M
            if p.crossing is None:
                lx, ly = lane_point(frame, x, y, dx, dy, p.side * offset)
                root.setPos(lx, ly, SIDEWALK_Z)
                root.setH(heading_deg(dx, dy))
            else:
                lateral = p.side * offset * (1 - 2 * p.crossing)
                lx, ly = lane_point(frame, x, y, dx, dy, lateral)
                root.setPos(lx, ly, ROAD_Z)
                root.setH(heading_deg(-p.side * dy, p.side * dx))
        for pid in [p for p in self._people if p not in alive]:
            root, actor, _, _ = self._people.pop(pid)
            actor.cleanup()
            root.removeNode()

        for edge, lamps in self.lights:
            state = sim.signal_state(edge)
            for name, lamp in lamps:
                lamp.setColor(*(LAMP_ON[name] if name == state else LAMP_OFF), 1)

    def destroy(self) -> None:
        for _, actor, _, _ in self._people.values():
            actor.cleanup()
        self.root.removeNode()
        for template in self._templates.values():
            template.removeNode()
