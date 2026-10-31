"""Affichage du trafic d'ambiance (``traffic.py``) : véhicules, deux-roues, piétons et feux tricolores.

Les modèles viennent du catalogue (mêmes véhicules et piétons que les obstacles). Un véhicule est posé avec
son avant et son arrière sur la voie (il suit les courbes) ; un deux-roues roule près du trottoir.
Les feux sont procéduraux : poteau, boîtier et trois lampes dont la couleur suit le contrôleur du carrefour.

Performances (GPU intégré) : les matériaux des véhicules sont convertis en couleurs de sommets pour fusionner
chaque modèle en un seul maillage ; les positions sont interpolées avec numpy (``EdgePaths``) ; au loin, les
véhicules ne projettent plus d'ombre puis sont masqués. Foule : une réserve de ``ACTOR_POOL`` personnages animés
(par le GPU) est attribuée aux piétons les plus proches ; les autres sont des silhouettes low-poly immobiles.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import shapely

from simulator.render3d.actors import lane_point
from simulator.render3d.catalog import Catalog, load_model
from simulator.render3d.masks import no_shadow
from simulator.render3d.traffic import SIDEWALK_OFFSET_M, TrafficSim, Vehicle
from simulator.render3d.world_mesh import LANE_OFFSET_M, ROAD_Z, SIDEWALK_Z, WorldFrame, heading_deg, road_width

LAMP_ON = {"red": (1.0, 0.16, 0.10), "yellow": (1.0, 0.72, 0.08), "green": (0.20, 1.0, 0.40)}
LAMP_OFF = (0.16, 0.16, 0.17)
LAMP_ORDER = ("red", "yellow", "green")
SHADOW_DISTANCE_M = 60.0        # au-delà, un véhicule ne projette plus d'ombre
DRAW_DISTANCE_M = 180.0         # au-delà, il n'est plus dessiné (brume)
ANIMATE_DISTANCE_M = 60.0       # au-delà, un piéton est une silhouette simple
ACTOR_POOL = 12                 # personnages animés au plus
SHIRTS = ((0.80, 0.25, 0.22), (0.20, 0.40, 0.70), (0.90, 0.88, 0.82), (0.25, 0.55, 0.35), (0.85, 0.65, 0.20),
          (0.45, 0.30, 0.55), (0.30, 0.30, 0.32))
SKINS = ((0.87, 0.70, 0.56), (0.70, 0.52, 0.38), (0.50, 0.35, 0.25))
PEOPLE_DISTANCE_M = 100.0       # au-delà, un piéton n'est plus dessiné (il ferait moins de 10 pixels)
PEOPLE_SHADOW_M = 30.0
LIGHT_PERSON_TRIANGLES = 3000   # piétons du trafic : seulement les modèles légers (les obstacles gardent tout)


class EdgePaths:
    """Géométrie des tronçons en tableaux numpy : position et direction à une distance donnée, sans shapely.

    Même convention que ``world.car.point_on_edge`` : la distance est rapportée à la longueur OSM du tronçon.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph
        self._paths: dict = {}

    def _path(self, edge):
        if edge not in self._paths:
            data = self.graph.edges[edge]
            geometry = data.get("geometry")
            if geometry is not None:
                xy = np.asarray(geometry.coords, dtype=np.float64)
            else:
                a, b = self.graph.nodes[edge[0]], self.graph.nodes[edge[1]]
                xy = np.array([[a["x"], a["y"]], [b["x"], b["y"]]], dtype=np.float64)
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
            self._paths[edge] = (cum, xy, float(data["length"]) or 1.0)
        return self._paths[edge]

    def pose(self, edge, offset_m: float) -> tuple[float, float, float, float]:
        """(x, y) dans le repère de la carte et direction unitaire (dx, dy) à ``offset_m`` du début du tronçon."""
        cum, xy, length = self._path(edge)
        d = min(max(offset_m, 0.0), length) / length * cum[-1]
        i = int(min(max(np.searchsorted(cum, d, side="right") - 1, 0), len(cum) - 2))
        seg = cum[i + 1] - cum[i] or 1.0
        t = (d - cum[i]) / seg
        (x0, y0), (x1, y1) = xy[i], xy[i + 1]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        return x0 + t * dx, y0 + t * dy, dx / norm, dy / norm


def bake_materials(node) -> None:
    """Remplace les matériaux sans texture par des couleurs de sommets, puis fusionne : un maillage par modèle."""
    from panda3d.core import ColorAttrib, MaterialAttrib, TextureAttrib

    for path in node.findAllMatches("**/+GeomNode"):
        geom_node = path.node()
        for i in range(geom_node.getNumGeoms()):
            state = geom_node.getGeomState(i)
            textured = state.hasAttrib(TextureAttrib) and state.getAttrib(TextureAttrib).getNumOnStages() > 0
            if not state.hasAttrib(MaterialAttrib) or textured:
                continue
            material = state.getAttrib(MaterialAttrib).getMaterial()
            color = material.getBaseColor() if material.hasBaseColor() else material.getDiffuse()
            state = state.removeAttrib(MaterialAttrib).removeAttrib(TextureAttrib)   # TextureAttrib vide de glTF
            geom_node.setGeomState(i, state.setAttrib(ColorAttrib.makeFlat(color)))
    node.flattenStrong()


def _point_along(paths: EdgePaths, vehicle: Vehicle, back_m: float) -> tuple[float, float, float, float]:
    """Point situé ``back_m`` mètres derrière l'avant du véhicule (sur le tronçon précédent si besoin)."""
    s = vehicle.s - back_m
    if s >= 0 or vehicle.prev_edge is None:
        return paths.pose(vehicle.edge, max(s, 0.0))
    previous = paths._path(vehicle.prev_edge)[2]
    return paths.pose(vehicle.prev_edge, max(previous + s, 0.0))


def vehicle_pose(paths: EdgePaths, frame: WorldFrame, vehicle: Vehicle, width: float) -> tuple[float, float, float]:
    """(x, y, cap) locaux du centre du véhicule, sur sa voie (deux-roues : près du trottoir)."""
    lateral = max(LANE_OFFSET_M, width / 2 - 1.0) if vehicle.kind in ("bicycle", "motorcycle") else LANE_OFFSET_M
    fx, fy, fdx, fdy = _point_along(paths, vehicle, 0.0)
    rx, ry, rdx, rdy = _point_along(paths, vehicle, vehicle.length)
    front = lane_point(frame, fx, fy, fdx, fdy, lateral)
    rear = lane_point(frame, rx, ry, rdx, rdy, lateral)
    dx, dy = front[0] - rear[0], front[1] - rear[1]
    if math.hypot(dx, dy) < 1e-3:
        dx, dy = fdx, fdy
    return (front[0] + rear[0]) / 2, (front[1] + rear[1]) / 2, heading_deg(dx, dy)


def person_proxy(shirt, skin):
    """Silhouette de piéton (environ 1,7 m, 40 triangles) pour les piétons éloignés."""
    from simulator.render3d.street_furniture import _Builder

    b = _Builder()

    def box(x0, x1, y0, y1, z0, z1, color):
        c = {(i, j, k): (x, y, z) for i, x in enumerate((x0, x1)) for j, y in enumerate((y0, y1))
             for k, z in enumerate((z0, z1))}
        for a, bb, cc, dd in (((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
                              ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)), ((0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)),
                              ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))):
            b.tri(c[a], c[bb], c[cc], color)
            b.tri(c[a], c[cc], c[dd], color)

    trousers = (0.22, 0.22, 0.26)
    box(-0.17, -0.02, -0.09, 0.09, 0.0, 0.85, trousers)
    box(0.02, 0.17, -0.09, 0.09, 0.0, 0.85, trousers)
    box(-0.21, 0.21, -0.12, 0.12, 0.85, 1.45, shirt)
    box(-0.10, 0.10, -0.10, 0.10, 1.47, 1.70, skin)
    return b.mesh()


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
        self.paths = EdgePaths(graph)
        self.root = parent.attachNewNode("trafic")
        self.minimap = minimap
        self._templates: dict[str, NodePath] = {}
        self._vehicles: dict[int, tuple[str, NodePath]] = {}
        self._people: dict[int, tuple[NodePath, object, str, str | None]] = {}
        self.lights = self._build_lights()

    # ------------------------------------------------------------------ modèles
    def _template(self, entry) -> NodePath:
        if entry.model not in self._templates:
            template = load_model(entry, self.loader)
            if not entry.animated:
                bake_materials(template)
            self._templates[entry.model] = template
        return self._templates[entry.model]

    def _vehicle_node(self, vehicle: Vehicle):
        variants = self.catalog.obstacles[vehicle.kind]
        entry = variants[vehicle.variant % len(variants)]
        node = self._template(entry).copyTo(self.root)
        if self.minimap is not None:
            marker = self.minimap.obstacle_marker(node, (0.25, 0.27, 0.32))
            marker.setScale(0.75)
        return entry.model, node

    def _person_node(self, variant: int):
        """Silhouette simple d'un piéton (remplacée par un personnage animé quand il est proche)."""
        from simulator.render3d.world_mesh import _geom_node

        mesh = person_proxy(SHIRTS[variant % len(SHIRTS)], SKINS[(variant // 7) % len(SKINS)])
        root = self.root.attachNewNode(_geom_node("silhouette", *mesh))
        return root

    def _actor_pool(self) -> list:
        """Personnages animés, créés une fois, attribués à chaque image aux piétons les plus proches."""
        if not hasattr(self, "_pool"):
            from direct.actor.Actor import Actor
            from panda3d.core import NodePath

            self._pool = []
            variants = self._light_people()
            for i in range(ACTOR_POOL):
                entry = variants[i % len(variants)]
                template = self._template(entry).getPythonTag("actor")
                root = NodePath("pieton_anime")
                actor = Actor(other=template)
                actor.reparentTo(root)
                actor.setTransform(template.getTransform())
                actor.loop(entry.animations["walk"], fromFrame=(i * 7) % 20)
                root.reparentTo(self.root)
                root.setShaderAuto(10)          # squelette animé par le GPU (hardware-animated-vertices)
                root.hide()
                self._pool.append({"root": root, "actor": actor, "pid": None})
        return self._pool

    def _light_people(self) -> list:
        if not hasattr(self, "_people_variants"):
            def triangles(entry) -> int:
                actor = self._template(entry).getPythonTag("actor")
                return sum(path.node().getGeom(i).getNumPrimitives() and path.node().getGeom(i).getPrimitive(0).getNumFaces()
                           for path in actor.findAllMatches("**/+GeomNode") for i in range(path.node().getNumGeoms()))

            people = self.catalog.obstacles["person"]
            light = [e for e in people if triangles(e) <= LIGHT_PERSON_TRIANGLES]
            self._people_variants = light or list(people)
        return self._people_variants

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
    def update(self, camera_pos) -> None:
        from simulator.render3d.masks import MAIN_MASK, SHADOW_MASK

        sim, frame, paths = self.sim, self.frame, self.paths
        cx, cy = camera_pos.x, camera_pos.y
        alive = set()
        for vehicle in sim.vehicles:
            alive.add(vehicle.vid)
            if vehicle.vid not in self._vehicles:
                self._vehicles[vehicle.vid] = self._vehicle_node(vehicle)
            _, node = self._vehicles[vehicle.vid]
            x, y, h = vehicle_pose(paths, frame, vehicle, sim.width[vehicle.edge])
            node.setPos(x, y, ROAD_Z)
            node.setH(h)
            distance = math.hypot(x - cx, y - cy)
            (node.show if distance < DRAW_DISTANCE_M else node.hide)(MAIN_MASK)
            (node.show if distance < SHADOW_DISTANCE_M else node.hide)(SHADOW_MASK)
        for vid in [v for v in self._vehicles if v not in alive]:
            self._vehicles.pop(vid)[1].removeNode()

        alive = set()
        poses = {}
        for p in sim.pedestrians:
            alive.add(p.pid)
            if p.pid not in self._people:
                self._people[p.pid] = self._person_node(p.variant)
            root = self._people[p.pid]
            x, y, dx, dy = paths.pose(p.edge, min(p.s, sim.length[p.edge]))
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
            distance = math.hypot(lx - cx, ly - cy)
            (root.show if distance < PEOPLE_DISTANCE_M else root.hide)(MAIN_MASK)
            (root.show if distance < PEOPLE_SHADOW_M else root.hide)(SHADOW_MASK)
            poses[p.pid] = (distance, root)
        for pid in [p for p in self._people if p not in alive]:
            self._people.pop(pid).removeNode()
        self._assign_actors(poses)

        for edge, lamps in self.lights:
            state = sim.signal_state(edge)
            for name, lamp in lamps:
                lamp.setColor(*(LAMP_ON[name] if name == state else LAMP_OFF), 1)

    def _assign_actors(self, poses: dict) -> None:
        """Les piétons les plus proches reçoivent un personnage animé, à la place de leur silhouette."""
        from simulator.render3d.masks import MAIN_MASK, SHADOW_MASK

        pool = self._actor_pool()
        near = sorted((d, pid) for pid, (d, _) in poses.items() if d < ANIMATE_DISTANCE_M)[:len(pool)]
        wanted = {pid for _, pid in near}
        for slot in pool:
            if slot["pid"] not in wanted:
                if slot["pid"] in poses:
                    poses[slot["pid"]][1].show(MAIN_MASK)
                slot["pid"] = None
                slot["root"].hide()
        taken = {slot["pid"] for slot in pool}
        free = [slot for slot in pool if slot["pid"] is None]
        for _, pid in near:
            if pid not in taken and free:
                free.pop()["pid"] = pid
        for slot in pool:
            if slot["pid"] is None:
                continue
            proxy = poses[slot["pid"]][1]
            slot["root"].setPosHpr(proxy.getPos(), proxy.getHpr())
            slot["root"].show(MAIN_MASK | SHADOW_MASK)
            proxy.hide(MAIN_MASK | SHADOW_MASK)

    def destroy(self) -> None:
        for slot in getattr(self, "_pool", []):
            slot["actor"].cleanup()
        self.root.removeNode()
        for template in self._templates.values():
            template.removeNode()
