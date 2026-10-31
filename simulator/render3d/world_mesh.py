"""Scène 3D de la carte OSM : sol, chaussées, trottoirs, marquages, bâtiments et signalisation.

Équivalent de l'import OpenDRIVE de CARLA, à partir de NOTRE graphe : la voiture 3D, Dijkstra et les obstacles
partagent exactement les mêmes tronçons. Voir le skill ``osm-to-3d``.

Deux couches :
- la géométrie, en Python pur (shapely, numpy) : testable sans GPU ;
- ``build_world`` la transforme en maillages Panda3D, découpés en tuiles et mis en cache (.bam).

Repère : les coordonnées UTM (environ 3,7 millions de mètres) perdent leur précision en float32 ; tout est donc
exprimé dans un repère local centré sur la carte (``WorldFrame``). X = est, Y = nord, Z = haut, en mètres.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
from shapely import affinity, ops
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from simulator.render3d.catalog import ASSETS_3D

MESH_VERSION = 1                 # à incrémenter quand la géométrie change : invalide le cache
CACHE_DIR = ASSETS_3D / "_cache"
TILE_M = 200.0                   # côté d'une tuile (une tuile = un nœud aplati = peu d'appels de dessin)
LANE_WIDTH_M = 3.2
SIDEWALK_M = 2.5
INTERSECTION_CLEAR_M = 8.0       # pas de marquage à moins de 8 m d'un carrefour
DASH_M, GAP_M = 3.0, 3.0         # ligne centrale en tirets
LANE_OFFSET_M = 1.8              # on roule à droite : centre de la voie de droite (comme ui/driver_view.py)

ROAD_Z, MARKING_Z, SIDEWALK_Z = 0.02, 0.04, 0.12

# Largeur par défaut selon le type de route OSM, quand ni « width » ni « lanes » ne sont renseignés.
HIGHWAY_WIDTH_M = {
    "motorway": 14.0, "trunk": 14.0, "primary": 12.0, "secondary": 10.0, "tertiary": 8.0,
    "residential": 6.5, "unclassified": 6.5, "living_street": 5.0, "service": 4.5,
}
DEFAULT_WIDTH_M = 6.5

GROUND_COLOR = (0.80, 0.78, 0.72)
ROAD_COLOR = (0.30, 0.31, 0.33)
SIDEWALK_COLOR = (0.70, 0.69, 0.66)
MARKING_COLOR = (0.95, 0.95, 0.92)
# Façades claires, typiques du centre de Casablanca ; toit un peu plus sombre.
BUILDING_COLORS = ((0.93, 0.91, 0.86), (0.90, 0.86, 0.78), (0.86, 0.84, 0.80), (0.95, 0.94, 0.90), (0.88, 0.82, 0.74))
LEVEL_HEIGHT_M = 3.2


# ---------------------------------------------------------------------- repère local
@dataclass(frozen=True)
class WorldFrame:
    origin_x: float
    origin_y: float

    @classmethod
    def from_graph(cls, graph: nx.MultiDiGraph) -> WorldFrame:
        xs = [d["x"] for _, d in graph.nodes(data=True)]
        ys = [d["y"] for _, d in graph.nodes(data=True)]
        return cls(round((min(xs) + max(xs)) / 2), round((min(ys) + max(ys)) / 2))

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        return x - self.origin_x, y - self.origin_y

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        return x + self.origin_x, y + self.origin_y

    def geometry(self, geom):
        return affinity.translate(geom, -self.origin_x, -self.origin_y)


def heading_deg(dx: float, dy: float) -> float:
    """Cap Panda3D (degrés) d'un objet dont l'avant (+Y local) doit pointer vers (dx, dy)."""
    return math.degrees(math.atan2(dy, dx)) - 90.0


# ---------------------------------------------------------------------- rues
def _first(value):
    return value[0] if isinstance(value, list) and value else value


def _number(value) -> float | None:
    value = _first(value)
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ".").split()[0])
    except (ValueError, IndexError):
        return None


def road_width(data: dict) -> float:
    """Largeur de chaussée (m) : balise « width », sinon « lanes » × 3,2 m, sinon selon le type de route."""
    width = _number(data.get("width"))
    if width is not None and 2.5 <= width <= 40:
        return width
    lanes = _number(data.get("lanes"))
    if lanes is not None and 1 <= lanes <= 10:
        return max(lanes * LANE_WIDTH_M, 4.0)
    highway = str(_first(data.get("highway")) or "").removesuffix("_link")
    return HIGHWAY_WIDTH_M.get(highway, DEFAULT_WIDTH_M)


def edge_line(graph: nx.MultiDiGraph, u: int, v: int, data: dict) -> LineString:
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry
    a, b = graph.nodes[u], graph.nodes[v]
    return LineString([(a["x"], a["y"]), (b["x"], b["y"])])


@dataclass(frozen=True)
class Street:
    line: LineString      # coordonnées locales
    width: float
    two_way: bool


def streets(graph: nx.MultiDiGraph, frame: WorldFrame) -> list[Street]:
    """Une rue par tronçon physique : une rue à double sens apparaît deux fois dans le graphe (u→v et v→u)."""
    seen: dict[tuple, Street] = {}
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v), round(float(data.get("length", 0.0)), 1))
        if key in seen:
            seen[key] = Street(seen[key].line, seen[key].width, True)
            continue
        oneway = _first(data.get("oneway"))
        two_way = str(oneway).lower() not in ("true", "1", "yes")
        seen[key] = Street(frame.geometry(edge_line(graph, u, v, data)), road_width(data), two_way)
    return list(seen.values())


@dataclass(frozen=True)
class RoadLayout:
    roads: Polygon | MultiPolygon
    sidewalks: Polygon | MultiPolygon
    markings: Polygon | MultiPolygon


def road_layout(street_list: list[Street]) -> RoadLayout:
    """Polygones des chaussées (les carrefours se forment par union), des trottoirs et des marquages."""
    roads = shapely.union_all([s.line.buffer(s.width / 2, cap_style="flat") for s in street_list] +
                              [Point(c).buffer(s.width / 2) for s in street_list for c in s.line.boundary.geoms])
    outer = shapely.union_all([s.line.buffer(s.width / 2 + SIDEWALK_M) for s in street_list])
    sidewalks = outer.difference(roads)
    dashes = []
    for street in street_list:
        if not street.two_way or street.width < 6.0:
            continue
        length = street.line.length
        position = INTERSECTION_CLEAR_M
        while position + DASH_M <= length - INTERSECTION_CLEAR_M:
            dashes.append(ops.substring(street.line, position, position + DASH_M).buffer(0.08, cap_style="flat"))
            position += DASH_M + GAP_M
    markings = shapely.union_all(dashes) if dashes else Polygon()
    return RoadLayout(roads, sidewalks, markings)


def tiles(geom, tile_m: float = TILE_M) -> Iterator[tuple[tuple[int, int], Polygon | MultiPolygon]]:
    """Découpe une géométrie en tuiles carrées (pour le découpage du rendu)."""
    if geom.is_empty:
        return
    minx, miny, maxx, maxy = geom.bounds
    for i in range(math.floor(minx / tile_m), math.floor(maxx / tile_m) + 1):
        for j in range(math.floor(miny / tile_m), math.floor(maxy / tile_m) + 1):
            part = geom.intersection(box(i * tile_m, j * tile_m, (i + 1) * tile_m, (j + 1) * tile_m))
            if not part.is_empty and part.area > 0:
                yield (i, j), part


def _polygons(geom) -> Iterable[Polygon]:
    if isinstance(geom, Polygon):
        yield geom
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _polygons(g)


def triangulate(geom) -> tuple[np.ndarray, np.ndarray]:
    """Triangulation contrainte d'un (multi)polygone, trous respectés : sommets (N, 2) et triangles (M, 3)."""
    vertices: list[tuple[float, float]] = []
    for polygon in _polygons(geom):
        for triangle in shapely.constrained_delaunay_triangles(polygon).geoms:
            coords = list(triangle.exterior.coords)[:3]
            # Sens trigonométrique : la face visible regarde vers le haut (+Z).
            (x0, y0), (x1, y1), (x2, y2) = coords
            if (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) < 0:
                coords = [coords[0], coords[2], coords[1]]
            vertices.extend(coords)
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    return points, np.arange(len(points), dtype=np.int32).reshape(-1, 3)


# ---------------------------------------------------------------------- bâtiments
def extrude(polygon: Polygon, height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Murs et toit d'un bâtiment : positions (N, 3), normales (N, 3), triangles (M, 3)."""
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    ring = shapely.geometry.polygon.orient(polygon, sign=1.0).exterior.coords   # sens trigonométrique
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        n = (dy / norm, -dx / norm, 0.0)                     # vers l'extérieur pour un anneau trigonométrique
        quad = [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, height), (x0, y0, 0.0), (x1, y1, height), (x0, y0, height)]
        positions.extend(quad)
        normals.extend([n] * 6)
    roof, _ = triangulate(polygon)
    positions.extend((x, y, height) for x, y in roof)
    normals.extend([(0.0, 0.0, 1.0)] * len(roof))
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    return pos, np.asarray(normals, dtype=np.float64).reshape(-1, 3), np.arange(len(pos), dtype=np.int32).reshape(-1, 3)


def building_height(height: float, levels: float, seed: int) -> float:
    """Hauteur OSM (balise height, sinon étages × 3,2 m), sinon estimée : 3 à 7 étages tirés avec la graine."""
    if np.isfinite(height) and 2.5 <= height <= 200:
        return float(height)
    if np.isfinite(levels) and 1 <= levels <= 60:
        return float(levels) * LEVEL_HEIGHT_M
    return LEVEL_HEIGHT_M * int(np.random.default_rng(seed).integers(3, 8))


# ---------------------------------------------------------------------- signalisation
@dataclass(frozen=True)
class PropPlacement:
    kind: str               # clé de catalog.props : "traffic_signals", "stop", "crossing"
    x: float
    y: float
    heading: float          # la face du panneau (-Y du modèle) regarde les véhicules qui arrivent


def prop_placements(graph: nx.MultiDiGraph, frame: WorldFrame) -> list[PropPlacement]:
    """Un feu ou un panneau par tronçon entrant dans un nœud OSM marqué, à droite de la chaussée."""
    placements = []
    for node, data in graph.nodes(data=True):
        kind = _first(data.get("highway"))
        if kind not in ("traffic_signals", "stop", "crossing"):
            continue
        for u, v, edge in graph.in_edges(node, data=True):
            line = edge_line(graph, u, v, edge)
            if line.length < 1.0:
                continue
            end = line.interpolate(line.length)
            before = line.interpolate(max(line.length - 4.0, 0.0))
            dx, dy = end.x - before.x, end.y - before.y
            norm = math.hypot(dx, dy) or 1.0
            dx, dy = dx / norm, dy / norm
            side = road_width(edge) / 2 + 1.0
            back = 3.0 if kind != "crossing" else 0.0
            x, y = frame.to_local(end.x - dx * back + dy * side, end.y - dy * back - dx * side)
            placements.append(PropPlacement(kind, x, y, heading_deg(dx, dy)))
            if kind == "crossing":
                break                       # un seul panneau par passage piéton
    return placements


# ---------------------------------------------------------------------- Panda3D
def _geom_node(name: str, positions: np.ndarray, normals: np.ndarray, colors: np.ndarray, triangles: np.ndarray):
    from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexArrayFormat, GeomVertexData,
                              GeomVertexFormat)

    array = GeomVertexArrayFormat()
    array.addColumn("vertex", 3, Geom.NT_float32, Geom.C_point)
    array.addColumn("normal", 3, Geom.NT_float32, Geom.C_normal)
    array.addColumn("color", 4, Geom.NT_float32, Geom.C_color)
    fmt = GeomVertexFormat.registerFormat(GeomVertexFormat(array))
    data = GeomVertexData(name, fmt, Geom.UH_static)
    data.uncleanSetNumRows(len(positions))
    interleaved = np.hstack([positions, normals, colors]).astype(np.float32)
    memoryview(data.modifyArray(0)).cast("B")[:] = interleaved.tobytes()
    prim = GeomTriangles(Geom.UH_static)
    prim.setIndexType(Geom.NT_uint32)
    prim.modifyVertices().uncleanSetNumRows(triangles.size)
    memoryview(prim.modifyVertices()).cast("B")[:] = triangles.astype(np.uint32).tobytes()
    geom = Geom(data)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def _flat(name: str, geom, z: float, color: tuple[float, float, float]):
    points, triangles = triangulate(geom)
    if not len(points):
        return None
    positions = np.column_stack([points, np.full(len(points), z)])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    colors = np.tile([*color, 1.0], (len(points), 1))
    return _geom_node(name, positions, normals, colors, triangles)


def _buildings_node(polygons: list[Polygon], heights: list[float]):
    parts_pos, parts_nrm, parts_col = [], [], []
    for i, (polygon, height) in enumerate(zip(polygons, heights)):
        pos, nrm, _ = extrude(polygon, height)
        wall = np.array(BUILDING_COLORS[i % len(BUILDING_COLORS)])
        col = np.where(nrm[:, 2:3] > 0.5, wall * 0.85, wall)        # toit un peu plus sombre
        parts_pos.append(pos), parts_nrm.append(nrm), parts_col.append(np.column_stack([col, np.ones(len(col))]))
    if not parts_pos:
        return None
    pos, nrm, col = np.vstack(parts_pos), np.vstack(parts_nrm), np.vstack(parts_col)
    return _geom_node("batiments", pos, nrm, col, np.arange(len(pos), dtype=np.int32).reshape(-1, 3))


def cache_key(graph_file: Path, buildings_file: Path | None) -> str:
    digest = hashlib.sha1(f"v{MESH_VERSION}".encode())
    for path in (graph_file, buildings_file):
        if path is not None and path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_world(graph: nx.MultiDiGraph, frame: WorldFrame, loader, catalog=None, buildings=None,
                cache_file: Path | None = None):
    """NodePath de toute la ville. ``buildings`` : (polygones UTM, hauteurs) ou None.

    Avec ``cache_file`` : relu s'il existe, sinon construit puis enregistré.
    """
    from panda3d.core import Filename, NodePath

    if cache_file is not None and cache_file.exists():
        cached = loader.loadModel(Filename.fromOsSpecific(str(cache_file)), noCache=True)
        if cached is not None and not cached.isEmpty():
            return cached

    world = NodePath("ville")
    street_list = streets(graph, frame)
    layout = road_layout(street_list)
    extent = shapely.union_all([layout.roads, layout.sidewalks]).envelope.buffer(150, join_style="mitre")
    ground = _flat("sol", extent, 0.0, GROUND_COLOR)
    world.attachNewNode(ground)

    footprints: dict[tuple[int, int], tuple[list[Polygon], list[float]]] = {}
    if buildings is not None:
        polys, heights = buildings
        for i, (poly, height) in enumerate(zip(polys, heights)):
            local = frame.geometry(poly)
            if local.intersects(layout.roads):          # bâtiment OSM qui empiète sur la chaussée : on le retire
                local = local.difference(layout.roads.buffer(0.5))
            for part in _polygons(local):
                if part.area < 12:
                    continue
                c = part.representative_point()
                key = (math.floor(c.x / TILE_M), math.floor(c.y / TILE_M))
                footprints.setdefault(key, ([], []))
                footprints[key][0].append(part)
                footprints[key][1].append(height)

    road_tiles = dict(tiles(layout.roads))
    side_tiles = dict(tiles(layout.sidewalks))
    mark_tiles = dict(tiles(layout.markings))
    for key in sorted(set(road_tiles) | set(side_tiles) | set(footprints)):
        tile = world.attachNewNode(f"tuile_{key[0]}_{key[1]}")
        for name, parts, z, color in (("chaussee", road_tiles, ROAD_Z, ROAD_COLOR),
                                      ("trottoir", side_tiles, SIDEWALK_Z, SIDEWALK_COLOR),
                                      ("marquage", mark_tiles, MARKING_Z, MARKING_COLOR)):
            if key in parts and (node := _flat(name, parts[key], z, color)) is not None:
                tile.attachNewNode(node)
        if key in footprints and (node := _buildings_node(*footprints[key])) is not None:
            tile.attachNewNode(node)
        tile.flattenStrong()

    if catalog is not None:
        from simulator.render3d.catalog import load_model

        props = world.attachNewNode("signalisation")
        templates = {kind: load_model(entry, loader) for kind, entry in catalog.props.items()}
        for placement in prop_placements(graph, frame):
            copy = templates[placement.kind].copyTo(props)
            copy.setPos(placement.x, placement.y, SIDEWALK_Z)
            copy.setH(placement.heading)
        props.flattenStrong()

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        world.writeBamFile(Filename.fromOsSpecific(str(cache_file)))
    return world
