"""Scène 3D de la carte OSM : sol, chaussées, trottoirs, marquages, bâtiments et signalisation.

Équivalent de l'import OpenDRIVE de CARLA, à partir de NOTRE graphe : la voiture 3D, Dijkstra et les obstacles
partagent exactement les mêmes tronçons. Voir le skill ``osm-to-3d``.

Deux couches :
- la géométrie, en Python pur (shapely, numpy) : testable sans GPU ;
- ``build_world`` la transforme en maillages Panda3D, découpés en tuiles et mis en cache (.bam).

Apparence (« ville blanche ») : façades claires Art Déco ou modernes, texturées étage par étage, rez-de-chaussée
commerçants à stores colorés, quelques façades pastel, boulevards bordés de palmiers, trottoirs en dalles avec
bordures, zébras aux passages piétons et aux feux. Textures : ``textures.py`` ; arbres et lampadaires :
``street_furniture.py``.

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

from simulator.render3d import street_furniture, textures
from simulator.render3d.catalog import ASSETS_3D

MESH_VERSION = 6                 # à incrémenter quand la géométrie change : invalide le cache
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

# Couleurs multipliées par les textures (textures claires : ces couleurs donnent la teinte finale).
GROUND_COLOR = (0.84, 0.82, 0.77)
ROAD_COLOR = (0.38, 0.39, 0.41)
SIDEWALK_COLOR = (0.84, 0.82, 0.78)
CURB_COLOR = (0.74, 0.73, 0.70)
MARKING_COLOR = (0.96, 0.96, 0.93)
GRASS_COLOR = (0.47, 0.66, 0.36)
ISLAND_MAX_M2 = 4000.0           # trou de la chaussée plus petit : îlot (rond-point, terre-plein) engazonné
# « Ville blanche » : surtout des blancs et des crèmes, quelques façades pastel (ocre, rose, sable, bleu-gris).
BUILDING_COLORS = (
    ((0.98, 0.97, 0.94), 5), ((0.96, 0.93, 0.87), 4), ((0.93, 0.90, 0.84), 3),
    ((0.95, 0.83, 0.64), 1), ((0.96, 0.86, 0.82), 1), ((0.91, 0.83, 0.69), 1),
    ((0.97, 0.92, 0.74), 1), ((0.84, 0.88, 0.92), 1),
)
ROOF_COLORS = ((0.66, 0.65, 0.63), (0.60, 0.59, 0.58), (0.72, 0.69, 0.65), (0.70, 0.55, 0.45))   # terrasses, rares tuiles
LEVEL_HEIGHT_M = 3.2
SHOP_MIN_HEIGHT_M = 7.0          # en dessous, pas de rez-de-chaussée commercial


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
    kind: str = "minor"   # "major" (boulevards : palmiers), "medium", "minor" (street_furniture.street_class)


def streets(graph: nx.MultiDiGraph, frame: WorldFrame) -> list[Street]:
    """Une rue par tronçon physique : une rue à double sens apparaît deux fois dans le graphe (u→v et v→u)."""
    seen: dict[tuple, Street] = {}
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v), round(float(data.get("length", 0.0)), 1))
        if key in seen:
            seen[key] = Street(seen[key].line, seen[key].width, True, seen[key].kind)
            continue
        oneway = _first(data.get("oneway"))
        two_way = str(oneway).lower() not in ("true", "1", "yes")
        seen[key] = Street(frame.geometry(edge_line(graph, u, v, data)), road_width(data), two_way,
                           street_furniture.street_class(data.get("highway")))
    return list(seen.values())


@dataclass(frozen=True)
class RoadLayout:
    roads: Polygon | MultiPolygon
    sidewalks: Polygon | MultiPolygon
    markings: Polygon | MultiPolygon
    islands: Polygon | MultiPolygon = Polygon()   # îlots entourés de chaussée : gazon


def traffic_islands(roads) -> Polygon | MultiPolygon:
    """Petits trous de la chaussée (centre des ronds-points, terre-pleins), réduits de 1 m pour garder une bordure."""
    holes = [Polygon(ring).buffer(-1.0) for polygon in _polygons(roads) for ring in polygon.interiors
             if Polygon(ring).area <= ISLAND_MAX_M2]
    return shapely.union_all([h for h in holes if not h.is_empty]) if holes else Polygon()


def crosswalks(graph: nx.MultiDiGraph, frame: WorldFrame) -> list[Polygon]:
    """Bandes de zébra : en travers de la rue aux passages piétons OSM, sur chaque branche des carrefours à feux."""
    stripes: list[Polygon] = []
    for node, data in graph.nodes(data=True):
        kind = _first(data.get("highway"))
        if kind not in ("crossing", "traffic_signals"):
            continue
        cx0, cy0 = frame.to_local(data["x"], data["y"])
        branches: dict[int, tuple[LineString, float]] = {}
        for u, v, edge in list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True)):
            line = frame.geometry(edge_line(graph, u, v, edge))
            if u != node:
                line = LineString(list(line.coords)[::-1])     # toujours orientée depuis le nœud
            branches.setdefault(v if u == node else u, (line, road_width(edge)))
        if not branches:
            continue
        widest = max(w for _, w in branches.values())
        chosen = list(branches.values())[:1] if kind == "crossing" else list(branches.values())
        for line, width in chosen:
            if line.length < 6:
                continue
            ahead = line.interpolate(min(2.0, line.length))
            dx, dy = ahead.x - cx0, ahead.y - cy0
            norm = math.hypot(dx, dy) or 1.0
            dx, dy = dx / norm, dy / norm
            back = 0.0 if kind == "crossing" else min(widest / 2 + 2.5, line.length / 2)
            cx, cy = cx0 + dx * back, cy0 + dy * back
            for k in range(int(width)):
                t = -width / 2 + 0.5 + k
                ox, oy = cx + dy * t, cy - dx * t
                stripes.append(Polygon([(ox + dx * a + dy * b, oy + dy * a - dx * b)
                                        for a, b in ((-1.5, -0.25), (1.5, -0.25), (1.5, 0.25), (-1.5, 0.25))]))
    return stripes


def road_layout(street_list: list[Street], crosswalk_stripes: list[Polygon] = ()) -> RoadLayout:
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
    paint = dashes + [stripe.intersection(roads) for stripe in crosswalk_stripes]
    markings = shapely.union_all(paint) if paint else Polygon()
    return RoadLayout(roads, sidewalks, markings, traffic_islands(roads))


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


def wall_band(polygon: Polygon, z0: float, z1: float, bay_m: float, unit_v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Murs d'un bâtiment entre z0 et z1, avec coordonnées de texture : positions, normales, uv (N, 2).

    Sur chaque mur, u compte des travées entières (les fenêtres ne sont jamais coupées) et v des étages
    entiers (``unit_v`` mètres par répétition), arrondis au plus proche.
    """
    positions, normals, uvs = [], [], []
    v_max = max(1, round((z1 - z0) / unit_v))
    ring = shapely.geometry.polygon.orient(polygon, sign=1.0).exterior.coords
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        n = (dy / length, -dx / length, 0.0)
        u_max = max(1, round(length / bay_m)) if length >= bay_m * 0.6 else length / bay_m
        positions.extend([(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z0), (x1, y1, z1), (x0, y0, z1)])
        normals.extend([n] * 6)
        uvs.extend([(0, 0), (u_max, 0), (u_max, v_max), (0, 0), (u_max, v_max), (0, v_max)])
    return (np.asarray(positions, dtype=np.float64).reshape(-1, 3), np.asarray(normals, dtype=np.float64).reshape(-1, 3),
            np.asarray(uvs, dtype=np.float64).reshape(-1, 2))


def building_style(index: int) -> tuple[str, int | None, tuple[float, float, float], tuple[float, float, float]]:
    """Style de façade, variante de vitrine (ou None), couleur des murs et du toit, tirés avec l'indice du bâtiment."""
    rng = np.random.default_rng([index, 17])
    style = textures.FACADE_STYLES[int(rng.choice(len(textures.FACADE_STYLES), p=(0.5, 0.2, 0.3)))]
    colors, weights = zip(*BUILDING_COLORS)
    wall = colors[int(rng.choice(len(colors), p=np.asarray(weights) / sum(weights)))]
    roof = ROOF_COLORS[int(rng.integers(len(ROOF_COLORS)))]
    shop = int(rng.integers(4)) if rng.random() < 0.7 else None
    return style, shop, wall, roof


def building_height(height: float, levels: float, seed: int) -> float:
    """Hauteur OSM (balise height, sinon étages × 3,2 m), sinon estimée : 3 à 7 étages tirés avec la graine."""
    if np.isfinite(height) and 2.5 <= height <= 200:
        return float(height)
    if np.isfinite(levels) and 1 <= levels <= 60:
        return float(levels) * LEVEL_HEIGHT_M
    return LEVEL_HEIGHT_M * int(np.random.default_rng(seed).integers(3, 8))


def curb_segments(geom) -> list[tuple[float, float, float, float]]:
    """Segments des contours des trottoirs (x0, y0, x1, y1), orientés pour que la normale (dy, -dx) sorte du trottoir."""
    segments = []
    for polygon in _polygons(geom):
        polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
        for ring in (polygon.exterior, *polygon.interiors):
            coords = list(ring.coords)
            segments.extend((x0, y0, x1, y1) for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:])
                            if math.hypot(x1 - x0, y1 - y0) > 1e-6)
    return segments


# ---------------------------------------------------------------------- signalisation
@dataclass(frozen=True)
class PropPlacement:
    kind: str               # clé de catalog.props : "traffic_signals", "stop", "crossing"
    x: float
    y: float
    heading: float          # la face du panneau (-Y du modèle) regarde les véhicules qui arrivent


def push_off_road(x: float, y: float, roads, boundary, margin: float = 0.8) -> tuple[float, float]:
    """Point sur la chaussée : ramené au bord le plus proche, puis ``margin`` mètres au-delà (trottoir, terre-plein)."""
    for _ in range(5):                    # l'espace entre deux chaussées peut être plus étroit que la marge
        point = Point(x, y)
        if not roads.contains(point):
            break
        edge = ops.nearest_points(boundary, point)[0]
        dx, dy = edge.x - x, edge.y - y
        norm = math.hypot(dx, dy) or 1.0
        x, y = edge.x + dx / norm * margin, edge.y + dy / norm * margin
    return x, y


def prop_placements(graph: nx.MultiDiGraph, frame: WorldFrame, roads=None) -> list[PropPlacement]:
    """Un feu ou un panneau par tronçon entrant dans un nœud OSM marqué, à droite de la chaussée.

    ``roads`` (polygones des chaussées, repère local) : un panneau qui tomberait sur une autre chaussée
    (grand carrefour, boulevard à deux chaussées) est repoussé hors de la chaussée.
    """
    boundary = roads.boundary if roads is not None else None
    if roads is not None:
        shapely.prepare(roads)
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
            if roads is not None:
                x, y = push_off_road(x, y, roads, boundary)
            placements.append(PropPlacement(kind, x, y, heading_deg(dx, dy)))
            if kind == "crossing":
                break                       # un seul panneau par passage piéton
    return placements


# ---------------------------------------------------------------------- Panda3D
def _geom_node(name: str, positions: np.ndarray, normals: np.ndarray, colors: np.ndarray, triangles: np.ndarray,
               uvs: np.ndarray | None = None):
    from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexArrayFormat, GeomVertexData,
                              GeomVertexFormat)

    array = GeomVertexArrayFormat()
    array.addColumn("vertex", 3, Geom.NT_float32, Geom.C_point)
    array.addColumn("normal", 3, Geom.NT_float32, Geom.C_normal)
    array.addColumn("color", 4, Geom.NT_float32, Geom.C_color)
    columns = [positions, normals, colors]
    if uvs is not None:
        array.addColumn("texcoord", 2, Geom.NT_float32, Geom.C_texcoord)
        columns.append(uvs)
    fmt = GeomVertexFormat.registerFormat(GeomVertexFormat(array))
    data = GeomVertexData(name, fmt, Geom.UH_static)
    data.uncleanSetNumRows(len(positions))
    memoryview(data.modifyArray(0)).cast("B")[:] = np.hstack(columns).astype(np.float32).tobytes()
    prim = GeomTriangles(Geom.UH_static)
    prim.setIndexType(Geom.NT_uint32)
    prim.modifyVertices().uncleanSetNumRows(triangles.size)
    memoryview(prim.modifyVertices()).cast("B")[:] = triangles.astype(np.uint32).tobytes()
    geom = Geom(data)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def _solid(color, n: int) -> np.ndarray:
    return np.tile([*color, 1.0], (n, 1))


def _indices(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int32).reshape(-1, 3)


def _flat(name: str, geom, z: float, color: tuple[float, float, float], textured: bool = True):
    points, triangles = triangulate(geom)
    if not len(points):
        return None
    positions = np.column_stack([points, np.full(len(points), z)])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    uvs = points / textures.GROUND_TILE_M if textured else None
    return _geom_node(name, positions, normals, _solid(color, len(points)), triangles, uvs)


def _curbs_node(segments: list[tuple[float, float, float, float]]):
    """Bordures : faces verticales entre la chaussée (ROAD_Z) et le dessus du trottoir (SIDEWALK_Z)."""
    if not segments:
        return None
    seg = np.asarray(segments, dtype=np.float64)
    x0, y0, x1, y1 = seg.T
    length = np.hypot(x1 - x0, y1 - y0)
    n = np.column_stack([(y1 - y0) / length, -(x1 - x0) / length, np.zeros(len(seg))])
    low, high = np.full(len(seg), ROAD_Z), np.full(len(seg), SIDEWALK_Z)
    corners = [(x0, y0, low), (x1, y1, low), (x1, y1, high), (x0, y0, low), (x1, y1, high), (x0, y0, high)]
    positions = np.stack([np.column_stack(c) for c in corners], axis=1).reshape(-1, 3)
    return _geom_node("bordures", positions, np.repeat(n, 6, axis=0), _solid(CURB_COLOR, len(positions)),
                      _indices(len(positions)))


def _buildings_nodes(entries: list[tuple[int, Polygon, float]]) -> list[tuple[str, object]]:
    """Bâtiments d'une tuile, regroupés par texture : [(clé de texture, GeomNode)]."""
    groups: dict[str, list] = {}
    for index, polygon, height in entries:
        style, shop, wall, roof = building_style(index)
        shop_band = shop is not None and height >= SHOP_MIN_HEIGHT_M
        z_floors = textures.GROUND_FLOOR_M if shop_band else 0.0
        if shop_band:
            pos, nrm, uv = wall_band(polygon, 0.0, z_floors, textures.SHOP_BAY_M, textures.GROUND_FLOOR_M)
            groups.setdefault(f"vitrine_{shop}", []).append((pos, nrm, _solid(wall, len(pos)), uv))
        pos, nrm, uv = wall_band(polygon, z_floors, height, textures.FACADE_BAY_M, textures.FLOOR_M)
        groups.setdefault(f"facade_{style}", []).append((pos, nrm, _solid(wall, len(pos)), uv))
        roof_pts, _ = triangulate(polygon)
        pos = np.column_stack([roof_pts, np.full(len(roof_pts), height)])
        groups.setdefault("toit", []).append((pos, np.tile([0.0, 0.0, 1.0], (len(pos), 1)), _solid(roof, len(pos)),
                                              roof_pts / textures.GROUND_TILE_M))
    nodes = []
    for key, parts in groups.items():
        pos, nrm, col, uv = (np.vstack([part[i] for part in parts]) for i in range(4))
        nodes.append((key, _geom_node(key, pos, nrm, col, _indices(len(pos)), uv)))
    return nodes


def _transformed(mesh, x: float, y: float, heading: float, scale: float):
    pos, nrm, col, _ = mesh
    a = math.radians(heading)
    rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return pos @ rot.T * scale + np.array([x, y, SIDEWALK_Z]), nrm @ rot.T, col


_FURNITURE_TEMPLATES: dict | None = None


def _furniture_node(items: list):
    """Arbres, palmiers et lampadaires d'une tuile, fusionnés en un seul maillage."""
    global _FURNITURE_TEMPLATES
    if _FURNITURE_TEMPLATES is None:
        _FURNITURE_TEMPLATES = {"palm": [street_furniture.palm(h) for h in (7.5, 9.0, 10.5)],
                                "tree": [street_furniture.round_tree(6.0, v) for v in range(3)],
                                "light": [street_furniture.street_light()]}
    parts = [_transformed(_FURNITURE_TEMPLATES[p.kind][p.variant % len(_FURNITURE_TEMPLATES[p.kind])],
                          p.x, p.y, p.heading, p.scale) for p in items]
    if not parts:
        return None
    pos, nrm, col = (np.vstack([part[i] for part in parts]) for i in range(3))
    return _geom_node("mobilier", pos, nrm, col, _indices(len(pos)))


def city_textures() -> dict:
    """Textures Panda3D de la ville, par clé de groupe."""
    out = {"sol": textures.panda_texture(textures.ground(), "sol"),
           "chaussee": textures.panda_texture(textures.asphalt(), "bitume"),
           "trottoir": textures.panda_texture(textures.pavers(), "dalles"),
           "toit": textures.panda_texture(textures.ground(), "toit")}
    for style in textures.FACADE_STYLES:
        out[f"facade_{style}"] = textures.panda_texture(textures.facade(style), f"facade_{style}")
    for variant in range(4):
        out[f"vitrine_{variant}"] = textures.panda_texture(textures.shopfront(variant), f"vitrine_{variant}")
    return out


def cache_key(graph_file: Path, buildings_file: Path | None) -> str:
    digest = hashlib.sha1(f"v{MESH_VERSION}".encode())
    for path in (graph_file, buildings_file):
        if path is not None and path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _tile_key(x: float, y: float) -> tuple[int, int]:
    return math.floor(x / TILE_M), math.floor(y / TILE_M)


def build_world(graph: nx.MultiDiGraph, frame: WorldFrame, loader, catalog=None, buildings=None,
                cache_file: Path | None = None):
    """NodePath de toute la ville. ``buildings`` : (polygones UTM, hauteurs) ou None.

    Avec ``cache_file`` : relu s'il existe, sinon construit puis enregistré (textures comprises).
    """
    from panda3d.core import BamFile, BamWriter, Filename, NodePath

    if cache_file is not None and cache_file.exists():
        cached = loader.loadModel(Filename.fromOsSpecific(str(cache_file)), noCache=True)
        if cached is not None and not cached.isEmpty():
            return cached

    world = NodePath("ville")
    texs = city_textures()
    street_list = streets(graph, frame)
    layout = road_layout(street_list, crosswalks(graph, frame))
    extent = shapely.union_all([layout.roads, layout.sidewalks]).envelope.buffer(150, join_style="mitre")
    ground = world.attachNewNode(_flat("sol", extent, 0.0, GROUND_COLOR))
    ground.setTexture(texs["sol"])
    # Sol dessiné en premier, sans écrire la profondeur : la chaussée, quelques centimètres au-dessus,
    # ne scintille jamais avec lui, même vue de loin (précision du tampon de profondeur).
    ground.setBin("background", 0)
    ground.setDepthWrite(False)

    footprints: dict[tuple[int, int], list[tuple[int, Polygon, float]]] = {}
    building_polygons: list[Polygon] = []
    if buildings is not None:
        polys, heights = buildings
        keep_out = layout.roads.buffer(0.5)             # un bâtiment OSM qui empiète sur la chaussée est rogné
        shapely.prepare(keep_out)
        for index, (poly, height) in enumerate(zip(polys, heights)):
            local = frame.geometry(poly)
            if keep_out.intersects(local):
                local = local.difference(keep_out)
            for part in _polygons(local):
                if part.area < 12:
                    continue
                c = part.representative_point()
                footprints.setdefault(_tile_key(c.x, c.y), []).append((index, part, height))
                building_polygons.append(part)

    furniture: dict[tuple[int, int], list] = {}
    for placement in street_furniture.placements(street_list, layout.roads, building_polygons):
        furniture.setdefault(_tile_key(placement.x, placement.y), []).append(placement)
    curbs: dict[tuple[int, int], list] = {}
    for segment in curb_segments(layout.sidewalks):
        curbs.setdefault(_tile_key((segment[0] + segment[2]) / 2, (segment[1] + segment[3]) / 2), []).append(segment)

    road_tiles = dict(tiles(layout.roads))
    side_tiles = dict(tiles(layout.sidewalks))
    mark_tiles = dict(tiles(layout.markings))
    grass_tiles = dict(tiles(layout.islands))
    for key in sorted(set(road_tiles) | set(side_tiles) | set(footprints)):
        tile = world.attachNewNode(f"tuile_{key[0]}_{key[1]}")
        for name, parts, z, color in (("chaussee", road_tiles, ROAD_Z, ROAD_COLOR),
                                      ("trottoir", side_tiles, SIDEWALK_Z, SIDEWALK_COLOR),
                                      ("marquage", mark_tiles, MARKING_Z, MARKING_COLOR)):
            if key in parts and (node := _flat(name, parts[key], z, color, textured=name != "marquage")) is not None:
                part = tile.attachNewNode(node)
                if name == "marquage":
                    part.setDepthOffset(2)      # posé sur la chaussée : toujours devant elle
                else:
                    part.setTexture(texs[name])
        if key in grass_tiles and (node := _flat("gazon", grass_tiles[key], SIDEWALK_Z + 0.02, GRASS_COLOR)) is not None:
            tile.attachNewNode(node).setTexture(texs["sol"])
        if key in curbs and (node := _curbs_node(curbs[key])) is not None:
            tile.attachNewNode(node)
        for tex_key, node in _buildings_nodes(footprints.get(key, [])):
            tile.attachNewNode(node).setTexture(texs[tex_key])
        if key in furniture and (node := _furniture_node(furniture[key])) is not None:
            tile.attachNewNode(node)
        tile.flattenStrong()

    if catalog is not None:
        from simulator.render3d.catalog import load_model

        props = world.attachNewNode("signalisation")
        templates = {kind: load_model(entry, loader) for kind, entry in catalog.props.items()}
        for placement in prop_placements(graph, frame, layout.roads):
            copy = templates[placement.kind].copyTo(props)
            copy.setPos(placement.x, placement.y, SIDEWALK_Z)
            copy.setH(placement.heading)
        props.flattenStrong()

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        bam = BamFile()
        bam.openWrite(Filename.fromOsSpecific(str(cache_file)))
        bam.getWriter().setFileTextureMode(BamWriter.BTM_rawdata)   # textures générées : stockées dans le cache
        bam.writeObject(world.node())
        bam.close()
    return world
