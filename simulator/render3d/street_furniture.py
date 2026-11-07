"""Végétation et mobilier urbain procéduraux : palmiers, arbres d'alignement, lampadaires.

Casablanca est connue pour ses boulevards bordés de palmiers : on en plante le long des grands axes (primary,
secondary, trunk), des arbres ronds le long des rues tertiaires, quelques arbres dans les rues résidentielles,
et des lampadaires le long des axes principaux. Ce décor est GÉNÉRÉ (pas de données OSM sur les arbres ici).

Géométrie en Python pur : chaque objet est un maillage (positions, normales, couleurs, triangles) en mètres,
base au sol (z = 0), « avant » vers +Y. Les emplacements évitent la chaussée, les bâtiments et les carrefours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Point

Mesh = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]   # positions, normales, couleurs RGBA, triangles

TRUNK = (0.47, 0.36, 0.25)
FROND = (0.30, 0.52, 0.22)
CROWN = ((0.33, 0.55, 0.27), (0.27, 0.48, 0.24), (0.38, 0.58, 0.30))
POLE = (0.28, 0.30, 0.32)
LAMP = (1.0, 0.96, 0.82)

SPACING_M = {"major": 16.0, "medium": 20.0, "minor": 34.0}   # distance entre deux arbres d'un même côté
INTERSECTION_CLEAR_M = 10.0
LIGHT_SPACING_M = 32.0


class _Builder:
    def __init__(self) -> None:
        self.pos: list = []
        self.nrm: list = []
        self.col: list = []

    def tri(self, a, b, c, color, double_sided: bool = False) -> None:
        a, b, c = map(np.asarray, (a, b, c))
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        self.pos += [a, b, c]
        self.nrm += [n, n, n]
        self.col += [(*color, 1.0)] * 3
        if double_sided:
            self.pos += [a, c, b]
            self.nrm += [-n, -n, -n]
            self.col += [(*color, 1.0)] * 3

    def prism(self, sides: int, z0: float, z1: float, r0: float, r1: float, color, center=(0.0, 0.0)) -> None:
        """Tronc ou poteau : prisme (éventuellement conique) sans couvercles."""
        cx, cy = center
        for i in range(sides):
            a0, a1 = 2 * math.pi * i / sides, 2 * math.pi * (i + 1) / sides
            p00 = (cx + r0 * math.cos(a0), cy + r0 * math.sin(a0), z0)
            p01 = (cx + r0 * math.cos(a1), cy + r0 * math.sin(a1), z0)
            p10 = (cx + r1 * math.cos(a0), cy + r1 * math.sin(a0), z1)
            p11 = (cx + r1 * math.cos(a1), cy + r1 * math.sin(a1), z1)
            self.tri(p00, p01, p11, color)
            self.tri(p00, p11, p10, color)

    def mesh(self) -> Mesh:
        pos = np.asarray(self.pos, dtype=np.float64).reshape(-1, 3)
        return (pos, np.asarray(self.nrm, dtype=np.float64).reshape(-1, 3),
                np.asarray(self.col, dtype=np.float64).reshape(-1, 4), np.arange(len(pos), dtype=np.int32).reshape(-1, 3))


def palm(height: float = 8.0, fronds: int = 8) -> Mesh:
    """Palmier : tronc conique à 6 faces et palmes retombantes à double face."""
    b = _Builder()
    b.prism(6, 0.0, height, 0.26, 0.17, TRUNK)
    top = np.array([0.0, 0.0, height])
    for i in range(fronds):
        a = 2 * math.pi * i / fronds
        d = np.array([math.cos(a), math.sin(a), 0.0])
        side = np.array([-math.sin(a), math.cos(a), 0.0]) * 0.55
        mid = top + d * 1.8 + np.array([0, 0, 0.45])
        tip = top + d * 3.4 + np.array([0, 0, -1.1])
        shade = tuple(min(1.0, c * (0.9 + 0.2 * (i % 2))) for c in FROND)
        b.tri(top, mid + side, mid - side, shade, double_sided=True)
        b.tri(mid - side, mid + side, tip, shade, double_sided=True)
    return b.mesh()


def round_tree(height: float = 6.0, variant: int = 0) -> Mesh:
    """Arbre d'alignement : tronc et houppier octaédrique (8 faces, à plat, style low-poly)."""
    b = _Builder()
    b.prism(5, 0.0, height * 0.45, 0.18, 0.14, TRUNK)
    color = CROWN[variant % len(CROWN)]
    c = np.array([0.0, 0.0, height * 0.68])
    r, h = height * 0.33, height * 0.34
    ring = [c + np.array([r * math.cos(a), r * math.sin(a), 0.0]) for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
    top, bottom = c + np.array([0, 0, h]), c - np.array([0, 0, h * 0.8])
    for p, q in zip(ring, ring[1:] + ring[:1]):
        b.tri(p, q, top, color)
        b.tri(q, p, bottom, tuple(x * 0.85 for x in color))
    return b.mesh()


def street_light(height: float = 7.0, arm: float = 1.6) -> Mesh:
    """Lampadaire : poteau, bras vers +Y (au-dessus de la chaussée) et lanterne."""
    b = _Builder()
    b.prism(6, 0.0, height, 0.09, 0.07, POLE)
    for i in range(4):                                       # bras horizontal à section carrée
        a0, a1 = math.pi / 4 + i * math.pi / 2, math.pi / 4 + (i + 1) * math.pi / 2
        o0 = np.array([0.05 * math.cos(a0), 0.0, 0.05 * math.sin(a0)])
        o1 = np.array([0.05 * math.cos(a1), 0.0, 0.05 * math.sin(a1)])
        start, end = np.array([0.0, 0.0, height]), np.array([0.0, arm, height])
        b.tri(start + o0, end + o0, end + o1, POLE, double_sided=True)
        b.tri(start + o0, end + o1, start + o1, POLE, double_sided=True)
    lamp = np.array([0.0, arm, height - 0.12])
    corners = [lamp + np.array(c) for c in ((-0.28, -0.14, 0), (0.28, -0.14, 0), (0.28, 0.14, 0), (-0.28, 0.14, 0))]
    b.tri(corners[0], corners[2], corners[1], LAMP, double_sided=True)
    b.tri(corners[0], corners[3], corners[2], LAMP, double_sided=True)
    return b.mesh()


@dataclass(frozen=True)
class Placement:
    kind: str           # "palm", "tree", "light"
    x: float
    y: float
    heading: float      # degrés (cap Panda3D) ; pour un lampadaire, le bras pointe vers la chaussée
    scale: float
    variant: int


def street_class(highway) -> str:
    highway = str(highway[0] if isinstance(highway, list) and highway else highway or "").removesuffix("_link")
    if highway in ("trunk", "primary", "secondary", "motorway"):
        return "major"
    if highway == "tertiary":
        return "medium"
    return "minor"


def placements(streets, roads, buildings, seed: int = 7, sidewalk_m: float = 2.5) -> list[Placement]:
    """Emplacements des arbres et lampadaires le long des rues (``streets`` : ``world_mesh.Street``).

    ``roads`` : polygones des chaussées ; ``buildings`` : liste de contours de bâtiments (repère local) ou None.
    Déterministe pour une graine donnée.
    """
    from simulator.render3d.world_mesh import heading_deg

    rng = np.random.default_rng(seed)
    shapely.prepare(roads)
    tree = shapely.STRtree(list(buildings)) if buildings else None
    taken: list[tuple[float, float]] = []
    out: list[Placement] = []

    def free(x: float, y: float, radius: float) -> bool:
        if roads.intersects(Point(x, y).buffer(0.4, quad_segs=2)):
            return False
        if tree is not None and len(tree.query(Point(x, y).buffer(radius, quad_segs=2), predicate="intersects")):
            return False
        return all((x - a) ** 2 + (y - b) ** 2 > (2 * radius) ** 2 for a, b in taken[-60:])

    for street in streets:
        line, half = street.line, street.width / 2
        length = line.length
        if length < 2 * INTERSECTION_CLEAR_M + 4:
            continue
        cls = street.kind
        sides = (1, -1) if cls != "minor" else (1 if rng.random() < 0.5 else -1,)
        for side in sides:
            spacing = SPACING_M[cls]
            position = INTERSECTION_CLEAR_M + rng.uniform(0, spacing / 2)
            while position < length - INTERSECTION_CLEAR_M:
                p, q = line.interpolate(position), line.interpolate(min(position + 1.0, length))
                dx, dy = q.x - p.x, q.y - p.y
                norm = math.hypot(dx, dy) or 1.0
                dx, dy = dx / norm, dy / norm
                offset = half + sidewalk_m * 0.62
                x, y = p.x + dy * offset * side, p.y - dx * offset * side
                if free(x, y, 1.2):
                    kind = "palm" if cls == "major" else "tree"
                    out.append(Placement(kind, x, y, float(rng.uniform(0, 360)), float(rng.uniform(0.85, 1.15)),
                                         int(rng.integers(3))))
                    taken.append((x, y))
                position += spacing
        if cls in ("major", "medium"):                    # lampadaires d'un seul côté, décalés des arbres
            position = INTERSECTION_CLEAR_M + LIGHT_SPACING_M / 2
            while position < length - INTERSECTION_CLEAR_M:
                p, q = line.interpolate(position), line.interpolate(min(position + 1.0, length))
                dx, dy = q.x - p.x, q.y - p.y
                norm = math.hypot(dx, dy) or 1.0
                dx, dy = dx / norm, dy / norm
                offset = half + 0.5
                x, y = p.x - dy * offset, p.y + dx * offset        # côté gauche : bras vers la droite (chaussée)
                if free(x, y, 0.6):
                    out.append(Placement("light", x, y, heading_deg(dy, -dx), 1.0, 0))
                    taken.append((x, y))
                position += LIGHT_SPACING_M
    return out
