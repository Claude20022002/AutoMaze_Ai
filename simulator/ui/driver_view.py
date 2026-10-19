"""Vue conducteur en pseudo-3D : la route vue depuis la voiture, et les obstacles tels que la caméra les voit.

Principe (projection perspective d'un sol plat) : un point du sol situé à ``z`` mètres devant la caméra et ``x``
mètres sur le côté apparaît à l'écran en

    x_écran = centre + f · x / z        y_écran = horizon + f · h / z

où ``h`` est la hauteur de la caméra et ``f`` la focale en pixels. Les rues sont des bandes au sol, dessinées de
la plus lointaine à la plus proche. Chaque obstacle est un panneau vertical qui montre **l'image même que le
modèle classe** : ce que voit la caméra, c'est exactement l'entrée du classifieur.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pygame

from simulator.contracts import CLASSES_FR, UNKNOWN
from simulator.ui import theme
from simulator.world.car import point_on_edge
from simulator.world.road_map import street_name

CAMERA_HEIGHT_M = 1.4
FIELD_OF_VIEW_DEG = 72.0
NEAR_M = 1.0
VIEW_DISTANCE_M = 240.0
LANE_OFFSET_M = 1.8                      # on roule à droite : caméra et obstacles décalés sur la voie de droite
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}

SKY_TOP, SKY_LOW = (122, 166, 214), (214, 226, 238)
GROUND_NEAR, GROUND_FAR = (182, 176, 162), (206, 210, 210)
ASPHALT, ASPHALT_MAJOR, FOG = (88, 90, 95), (74, 76, 81), (200, 206, 210)

# Taille d'affichage de chaque classe (largeur, hauteur) en mètres ; l'image est ajustée sans déformation.
OBSTACLE_SIZE_M = {"person": (0.8, 1.8), "bicycle": (1.7, 1.6), "motorcycle": (1.9, 1.5),
                   "car": (2.8, 1.7), "bus": (3.4, 3.2), "truck": (3.2, 3.4)}


def _lerp(c1, c2, t: float):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


class DriverView:
    def __init__(self, graph: nx.MultiDiGraph, rect: pygame.Rect) -> None:
        self.graph, self.rect = graph, rect
        self.focal = (rect.w / 2) / math.tan(math.radians(FIELD_OF_VIEW_DEG / 2))
        self.horizon = rect.y + int(rect.h * 0.42)
        self._heading: np.ndarray | None = None
        self._images: dict = {}
        self._background = self._render_background()
        self._build_segments()

    # ------------------------------------------------------------------ préparation
    def _build_segments(self) -> None:
        """Découpe chaque rue en segments droits (forme OSM), une seule fois au démarrage."""
        starts, ends, widths, owners, seen = [], [], [], [], set()
        self.edge_segments: dict = {}
        for edge in self.graph.edges(keys=True):
            data = self.graph.edges[edge]
            coords = list(data["geometry"].coords) if data.get("geometry") is not None else [
                (self.graph.nodes[edge[0]]["x"], self.graph.nodes[edge[0]]["y"]),
                (self.graph.nodes[edge[1]]["x"], self.graph.nodes[edge[1]]["y"])]
            highway = data.get("highway")
            highway = highway[0] if isinstance(highway, list) else highway
            ids = []
            for a, b in zip(coords, coords[1:]):
                ids.append(len(starts))
                starts.append(a), ends.append(b)
                widths.append(11.0 if highway in MAJOR_ROADS else 7.0)
                # Une rue à double sens apparaît deux fois (u->v et v->u) : on ne dessine l'asphalte qu'une fois.
                key = (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2], round(a[0] + b[0]), round(a[1] + b[1]))
                owners.append(key not in seen)
                seen.add(key)
            self.edge_segments[edge] = ids
        self.seg_a, self.seg_b = np.array(starts), np.array(ends)
        self.seg_w, self.seg_draw = np.array(widths), np.array(owners)

    def _render_background(self) -> pygame.Surface:
        surface = pygame.Surface(self.rect.size)
        h_line = self.horizon - self.rect.y
        for y in range(self.rect.h):
            if y < h_line:
                color = _lerp(SKY_TOP, SKY_LOW, y / max(h_line, 1))
            else:
                color = _lerp(GROUND_FAR, GROUND_NEAR, min((y - h_line) / (self.rect.h - h_line) * 1.6, 1.0))
            pygame.draw.line(surface, color, (0, y), (self.rect.w, y))
        return surface

    # ------------------------------------------------------------------ projection
    def _camera(self, position, heading):
        target = np.array(heading, dtype=float)
        if self._heading is None:
            self._heading = target
        else:  # lissage : la caméra tourne progressivement aux carrefours
            mixed = self._heading * 0.7 + target * 0.3
            self._heading = mixed / (np.linalg.norm(mixed) or 1.0)
        forward = self._heading
        right = np.array([forward[1], -forward[0]])
        eye = np.array(position, dtype=float) + right * LANE_OFFSET_M
        return eye, forward, right

    def _to_screen(self, rel: np.ndarray, forward, right, height: float = 0.0):
        """rel : points (N, 2) relatifs à la caméra -> (x_écran, y_écran, profondeur z)."""
        z = rel @ forward
        x = rel @ right
        z_safe = np.maximum(z, NEAR_M * 0.5)
        sx = self.rect.centerx + self.focal * x / z_safe
        sy = self.horizon + self.focal * (CAMERA_HEIGHT_M - height) / z_safe
        return sx, sy, z

    def _draw_strips(self, surface, ids, eye, forward, right, width=None, color=None, fog=True) -> None:
        """Dessine des segments de rue comme des bandes au sol, du plus loin au plus proche."""
        if len(ids) == 0:
            return
        a, b = self.seg_a[ids] - eye, self.seg_b[ids] - eye
        w = self.seg_w[ids] if width is None else np.full(len(ids), width)
        za, zb = a @ forward, b @ forward
        keep = np.maximum(za, zb) > NEAR_M
        a, b, w, za, zb, ids = a[keep], b[keep], w[keep], za[keep], zb[keep], np.asarray(ids)[keep]
        # Coupe au plan proche la partie du segment qui passe derrière la caméra.
        t_a = np.where(za < NEAR_M, (NEAR_M - za) / np.where(zb - za == 0, 1, zb - za), 0.0)[:, None]
        t_b = np.where(zb < NEAR_M, (NEAR_M - zb) / np.where(za - zb == 0, 1, za - zb), 0.0)[:, None]
        a, b = a + (b - a) * t_a, b + (a - b) * t_b
        direction = b - a
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.where(length == 0, 1, length)
        normal = np.column_stack([-direction[:, 1], direction[:, 0]]) * (w[:, None] / 2)
        a_ext, b_ext = a - direction * (w[:, None] * 0.3), b + direction * (w[:, None] * 0.3)  # comble les joints
        corners = [a_ext + normal, b_ext + normal, b_ext - normal, a_ext - normal]
        projected = [self._to_screen(c, forward, right) for c in corners]
        depth = np.maximum(za[keep] if False else np.maximum(a @ forward, b @ forward), NEAR_M)
        for i in np.argsort(-depth):
            if depth[i] > VIEW_DISTANCE_M * 1.2:
                continue
            points = [(projected[k][0][i], projected[k][1][i]) for k in range(4)]
            base = color if color is not None else (ASPHALT_MAJOR if self.seg_w[ids[i]] > 8 else ASPHALT)
            shade = _lerp(base, FOG, min(depth[i] / VIEW_DISTANCE_M, 1.0) ** 1.3 * 0.85) if fog else base
            pygame.draw.polygon(surface, shade, points)

    def _nearby(self, eye) -> np.ndarray:
        """Segments à moins de VIEW_DISTANCE_M de la caméra (distance point-segment)."""
        ab = self.seg_b - self.seg_a
        t = np.clip(((eye - self.seg_a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        closest = self.seg_a + ab * t[:, None]
        return np.nonzero(np.linalg.norm(closest - eye, axis=1) < VIEW_DISTANCE_M)[0]

    # ------------------------------------------------------------------ dessin
    def draw(self, surface: pygame.Surface, episode, image_for, position, heading) -> None:
        surface.set_clip(self.rect)
        surface.blit(self._background, self.rect.topleft)
        eye, forward, right = self._camera(position, heading)

        near = self._nearby(eye)
        self._draw_strips(surface, near[self.seg_draw[near]], eye, forward, right)

        # Rues bloquées en rouge, puis ligne de guidage bleue sur l'itinéraire restant.
        blocked = [i for e in episode.costs.blocked for i in self.edge_segments.get(e, [])]
        self._draw_strips(surface, np.array(blocked, dtype=int), eye, forward, right, width=4.0, color=theme.BLOCKED)
        car = episode.car
        route = [i for e in car.route.edges[car.edge_index:car.edge_index + 25] for i in self.edge_segments[e]]
        self._draw_strips(surface, np.array(route, dtype=int), eye, forward, right, width=1.2, color=theme.ROUTE, fog=False)

        self._draw_sensor_line(surface, episode)
        self._draw_obstacles(surface, episode, image_for, eye, forward, right)
        self._draw_hood(surface)
        self._draw_overlays(surface, episode)
        surface.set_clip(None)

    def _draw_sensor_line(self, surface, episode) -> None:
        """Repère horizontal à la portée du capteur : un obstacle qui le franchit est classé."""
        z = episode.config.sensor_range_m
        y = self.horizon + self.focal * CAMERA_HEIGHT_M / z
        overlay = pygame.Surface((self.rect.w, 2), pygame.SRCALPHA)
        overlay.fill((*theme.ROUTE, 110))
        surface.blit(overlay, (self.rect.x, y))
        theme.draw_text(surface, f"portée du capteur : {z:.0f} m", (self.rect.x + 12, y - 20), 12, theme.PANEL)

    def _image(self, obstacle, image_for) -> pygame.Surface:
        if obstacle not in self._images:
            rgb = np.ascontiguousarray(image_for(obstacle).swapaxes(0, 1))
            self._images[obstacle] = pygame.surfarray.make_surface(rgb)
        return self._images[obstacle]

    def _draw_obstacles(self, surface, episode, image_for, eye, forward, right) -> None:
        step = episode.step_count
        items = []
        for state in episode.obstacles:
            if not state.present(step):
                continue                      # la personne a fini de traverser
            ob = state.obstacle
            u, v = self.graph.nodes[ob.edge[0]], self.graph.nodes[ob.edge[1]]
            dx, dy = v["x"] - u["x"], v["y"] - u["y"]
            norm = math.hypot(dx, dy) or 1.0
            px, py = point_on_edge(self.graph, ob.edge, ob.offset_m)
            world = np.array([px + dy / norm * LANE_OFFSET_M, py - dx / norm * LANE_OFFSET_M])
            rel = world - eye
            z = float(rel @ forward)
            if NEAR_M * 2 < z < VIEW_DISTANCE_M:
                items.append((z, state, rel))

        for z, state, rel in sorted(items, key=lambda it: -it[0]):
            ob = state.obstacle
            box_w, box_h = OBSTACLE_SIZE_M[ob.true_label]
            image = self._image(ob, image_for)
            fit = min(box_w / image.get_width(), box_h / image.get_height())
            scale = self.focal / z
            w_px, h_px = max(int(image.get_width() * fit * scale), 2), max(int(image.get_height() * fit * scale), 2)
            if h_px > self.rect.h * 2:
                continue
            cx = self.rect.centerx + self.focal * float(rel @ right) / z
            bottom = self.horizon + self.focal * CAMERA_HEIGHT_M / z
            target = pygame.Rect(int(cx - w_px / 2), int(bottom - h_px), w_px, h_px)

            shadow = pygame.Surface((w_px, max(h_px // 8, 2)), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow, (0, 0, 0, 70), shadow.get_rect())
            surface.blit(shadow, (target.x, bottom - shadow.get_height() // 2))
            sprite = pygame.transform.smoothscale(image, (w_px, h_px))
            fade = min(z / VIEW_DISTANCE_M, 1.0) ** 1.5 * 0.6  # brume au loin
            if fade > 0.02:
                veil = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
                veil.fill((*FOG, int(255 * fade)))
                sprite.blit(veil, (0, 0))
            surface.blit(sprite, target)
            self._draw_detection_box(surface, episode, state, target, z)

    def _draw_detection_box(self, surface, episode, state, target: pygame.Rect, z: float) -> None:
        """Cadre façon caméra embarquée : « ? » avant la détection, puis classe prédite, score et verdict."""
        pad = max(4, target.h // 12)
        box = target.inflate(pad * 2, pad * 2)
        if not state.handled:
            pygame.draw.rect(surface, theme.PANEL, box, 2, border_radius=4)
            label, color, verdict = f"?  objet non identifié · {z:.0f} m", theme.TEXT, None
        else:
            predicted = state.predicted_label or UNKNOWN
            color = theme.CLASS_COLORS[predicted]
            fresh = episode.step_count - (state.detected_step or 0) < 3
            pygame.draw.rect(surface, color, box, 5 if fresh else 3, border_radius=4)
            name = CLASSES_FR.get(predicted, "Inconnu")
            label = f"{name} {state.predicted_score:.2f} · {z:.0f} m"
            if predicted == UNKNOWN:
                verdict = ("inconnu", theme.WARNING)
            elif predicted == state.obstacle.true_label:
                verdict = ("juste", theme.GOOD)
            else:
                verdict = (f"faux · vrai : {CLASSES_FR[state.obstacle.true_label]}", theme.BLOCKED)
        size = 14 if box.w > 60 else 12
        f = theme.font(size, bold=True)
        tag_w = f.size(label)[0] + 12
        tag = pygame.Rect(box.x, box.y - f.get_height() - 6, tag_w, f.get_height() + 4)
        pygame.draw.rect(surface, color if state.handled else theme.PANEL, tag, border_radius=4)
        theme.draw_text(surface, label, (tag.x + 6, tag.y + 2), size, theme.PANEL if state.handled else theme.TEXT, bold=True)
        if verdict:
            text, vcolor = verdict
            vtag = pygame.Rect(box.x, box.bottom + 4, theme.font(12, bold=True).size(text)[0] + 12, 20)
            pygame.draw.rect(surface, vcolor, vtag, border_radius=4)
            theme.draw_text(surface, text, (vtag.x + 6, vtag.y + 1), 12, theme.PANEL, bold=True)

    def _draw_hood(self, surface) -> None:
        r = self.rect
        hood = [(r.x + r.w * 0.08, r.bottom), (r.x + r.w * 0.30, r.bottom - 58),
                (r.x + r.w * 0.70, r.bottom - 58), (r.x + r.w * 0.92, r.bottom)]
        pygame.draw.polygon(surface, (34, 36, 40), hood)
        pygame.draw.line(surface, (70, 72, 78), hood[1], hood[2], 2)

    def _draw_overlays(self, surface, episode) -> None:
        r, car = self.rect, episode.car
        # Rue actuelle et prochaine manœuvre (en haut à gauche, comme un GPS).
        street = theme.latin(street_name(self.graph, car.current_edge)) if car.current_edge else "Arrivé"
        panel = pygame.Rect(r.x + 14, r.y + 14, 330, 76)
        self._glass(surface, panel)
        theme.draw_text(surface, street, (panel.x + 12, panel.y + 8), 17, theme.PANEL, bold=True, max_width=panel.w - 24)
        turn = next_turn(self.graph, car)
        if turn is not None:
            direction, distance, next_street = turn
            arrow = {"gauche": "←", "droite": "→", "demi-tour": "↶"}[direction]
            theme.draw_text(surface, f"{arrow}  {direction} dans {distance:.0f} m", (panel.x + 12, panel.y + 34), 15, theme.PANEL)
            theme.draw_text(surface, theme.latin(next_street), (panel.x + 12, panel.y + 54), 13, (210, 214, 220), max_width=panel.w - 24)
        else:
            theme.draw_text(surface, "Tout droit jusqu'à l'arrivée", (panel.x + 12, panel.y + 34), 15, theme.PANEL)

        # Tableau de bord : vitesse et état.
        speed_kmh = episode.speed_mps * 3.6
        dash = pygame.Rect(r.centerx - 90, r.bottom - 52, 180, 44)
        self._glass(surface, dash)
        theme.draw_text(surface, f"{speed_kmh:.0f} km/h", (dash.x + 16, dash.y + 6), 22, theme.PANEL, bold=True)
        state, color = ("ARRÊT", theme.WARNING) if speed_kmh == 0 and not episode.done else (
            ("RALENTI", theme.WARNING) if speed_kmh < episode.config.car_speed_mps * 3.6 - 0.1 and not episode.done else ("", theme.PANEL))
        if episode.done:
            state, color = episode.result.status.upper(), theme.GOOD if episode.result.status == "arrivé" else theme.BLOCKED
        theme.draw_text(surface, state, (dash.x + 112, dash.y + 13), 13, color, bold=True)

        # Collision : flash rouge pendant deux pas.
        if episode.last_collision_step is not None and episode.step_count - episode.last_collision_step < 2:
            flash = pygame.Surface(r.size, pygame.SRCALPHA)
            flash.fill((*theme.BLOCKED, 90))
            surface.blit(flash, r.topleft)
            theme.draw_text(surface, "COLLISION", (r.centerx - 70, r.centery - 20), 30, theme.PANEL, bold=True)

    @staticmethod
    def _glass(surface, rect: pygame.Rect) -> None:
        glass = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glass, (20, 24, 30, 175), glass.get_rect(), border_radius=8)
        surface.blit(glass, rect.topleft)


def next_turn(graph: nx.MultiDiGraph, car, min_angle_deg: float = 35.0):
    """Prochain changement de direction sur l'itinéraire : (sens, distance en m, nom de la rue suivante) ou None."""
    edges = car.route.edges
    for i in range(car.edge_index, len(edges) - 1):
        a, b = _bearing(graph, edges[i], end=True), _bearing(graph, edges[i + 1], end=False)
        angle = math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
        if abs(angle) >= min_angle_deg:
            direction = "demi-tour" if abs(angle) > 150 else ("gauche" if angle > 0 else "droite")
            return direction, car.distance_to(i + 1, 0.0), street_name(graph, edges[i + 1])
    return None


def _bearing(graph, edge, end: bool) -> tuple[float, float]:
    data = graph.edges[edge]
    if data.get("geometry") is not None:
        coords = list(data["geometry"].coords)
    else:
        coords = [(graph.nodes[edge[0]]["x"], graph.nodes[edge[0]]["y"]), (graph.nodes[edge[1]]["x"], graph.nodes[edge[1]]["y"])]
    (x1, y1), (x2, y2) = (coords[-2], coords[-1]) if end else (coords[0], coords[1])
    n = math.hypot(x2 - x1, y2 - y1) or 1.0
    return (x2 - x1) / n, (y2 - y1) / n
