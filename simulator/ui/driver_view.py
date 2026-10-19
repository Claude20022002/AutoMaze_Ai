"""Vue conducteur en pseudo-3D : la route vue depuis la voiture, et les obstacles tels que la caméra les voit.

Principe (projection perspective d'un sol plat) : un point situé à ``z`` mètres devant la caméra, ``x`` mètres
sur le côté et ``y`` mètres de haut apparaît à l'écran en

    x_écran = centre + f · x / z        y_écran = horizon + f · (h − y) / z

où ``h`` est la hauteur de la caméra et ``f`` la focale en pixels. Rues, marquages et façades sont des polygones
dessinés du plus lointain au plus proche (algorithme du peintre). Chaque obstacle est un panneau vertical qui
montre **l'image même que le modèle classe** : ce que voit la caméra, c'est exactement l'entrée du classifieur.

Les façades sont un décor généré le long des rues (hauteurs et teintes tirées au hasard) : ce ne sont pas
les vrais bâtiments de Casablanca.
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

CAMERA_HEIGHT_M = 1.5
FIELD_OF_VIEW_DEG = 56.0                 # champ étroit : les obstacles restent lisibles à 80 m
NEAR_M = 1.0
VIEW_DISTANCE_M = 240.0
FACADE_DISTANCE_M = 170.0
MARKING_DISTANCE_M = 90.0
LANE_OFFSET_M = 1.8                      # on roule à droite : caméra et obstacles décalés sur la voie de droite
MIN_SPRITE_PX = 26                       # taille minimale d'affichage d'un obstacle, pour rester lisible au loin
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}

SKY_TOP, SKY_LOW = (122, 166, 214), (214, 226, 238)
GROUND_NEAR, GROUND_FAR = (182, 176, 162), (206, 210, 210)
ASPHALT, ASPHALT_MAJOR, FOG, MARKING = (88, 90, 95), (74, 76, 81), (200, 206, 210), (236, 236, 228)
# Façades décoratives : tons clairs des immeubles de Casablanca.
FACADE_COLORS = np.array([(242, 238, 226), (233, 226, 208), (247, 245, 238), (226, 214, 192), (238, 232, 220)])
LIGHT = np.array([-0.6, -0.8])           # direction d'où vient la lumière (pour ombrer les façades)

# Taille d'affichage de chaque classe (largeur, hauteur) en mètres ; l'image est ajustée sans déformation.
OBSTACLE_SIZE_M = {"person": (0.8, 1.8), "bicycle": (1.7, 1.6), "motorcycle": (1.9, 1.5),
                   "car": (2.8, 1.7), "bus": (3.4, 3.2), "truck": (3.2, 3.4)}


def _lerp(c1, c2, t: float):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def _clip_near(a: np.ndarray, b: np.ndarray, forward: np.ndarray, near: float = NEAR_M):
    """Coupe au plan ``z = near`` la partie des segments [a, b] (relatifs à la caméra) qui passe devant ce plan."""
    za, zb = a @ forward, b @ forward
    keep = np.maximum(za, zb) > near
    a, b, za, zb = a[keep], b[keep], za[keep], zb[keep]
    t_a = np.where(za < near, (near - za) / np.where(zb == za, 1, zb - za), 0.0)[:, None]
    t_b = np.where(zb < near, (near - zb) / np.where(za == zb, 1, za - zb), 0.0)[:, None]
    return a + (b - a) * t_a, b + (a - b) * t_b, keep


class DriverView:
    def __init__(self, graph: nx.MultiDiGraph, rect: pygame.Rect) -> None:
        self.graph, self.rect = graph, rect
        self.focal = (rect.w / 2) / math.tan(math.radians(FIELD_OF_VIEW_DEG / 2))
        self.horizon = rect.y + int(rect.h * 0.42)
        self._heading: np.ndarray | None = None
        self._images: dict = {}
        self._background = self._render_background()
        self._build_segments()
        self._build_markings()
        self._build_facades()

    # ------------------------------------------------------------------ préparation (une seule fois)
    def _build_segments(self) -> None:
        """Découpe chaque rue en segments droits (forme OSM)."""
        starts, ends, widths, owners, seen = [], [], [], [], set()
        self.edge_segments: dict = {}
        for edge in self.graph.edges(keys=True):
            data = self.graph.edges[edge]
            coords = list(data["geometry"].coords) if data.get("geometry") is not None else [
                (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in edge[:2]]
            highway = data.get("highway")
            highway = highway[0] if isinstance(highway, list) else highway
            ids = []
            for a, b in zip(coords, coords[1:]):
                ids.append(len(starts))
                starts.append(a), ends.append(b)
                widths.append(11.0 if highway in MAJOR_ROADS else 7.0)
                # Une rue à double sens apparaît deux fois (u->v et v->u) : on ne la dessine qu'une fois.
                key = (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2], round(a[0] + b[0]), round(a[1] + b[1]))
                owners.append(key not in seen)
                seen.add(key)
            self.edge_segments[edge] = ids
        self.seg_a, self.seg_b = np.array(starts, dtype=float), np.array(ends, dtype=float)
        self.seg_w, self.seg_draw = np.array(widths), np.array(owners)

    def _build_markings(self) -> None:
        """Tirets blancs au milieu de chaque rue (3 m de trait tous les 9 m) : ils donnent la sensation de vitesse."""
        a, b = self.seg_a[self.seg_draw], self.seg_b[self.seg_draw]
        starts, ends = [], []
        for p, q in zip(a, b):
            length = float(np.linalg.norm(q - p))
            direction = (q - p) / (length or 1.0)
            for s in np.arange(2.0, length - 3.0, 9.0):
                starts.append(p + direction * s)
                ends.append(p + direction * (s + 3.0))
        self.mark_a, self.mark_b = np.array(starts), np.array(ends)

    def _build_facades(self) -> None:
        """Façades le long des rues, en retrait du trottoir, coupées près des carrefours."""
        a, b, w = self.seg_a[self.seg_draw], self.seg_b[self.seg_draw], self.seg_w[self.seg_draw]
        d = b - a
        length = np.linalg.norm(d, axis=1)
        direction = d / np.maximum(length, 1e-9)[:, None]
        left = np.column_stack([-direction[:, 1], direction[:, 0]])
        trim = np.minimum(w / 2 + 6.0, length * 0.3)[:, None]
        long_enough = length > 2 * trim[:, 0] + 6.0
        rng = np.random.default_rng(42)
        fa, fb, fn = [], [], []
        for side in (1.0, -1.0):
            setback = (w / 2 + 3.5)[:, None] * side
            fa.append((a + direction * trim + left * setback)[long_enough])
            fb.append((b - direction * trim + left * setback)[long_enough])
            fn.append((-left * side)[long_enough])  # normale tournée vers la rue
        self.fac_a, self.fac_b, self.fac_n = np.vstack(fa), np.vstack(fb), np.vstack(fn)
        self.fac_h = rng.uniform(8.0, 24.0, len(self.fac_a))
        shade = 0.80 + 0.20 * (self.fac_n @ LIGHT)
        self.fac_color = FACADE_COLORS[rng.integers(len(FACADE_COLORS), size=len(self.fac_a))] * shade[:, None]

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
        return np.array(position, dtype=float) + right * LANE_OFFSET_M, forward, right

    def _project(self, rel: np.ndarray, forward, right, height=0.0):
        """Points (N, 2) relatifs à la caméra, à ``height`` mètres du sol -> coordonnées écran (N,), (N,)."""
        z = np.maximum(rel @ forward, NEAR_M * 0.5)
        return (self.rect.centerx + self.focal * (rel @ right) / z,
                self.horizon + self.focal * (CAMERA_HEIGHT_M - height) / z)

    def _ground_quads(self, surface, a_abs, b_abs, widths, colors, eye, forward, right,
                      fog: bool = True, extend: float = 0.3, near: float = NEAR_M) -> None:
        """Bandes au sol de largeur ``widths`` entre a et b, dessinées du plus loin au plus proche.
        ``near`` : distance minimale dessinée (une bande fine trop près de la caméra remplirait l'écran)."""
        if len(a_abs) == 0:
            return
        a, b, keep = _clip_near(a_abs - eye, b_abs - eye, forward, near)
        w = widths[keep][:, None]
        colors = colors[keep]
        direction = b - a
        direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack([-direction[:, 1], direction[:, 0]]) * (w / 2)
        a, b = a - direction * w * extend, b + direction * w * extend  # comble les joints entre segments
        corners = [self._project(c, forward, right) for c in (a + normal, b + normal, b - normal, a - normal)]
        depth = np.maximum(a @ forward, b @ forward)
        for i in np.argsort(-depth):
            shade = colors[i]
            if fog:
                shade = _lerp(shade, FOG, min(depth[i] / VIEW_DISTANCE_M, 1.0) ** 1.3 * 0.85)
            pygame.draw.polygon(surface, shade, [(corners[k][0][i], corners[k][1][i]) for k in range(4)])

    @staticmethod
    def _within(a: np.ndarray, b: np.ndarray, eye: np.ndarray, radius: float) -> np.ndarray:
        """Indices des segments à moins de ``radius`` mètres de la caméra (distance point-segment)."""
        ab = b - a
        t = np.clip(((eye - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return np.nonzero(np.linalg.norm(a + ab * t[:, None] - eye, axis=1) < radius)[0]

    # ------------------------------------------------------------------ dessin
    def draw(self, surface: pygame.Surface, episode, image_for, position, heading) -> None:
        surface.set_clip(self.rect)
        surface.blit(self._background, self.rect.topleft)
        eye, forward, right = self._camera(position, heading)

        # 1. Rues (asphalte), rues bloquées, marquages, guidage de l'itinéraire.
        near = self._within(self.seg_a, self.seg_b, eye, VIEW_DISTANCE_M)
        near = near[self.seg_draw[near]]
        colors = np.array([ASPHALT_MAJOR if w > 8 else ASPHALT for w in self.seg_w[near]])
        self._ground_quads(surface, self.seg_a[near], self.seg_b[near], self.seg_w[near], colors, eye, forward, right)

        blocked = np.array([i for e in episode.costs.blocked for i in self.edge_segments.get(e, [])], dtype=int)
        if len(blocked):
            self._ground_quads(surface, self.seg_a[blocked], self.seg_b[blocked], self.seg_w[blocked] * 0.9,
                               np.tile(theme.BLOCKED, (len(blocked), 1)), eye, forward, right)

        marks = self._within(self.mark_a, self.mark_b, eye, MARKING_DISTANCE_M)
        self._ground_quads(surface, self.mark_a[marks], self.mark_b[marks], np.full(len(marks), 0.18),
                           np.tile(MARKING, (len(marks), 1)), eye, forward, right, extend=0.0, near=3.0)

        car = episode.car
        route = np.array([i for e in car.route.edges[car.edge_index:car.edge_index + 25]
                          for i in self.edge_segments[e]], dtype=int)
        if len(route):
            self._ground_quads(surface, self.seg_a[route], self.seg_b[route], np.full(len(route), 0.55),
                               np.tile(theme.ROUTE, (len(route), 1)), eye, forward, right, fog=False, extend=0.0, near=6.0)

        self._draw_sensor_line(surface, episode)
        # 2. Façades et obstacles, triés ensemble du plus loin au plus proche.
        items = self._facade_items(eye, forward, right) + self._obstacle_items(episode, eye, forward)
        for depth, kind, payload in sorted(items, key=lambda it: -it[0]):
            if kind == "facade":
                points, color, floors = payload
                pygame.draw.polygon(surface, color, points)
                line = _lerp(color, (90, 90, 90), 0.25)
                pygame.draw.lines(surface, line, True, points, 1)
                for left_pt, right_pt in floors:  # un trait par étage sur les façades proches
                    pygame.draw.line(surface, line, left_pt, right_pt, 1)
            else:
                self._draw_obstacle(surface, episode, image_for, payload, depth, forward, right)
        # 3. Capot et informations.
        self._draw_hood(surface)
        self._draw_overlays(surface, episode)
        surface.set_clip(None)

    def _facade_items(self, eye, forward, right) -> list:
        idx = self._within(self.fac_a, self.fac_b, eye, FACADE_DISTANCE_M)
        if len(idx) == 0:
            return []
        a_rel, b_rel = self.fac_a[idx] - eye, self.fac_b[idx] - eye
        visible = ((-(a_rel + b_rel) / 2) * self.fac_n[idx]).sum(1) > 0  # on ne voit que la face côté rue
        idx, a_rel, b_rel = idx[visible], a_rel[visible], b_rel[visible]
        a, b, keep = _clip_near(a_rel, b_rel, forward)
        idx = idx[keep]
        h = self.fac_h[idx]
        ax, ay0 = self._project(a, forward, right)
        bx, by0 = self._project(b, forward, right)
        _, ay1 = self._project(a, forward, right, h)
        _, by1 = self._project(b, forward, right, h)
        depth = ((a + b) / 2) @ forward
        za, zb = np.maximum(a @ forward, NEAR_M * 0.5), np.maximum(b @ forward, NEAR_M * 0.5)
        items = []
        for i in range(len(idx)):
            fade = min(depth[i] / FACADE_DISTANCE_M, 1.0) ** 1.4 * 0.75
            color = _lerp(self.fac_color[idx[i]], FOG, fade)
            floors = []
            if depth[i] < 70:
                for level in np.arange(3.2, h[i] - 1.0, 3.2):
                    y_a = self.horizon + self.focal * (CAMERA_HEIGHT_M - level) / za[i]
                    y_b = self.horizon + self.focal * (CAMERA_HEIGHT_M - level) / zb[i]
                    floors.append(((ax[i], y_a), (bx[i], y_b)))
            quad = [(ax[i], ay0[i]), (bx[i], by0[i]), (bx[i], by1[i]), (ax[i], ay1[i])]
            items.append((depth[i], "facade", (quad, color, floors)))
        return items

    def _obstacle_items(self, episode, eye, forward) -> list:
        items = []
        for state in episode.obstacles:
            if not state.present(episode.step_count):
                continue                      # la personne a fini de traverser
            ob = state.obstacle
            u, v = self.graph.nodes[ob.edge[0]], self.graph.nodes[ob.edge[1]]
            dx, dy = v["x"] - u["x"], v["y"] - u["y"]
            norm = math.hypot(dx, dy) or 1.0
            px, py = point_on_edge(self.graph, ob.edge, ob.offset_m)
            rel = np.array([px + dy / norm * LANE_OFFSET_M, py - dx / norm * LANE_OFFSET_M]) - eye
            z = float(rel @ forward)
            if NEAR_M * 2 < z < VIEW_DISTANCE_M:
                items.append((z, "obstacle", (state, rel)))
        return items

    def _draw_sensor_line(self, surface, episode) -> None:
        """Repère horizontal à la portée du capteur : un obstacle qui le franchit est classé."""
        z = episode.config.sensor_range_m
        y = self.horizon + self.focal * CAMERA_HEIGHT_M / z
        overlay = pygame.Surface((self.rect.w, 2), pygame.SRCALPHA)
        overlay.fill((*theme.ROUTE, 120))
        surface.blit(overlay, (self.rect.x, y))
        theme.draw_text(surface, f"portée du capteur : {z:.0f} m", (self.rect.x + 12, y + 4), 12, theme.PANEL)

    def _image(self, obstacle, image_for) -> pygame.Surface:
        if obstacle not in self._images:
            rgb = np.ascontiguousarray(image_for(obstacle).swapaxes(0, 1))
            self._images[obstacle] = pygame.surfarray.make_surface(rgb)
        return self._images[obstacle]

    def _draw_obstacle(self, surface, episode, image_for, payload, z, forward, right) -> None:
        state, rel = payload
        ob = state.obstacle
        box_w, box_h = OBSTACLE_SIZE_M[ob.true_label]
        image = self._image(ob, image_for)
        fit = min(box_w / image.get_width(), box_h / image.get_height())
        scale = self.focal / z
        w_px, h_px = image.get_width() * fit * scale, image.get_height() * fit * scale
        if h_px < MIN_SPRITE_PX:  # au loin, on agrandit un peu pour que l'image reste lisible
            w_px, h_px = w_px * MIN_SPRITE_PX / h_px, MIN_SPRITE_PX
        if h_px > self.rect.h * 2:
            return
        cx = self.rect.centerx + self.focal * float(rel @ right) / z
        bottom = self.horizon + self.focal * CAMERA_HEIGHT_M / z
        target = pygame.Rect(int(cx - w_px / 2), int(bottom - h_px), max(int(w_px), 2), max(int(h_px), 2))

        shadow = pygame.Surface((target.w, max(target.h // 8, 2)), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 70), shadow.get_rect())
        surface.blit(shadow, (target.x, bottom - shadow.get_height() // 2))
        sprite = pygame.transform.smoothscale(image, target.size)
        fade = min(z / VIEW_DISTANCE_M, 1.0) ** 1.5 * 0.5  # brume au loin
        if fade > 0.02:
            veil = pygame.Surface(target.size, pygame.SRCALPHA)
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
            label, color, verdict = f"?  non identifié · {z:.0f} m", theme.TEXT, None
        else:
            predicted = state.predicted_label or UNKNOWN
            color = theme.CLASS_COLORS[predicted]
            fresh = episode.step_count - (state.detected_step or 0) < 3   # vient d'être classé : cadre épais
            pygame.draw.rect(surface, color, box, 5 if fresh else 3, border_radius=4)
            label = f"{CLASSES_FR.get(predicted, 'Inconnu')} {state.predicted_score:.2f} · {z:.0f} m"
            if predicted == UNKNOWN:
                verdict = ("inconnu : arrêt prudent", theme.WARNING)
            elif predicted == state.obstacle.true_label:
                verdict = ("juste", theme.GOOD)
            else:
                verdict = (f"faux · vrai : {CLASSES_FR[state.obstacle.true_label]}", theme.BLOCKED)
        size = 14 if box.w > 60 else 12
        f = theme.font(size, bold=True)
        tag = pygame.Rect(0, 0, f.size(label)[0] + 12, f.get_height() + 4)
        tag.bottomleft = (box.x, box.y - 2)
        tag.clamp_ip(self.rect)
        pygame.draw.rect(surface, color if state.handled else theme.PANEL, tag, border_radius=4)
        theme.draw_text(surface, label, (tag.x + 6, tag.y + 2), size, theme.PANEL if state.handled else theme.TEXT, bold=True)
        if verdict:
            text, vcolor = verdict
            vtag = pygame.Rect(0, 0, theme.font(12, bold=True).size(text)[0] + 12, 20)
            vtag.topleft = (box.x, box.bottom + 4)
            vtag.clamp_ip(self.rect)
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
        dash = pygame.Rect(r.centerx - 100, r.bottom - 52, 200, 44)
        self._glass(surface, dash)
        theme.draw_text(surface, f"{speed_kmh:.0f} km/h", (dash.x + 16, dash.y + 6), 22, theme.PANEL, bold=True)
        if episode.done:
            status = episode.result.status
            state, color = status.upper(), theme.GOOD if status == "arrivé" else theme.BLOCKED
        elif speed_kmh == 0:
            state, color = "ARRÊT", theme.WARNING
        elif speed_kmh < episode.config.car_speed_mps * 3.6 - 0.1:
            state, color = "RALENTI", theme.WARNING
        else:
            state, color = "", theme.PANEL
        theme.draw_text(surface, state, (dash.x + 116, dash.y + 13), 13, color, bold=True)

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
        coords = [(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in edge[:2]]
    (x1, y1), (x2, y2) = (coords[-2], coords[-1]) if end else (coords[0], coords[1])
    n = math.hypot(x2 - x1, y2 - y1) or 1.0
    return (x2 - x1) / n, (y2 - y1) / n
