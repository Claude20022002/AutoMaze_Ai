"""Dessin de la carte : rues, itinéraires, obstacles et voiture."""

from __future__ import annotations

import networkx as nx
import pygame

from simulator.ui import theme
from simulator.world.car import point_on_edge

MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}


class MapView:
    def __init__(self, graph: nx.MultiDiGraph, rect: pygame.Rect, focus_nodes=None,
                 padding_m: float = 450.0, margin: int = 24) -> None:
        """``focus_nodes`` : intersections à cadrer (par exemple départ et arrivée), avec ``padding_m`` mètres
        autour pour voir les détours. Sans ``focus_nodes``, toute la carte est affichée."""
        self.graph, self.rect = graph, rect
        nodes = focus_nodes or graph.nodes
        xs = [graph.nodes[n]["x"] for n in nodes]
        ys = [graph.nodes[n]["y"] for n in nodes]
        pad = padding_m if focus_nodes else 0.0
        min_x, max_x, min_y, max_y = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        self.min_x, self.min_y = min_x, min_y
        self.scale = min((rect.w - 2 * margin) / (max_x - min_x), (rect.h - 2 * margin) / (max_y - min_y))
        # Centre la zone cadrée dans son cadre.
        self.offset_x = rect.x + (rect.w - (max_x - min_x) * self.scale) / 2
        self.offset_y = rect.y + (rect.h + (max_y - min_y) * self.scale) / 2
        self._points: dict = {}
        self.base = self._render_streets()

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Mètres (UTM) -> pixels. L'axe y de l'écran est inversé (le nord est en haut)."""
        return round(self.offset_x + (x - self.min_x) * self.scale), round(self.offset_y - (y - self.min_y) * self.scale)

    def edge_points(self, edge) -> list[tuple[int, int]]:
        if edge not in self._points:
            data = self.graph.edges[edge]
            if data.get("geometry") is not None:
                coords = list(data["geometry"].coords)
            else:
                u, v = self.graph.nodes[edge[0]], self.graph.nodes[edge[1]]
                coords = [(u["x"], u["y"]), (v["x"], v["y"])]
            self._points[edge] = [self.to_screen(x, y) for x, y in coords]
        return self._points[edge]

    def _render_streets(self) -> pygame.Surface:
        surface = pygame.Surface(self.rect.size)
        surface.fill(theme.BACKGROUND)
        for edge in self.graph.edges(keys=True):
            highway = self.graph.edges[edge].get("highway")
            highway = highway[0] if isinstance(highway, list) else highway
            width = 3 if highway in MAJOR_ROADS else 1
            points = [(x - self.rect.x, y - self.rect.y) for x, y in self.edge_points(edge)]
            pygame.draw.lines(surface, theme.STREET, False, points, width)
        return surface

    def _polyline(self, surface, edges, color, width) -> None:
        for edge in edges:
            pygame.draw.lines(surface, color, False, self.edge_points(edge), width)

    def draw(self, surface: pygame.Surface, episode, start_name: str, goal_name: str) -> None:
        surface.blit(self.base, self.rect.topleft)
        car = episode.car

        self._polyline(surface, episode.planned.edges, theme.PLANNED, 3)              # itinéraire initial
        self._polyline(surface, car.route.edges[car.edge_index:], theme.ROUTE, 4)     # itinéraire actuel
        self._polyline(surface, car.driven_edges, theme.DRIVEN, 3)                     # chemin parcouru
        self._polyline(surface, list(episode.costs.blocked), theme.BLOCKED, 4)         # rues bloquées

        for name, node, label in ((start_name, episode.start, "A"), (goal_name, episode.goal, "B")):
            x, y = self.to_screen(self.graph.nodes[node]["x"], self.graph.nodes[node]["y"])
            pygame.draw.circle(surface, theme.TEXT, (x, y), 11)
            theme.draw_text(surface, label, (x - 5, y - 11), 15, theme.PANEL, bold=True)
            width, height = theme.font(13, bold=True).size(name)
            if x + 14 + width < self.rect.right - 8:      # à droite du point
                label = pygame.Rect(x + 14, y - height // 2, width, height)
            else:                                         # au-dessus, si la droite sort de la carte
                label = pygame.Rect(min(x - width // 2, self.rect.right - 8 - width), y - 16 - height, width, height)
            pygame.draw.rect(surface, theme.PANEL, label.inflate(8, 2), border_radius=4)
            theme.draw_text(surface, name, label.topleft, 13, theme.TEXT, bold=True)

        for i, state in enumerate(episode.obstacles):
            obstacle = state.obstacle
            pos = self.to_screen(*point_on_edge(self.graph, obstacle.edge, obstacle.offset_m))
            present = state.present(episode.step_count)
            if not state.handled:                       # pas encore vu : la voiture ne sait pas ce que c'est
                fill, letter = theme.PANEL, "?"
            else:
                predicted = state.predicted_label or "unknown"
                fill, letter = theme.CLASS_COLORS[predicted], theme.CLASS_LETTERS[predicted]
            radius = 9 if present else 6
            pygame.draw.circle(surface, fill, pos, radius)
            pygame.draw.circle(surface, theme.BLOCKED if i in episode.collided else theme.TEXT, pos, radius, 3 if i in episode.collided else 1)
            if present:
                text_color = theme.TEXT if fill == theme.PANEL else theme.PANEL
                theme.draw_text(surface, letter, (pos[0] - 4, pos[1] - 9), 12, text_color, bold=True)

        x, y = self.to_screen(*car.position_xy())
        pygame.draw.circle(surface, theme.PANEL, (x, y), 9)
        pygame.draw.circle(surface, theme.CAR, (x, y), 7)
        # Cercle de portée du capteur.
        pygame.draw.circle(surface, theme.ROUTE, (x, y), max(int(episode.config.sensor_range_m * self.scale), 10), 1)

        theme.draw_text(surface, theme.OSM_ATTRIBUTION, (self.rect.x + 10, self.rect.bottom - 22), 12, theme.QUIET)
