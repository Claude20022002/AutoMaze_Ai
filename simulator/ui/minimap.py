"""Minicarte ronde, centrée sur la voiture et orientée dans son sens de marche (comme dans les jeux)."""

from __future__ import annotations

import math

import networkx as nx
import pygame

from simulator.ui import theme
from simulator.world.car import point_on_edge


class MiniMap:
    def __init__(self, graph: nx.MultiDiGraph, center: tuple[int, int], radius: int = 115,
                 meters_across: float = 650.0) -> None:
        self.graph, self.center, self.radius = graph, center, radius
        self.scale = 2 * radius / meters_across               # pixels par mètre
        xs = [d["x"] for _, d in graph.nodes(data=True)]
        ys = [d["y"] for _, d in graph.nodes(data=True)]
        self.min_x, self.max_y = min(xs), max(ys)
        self.margin = int(radius * 1.5)                       # place pour tourner sans bord vide
        width = int((max(xs) - self.min_x) * self.scale) + 2 * self.margin
        height = int((self.max_y - min(ys)) * self.scale) + 2 * self.margin
        self.base = pygame.Surface((width, height))
        self.base.fill((228, 226, 220))
        for edge in graph.edges(keys=True):
            pygame.draw.lines(self.base, (255, 255, 255), False, self._edge_points(edge), 3)

    def _to_base(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.min_x) * self.scale + self.margin, (self.max_y - y) * self.scale + self.margin

    def _edge_points(self, edge):
        data = self.graph.edges[edge]
        if data.get("geometry") is not None:
            coords = list(data["geometry"].coords)
        else:
            coords = [(self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in edge[:2]]
        return [self._to_base(x, y) for x, y in coords]

    def draw(self, surface: pygame.Surface, episode, position, heading) -> None:
        size = int(self.radius * 2 * 1.5)
        bx, by = self._to_base(*position)
        view = pygame.Surface((size, size))
        view.blit(self.base, (size / 2 - bx, size / 2 - by))
        shift = (size / 2 - bx, size / 2 - by)

        def pts(edge):
            return [(x + shift[0], y + shift[1]) for x, y in self._edge_points(edge)]

        car = episode.car
        for edge in car.driven_edges:
            pygame.draw.lines(view, theme.DRIVEN, False, pts(edge), 4)
        for edge in car.route.edges[car.edge_index:]:
            pygame.draw.lines(view, theme.ROUTE, False, pts(edge), 5)
        for edge in episode.costs.blocked:
            pygame.draw.lines(view, theme.BLOCKED, False, pts(edge), 5)
        for state in episode.obstacles:
            if not state.present(episode.step_count):
                continue
            x, y = self._to_base(*point_on_edge(self.graph, state.obstacle.edge, state.obstacle.offset_m))
            color = theme.CLASS_COLORS[state.predicted_label] if state.handled else theme.PANEL
            pygame.draw.circle(view, color, (x + shift[0], y + shift[1]), 6)
            pygame.draw.circle(view, theme.TEXT, (x + shift[0], y + shift[1]), 6, 1)
        gx, gy = self._to_base(self.graph.nodes[episode.goal]["x"], self.graph.nodes[episode.goal]["y"])
        pygame.draw.circle(view, theme.TEXT, (gx + shift[0], gy + shift[1]), 8)
        theme.draw_text(view, "B", (gx + shift[0] - 4, gy + shift[1] - 9), 12, theme.PANEL, bold=True)

        # Rotation : le sens de marche pointe vers le haut.
        angle = math.degrees(math.atan2(heading[0], heading[1]))
        rotated = pygame.transform.rotate(view, angle)
        crop = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        crop.center = rotated.get_rect().center
        disc = pygame.Surface(crop.size, pygame.SRCALPHA)
        disc.blit(rotated, (0, 0), crop)
        mask = pygame.Surface(crop.size, pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (self.radius, self.radius), self.radius)
        disc.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

        cx, cy = self.center
        pygame.draw.circle(surface, (20, 24, 30), (cx, cy), self.radius + 4)
        surface.blit(disc, (cx - self.radius, cy - self.radius))
        # Voiture : flèche toujours vers le haut.
        pygame.draw.polygon(surface, theme.PANEL, [(cx, cy - 12), (cx - 9, cy + 9), (cx, cy + 4), (cx + 9, cy + 9)])
        pygame.draw.polygon(surface, theme.ROUTE, [(cx, cy - 9), (cx - 6, cy + 6), (cx, cy + 2), (cx + 6, cy + 6)])
        # Repère du nord, sur le bord du cercle.
        a = math.radians(angle)
        nx_, ny_ = cx - math.sin(a) * (self.radius - 2), cy - math.cos(a) * (self.radius - 2)
        pygame.draw.circle(surface, (20, 24, 30), (nx_, ny_), 11)
        theme.draw_text(surface, "N", (nx_ - 5, ny_ - 10), 13, theme.PANEL, bold=True)
