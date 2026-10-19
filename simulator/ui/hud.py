"""Panneau d'informations : classifieur, état de l'épisode, journal des décisions, commandes."""

from __future__ import annotations

import pygame

from simulator.contracts import CLASSES_FR, Action
from simulator.engine import ARRIVED, BLOCKED
from simulator.ui import theme
from simulator.world.road_map import street_name

CONTROLS = ("Espace pause · → pas à pas · ↑↓ vitesse", "R recommencer · N épisode suivant · Échap quitter")


def _fr(label: str) -> str:
    return CLASSES_FR.get(label, "Inconnu")


class Hud:
    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect

    def draw(self, surface: pygame.Surface, episode, paused: bool, steps_per_second: float) -> None:
        pygame.draw.rect(surface, theme.PANEL, self.rect, border_radius=8)
        pygame.draw.rect(surface, theme.BORDER, self.rect, 1, border_radius=8)
        x, y, width = self.rect.x + 16, self.rect.y + 12, self.rect.w - 32

        name = episode.classifier.name
        simulated = "SIMULATION" in name.upper()
        banner = pygame.Rect(x - 4, y, width + 8, 30)
        pygame.draw.rect(surface, theme.WARNING if simulated else theme.ROUTE, banner, border_radius=6)
        theme.draw_text(surface, name, (x + 4, y + 4), 15, theme.PANEL, bold=True, max_width=width - 8)
        y += 42

        r, car = episode.result, episode.car
        if r.status == ARRIVED:
            status, color = "Arrivé à destination", theme.GOOD
        elif r.status == BLOCKED:
            status, color = "Bloqué : aucun chemin", theme.BLOCKED
        elif r.status:
            status, color = r.status.capitalize(), theme.BLOCKED
        elif episode.wait > 0:
            status, color = "À l'arrêt", theme.WARNING
        else:
            status, color = ("En pause" if paused else "En route"), theme.TEXT
        y += theme.draw_text(surface, status, (x, y), 18, color, bold=True) + 2

        street = theme.latin(street_name(episode.graph, car.current_edge)) if car.current_edge else "—"
        lines = [
            f"Épisode {episode.episode} · pas {episode.step_count} · {steps_per_second:g} pas/s",
            f"Rue : {street}",
            f"Distance : {car.distance_m:,.0f} m (trajet prévu {episode.planned.length_m:,.0f} m)".replace(",", " "),
            f"Recalculs : {r.replans} · Collisions : {r.collisions} · Attente : {r.waiting_steps} pas",
        ]
        for line in lines:
            y += theme.draw_text(surface, line, (x, y), 14, theme.TEXT, max_width=width) + 1

        y += 12
        y += theme.draw_text(surface, "Décisions (vraie classe → classe prédite)", (x, y), 14, theme.QUIET, bold=True) + 4
        if not r.encounters:
            theme.draw_text(surface, "Aucun obstacle rencontré pour l'instant.", (x, y), 14, theme.QUIET)
        for log in r.encounters[-7:][::-1]:
            ok = log.correct
            pygame.draw.circle(surface, theme.GOOD if ok else theme.BLOCKED, (x + 5, y + 10), 5)
            text = f"{_fr(log.true_label)} → {_fr(log.predicted_label)} ({log.score:.2f}) : {Action[log.action].value}"
            y += theme.draw_text(surface, text, (x + 16, y), 14, theme.TEXT, max_width=width - 16) + 3

        legend_y = self.rect.bottom - 80
        lx = x
        for label in ("person", "bicycle", "car", "motorcycle", "bus", "truck"):
            pygame.draw.circle(surface, theme.CLASS_COLORS[label], (lx + 7, legend_y + 9), 7)
            theme.draw_text(surface, theme.CLASS_LETTERS[label], (lx + 3, legend_y), 11, theme.PANEL, bold=True)
            lx += 18
            w = theme.font(12).size(_fr(label))[0]
            theme.draw_text(surface, _fr(label), (lx, legend_y), 12, theme.QUIET)
            lx += w + 12
        for i, line in enumerate(CONTROLS):
            theme.draw_text(surface, line, (x, self.rect.bottom - 48 + 18 * i), 12, theme.QUIET, max_width=width)
