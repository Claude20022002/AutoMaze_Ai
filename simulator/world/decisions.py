"""Traduction d'une ``Action`` en changements de coûts sur la carte."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from simulator.config import SimConfig
from simulator.contracts import Action
from simulator.world.pathfinding import CostModel, EdgeKey


@dataclass(frozen=True)
class Reaction:
    """Ce que la voiture doit faire après une décision."""

    wait_steps: int = 0        # pas d'arrêt avant de repartir
    replan: bool = False       # recalculer l'itinéraire avec les nouveaux coûts
    slow_edge: bool = False    # rouler au ralenti sur le tronçon de l'obstacle


def apply_action(action: Action, edge: EdgeKey, graph: nx.MultiDiGraph, costs: CostModel, config: SimConfig) -> Reaction:
    """Modifie ``costs`` selon l'action et renvoie la réaction attendue de la voiture."""
    if action is Action.STOP_WAIT:
        # On attend que la personne traverse ; l'itinéraire ne change pas.
        return Reaction(wait_steps=config.wait_steps)

    if action is Action.SLOW_DOWN:
        # Le tronçon devient plus coûteux : Dijkstra peut préférer un autre chemin, sinon on passe au ralenti.
        costs.penalize(edge, config.slow_factor)
        return Reaction(replan=True, slow_edge=True)

    if action is Action.REROUTE:
        costs.block(edge)
        return Reaction(replan=True)

    if action is Action.REROUTE_WIDE:
        # Gros véhicule : on bloque son tronçon et on évite aussi les rues qui partent de ses extrémités.
        costs.block(edge)
        for node in edge[:2]:
            for neighbour_edge in list(graph.in_edges(node, keys=True)) + list(graph.out_edges(node, keys=True)):
                if not costs.is_blocked(neighbour_edge):
                    costs.penalize(neighbour_edge, config.wide_penalty, both_directions=False)
        return Reaction(replan=True)

    if action is Action.CAUTIOUS_STOP:
        costs.block(edge)
        return Reaction(wait_steps=config.cautious_wait_steps, replan=True)

    raise ValueError(f"action non gérée : {action}")
