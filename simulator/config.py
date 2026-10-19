"""Paramètres partagés du simulateur (fichier partagé : PR approuvée par les deux équipes)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from simulator.contracts import CLASSES, UNKNOWN, Action

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CROPS_TEST_DIR = ROOT / "data" / "crops" / "test"   # images d'obstacles (jeu de test uniquement)
REPORTS_DIR = ROOT / "reports" / "simulator"         # journaux et tableaux de comparaison

OUR_METADATA = MODELS_DIR / "metadata.json"               # produit par le notebook 04
REFERENCE_METADATA = MODELS_DIR / "reference_metadata.json"  # produit par le notebook 06

# Table classe prédite -> action. Proposition à valider avec l'équipe modèle.
DECISION_RULES: dict[str, Action] = {
    "person": Action.STOP_WAIT,
    "bicycle": Action.SLOW_DOWN,
    "motorcycle": Action.SLOW_DOWN,
    "car": Action.REROUTE,
    "bus": Action.REROUTE_WIDE,
    "truck": Action.REROUTE_WIDE,
    UNKNOWN: Action.CAUTIOUS_STOP,
}


@dataclass(frozen=True)
class SimConfig:
    """Réglages d'un épisode. Les valeurs par défaut sont modifiables en ligne de commande."""

    seed: int = 7
    grid_rows: int = 21                  # dimensions impaires : convention des labyrinthes en grille
    grid_cols: int = 31
    extra_openings: float = 0.08         # part de murs retirés pour créer des boucles (sinon un seul chemin)
    n_obstacles: int = 6
    sensor_range: int = 2                # nombre de cases vues devant la voiture
    max_steps: int = 2000                # au-delà, l'épisode se termine avec le statut « bloqué »

    # Effets des actions sur la recherche de chemin (équipe A)
    wait_steps: int = 5                  # STOP_WAIT : nombre de pas où la case reste bloquée
    slow_cost: float = 5.0               # SLOW_DOWN : coût d'une case (1.0 = case libre)
    wide_radius: int = 1                 # REROUTE_WIDE : cases voisines bloquées autour de l'obstacle

    # Perception (équipe B)
    confidence_threshold: float = 0.5    # sous ce score, le label devient UNKNOWN ; à régler sur la validation
    simulated_error_rate: float = 0.1    # mode simulation : part des prédictions volontairement fausses

    decision_rules: dict[str, Action] = field(default_factory=lambda: dict(DECISION_RULES))


def check_metadata_classes(metadata_path: Path) -> None:
    """Vérifie que l'ordre des classes d'un modèle est celui du simulateur. Lève ValueError sinon."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = tuple(metadata.get("classes", ()))
    if classes != CLASSES:
        raise ValueError(
            f"{metadata_path.name} : classes {classes} différentes de celles du simulateur {CLASSES}. "
            "L'ordre doit être identique à celui des notebooks."
        )
