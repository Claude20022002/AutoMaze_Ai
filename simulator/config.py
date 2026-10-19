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

# Carte : réseau routier réel (OpenStreetMap) d'une partie de Casablanca, du Maârif au centre-ville.
MAP_FILE = Path(__file__).resolve().parent / "assets" / "casablanca_drive.graphml"
MAP_BBOX = (-7.645, 33.578, -7.612, 33.600)  # (ouest, sud, est, nord) en degrés, format OSMnx 2

# Lieux de départ et d'arrivée possibles (latitude, longitude), coordonnées obtenues par géocodage OSM.
LANDMARKS: dict[str, tuple[float, float]] = {
    "Twin Center (Maârif)": (33.58658, -7.63229),
    "Place Mohammed V (centre-ville)": (33.59164, -7.61984),
    "Boulevard d'Anfa": (33.59078, -7.63667),
}

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
    start: str = "Twin Center (Maârif)"             # clé de LANDMARKS
    goal: str = "Place Mohammed V (centre-ville)"   # clé de LANDMARKS
    n_obstacles: int = 6
    sensor_range_m: float = 35.0         # distance de détection devant la voiture (caméra en ville, 29 km/h)
    car_speed_mps: float = 8.0           # environ 29 km/h, vitesse de circulation en ville
    step_seconds: float = 1.0            # durée simulée d'un pas
    max_steps: int = 2000                # au-delà, l'épisode se termine avec le statut « bloqué »

    # Effets des actions (équipe A)
    wait_steps: int = 5                  # STOP_WAIT : nombre de pas d'attente avant de repartir
    cautious_wait_steps: int = 2         # CAUTIOUS_STOP : attente avant de contourner
    slow_factor: float = 3.0             # SLOW_DOWN : coût du tronçon multiplié, vitesse divisée d'autant
    wide_penalty: float = 10.0           # REROUTE_WIDE : pénalité des tronçons voisins de l'obstacle
    patience_steps: int = 30             # sans aucun chemin pendant ce nombre de pas : épisode « bloqué »

    # Comportement réel des obstacles, selon leur VRAIE classe : nombre de pas avant qu'ils libèrent
    # la voie (None = immobile). Une mauvaise classification peut donc mener à une collision.
    clear_steps: dict[str, int | None] = field(default_factory=lambda: {
        "person": 4, "bicycle": 3, "motorcycle": 2, "car": None, "bus": None, "truck": None,
    })

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
