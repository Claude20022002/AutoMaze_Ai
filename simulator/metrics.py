"""Agrégation des résultats du simulateur (équipe B) : une ligne de mesures par classifieur.

Deux niveaux, comme au protocole de la section 6 du plan :
- par obstacle (``EncounterLog``) : bien classés, « Inconnu », fautes graves, décisions inadaptées, temps ;
- par épisode (``EpisodeResult`` de ``engine.py``) : arrivés, collisions, recalculs, distance, attente.

Toutes les valeurs sont calculées à partir des journaux réellement produits : rien n'est estimé.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from simulator.config import DECISION_RULES
from simulator.contracts import CLASSES, UNKNOWN, Action, EncounterLog

if TYPE_CHECKING:
    from simulator.engine import EpisodeResult

PREDICTED_LABELS: tuple[str, ...] = (*CLASSES, UNKNOWN)


# ---------------------------------------------------------------------- journaux CSV
def read_encounters_csv(path: Path) -> list[EncounterLog]:
    """Relit le CSV écrit par ``engine.save_encounters_csv`` (une ligne par obstacle rencontré)."""
    types = {f.name: f.type for f in fields(EncounterLog)}
    expected = list(types)
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected:
            raise ValueError(f"{Path(path).name} : colonnes {reader.fieldnames}, attendu {expected}")
        return [EncounterLog(**{name: _cast(types[name], row[name]) for name in expected}) for row in reader]


def _cast(type_name: str, value: str):
    # Les annotations sont des chaînes (``from __future__ import annotations`` dans contracts.py).
    return {"int": int, "float": float}.get(type_name, str)(value)


# ---------------------------------------------------------------------- par obstacle
def expected_action(true_label: str, rules: Mapping[str, Action] = DECISION_RULES) -> Action:
    """Action qu'aurait déclenchée un classifieur parfait : celle de la vraie classe."""
    return rules[true_label]


def is_inadequate(log: EncounterLog, rules: Mapping[str, Action] = DECISION_RULES) -> bool:
    """Décision inadaptée : l'action prise n'est pas celle que la vraie classe demandait."""
    return log.action != expected_action(log.true_label, rules).name


@dataclass(frozen=True)
class EncounterSummary:
    encounters: int
    correct: int
    unknown: int
    severe_errors: int        # une personne classée autrement que « personne » (EncounterLog.severe_error)
    persons: int              # nombre de personnes rencontrées (dénominateur du rappel « personne »)
    inadequate: int
    mean_infer_ms: float | None
    median_infer_ms: float | None

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.encounters if self.encounters else None

    @property
    def person_recall(self) -> float | None:
        return (self.persons - self.severe_errors) / self.persons if self.persons else None


def summarize_encounters(logs: Iterable[EncounterLog], rules: Mapping[str, Action] = DECISION_RULES) -> EncounterSummary:
    logs = list(logs)
    times = [log.infer_ms for log in logs]
    return EncounterSummary(
        encounters=len(logs),
        correct=sum(log.correct for log in logs),
        unknown=sum(log.predicted_label == UNKNOWN for log in logs),
        severe_errors=sum(log.severe_error for log in logs),
        persons=sum(log.true_label == "person" for log in logs),
        inadequate=sum(is_inadequate(log, rules) for log in logs),
        mean_infer_ms=statistics.fmean(times) if times else None,
        median_infer_ms=statistics.median(times) if times else None,
    )


def confusion_matrix(logs: Iterable[EncounterLog]) -> dict[str, dict[str, int]]:
    """Matrice de confusion ``[vraie classe][classe prédite]``, colonne « unknown » comprise."""
    counts = Counter((log.true_label, log.predicted_label) for log in logs)
    return {t: {p: counts[(t, p)] for p in PREDICTED_LABELS} for t in CLASSES}


# ---------------------------------------------------------------------- par épisode
@dataclass(frozen=True)
class EpisodeSummary:
    episodes: int
    statuses: dict[str, int]          # statut (« arrivé », « bloqué », « temps dépassé ») -> nombre d'épisodes
    collisions: int
    replans: int
    waiting_steps: int
    mean_distance_m: float | None
    mean_detour: float | None         # distance parcourue / longueur de l'itinéraire initial (1.0 = aucun détour)

    def count(self, status: str) -> int:
        return self.statuses.get(status, 0)


def summarize_episodes(results: Sequence[EpisodeResult]) -> EpisodeSummary:
    distances = [r.distance_m for r in results]
    detours = [r.distance_m / r.planned_length_m for r in results if r.planned_length_m > 0]
    return EpisodeSummary(
        episodes=len(results),
        statuses=dict(Counter(str(r.status) for r in results)),
        collisions=sum(r.collisions for r in results),
        replans=sum(r.replans for r in results),
        waiting_steps=sum(r.waiting_steps for r in results),
        mean_distance_m=statistics.fmean(distances) if distances else None,
        mean_detour=statistics.fmean(detours) if detours else None,
    )


# ---------------------------------------------------------------------- rapport
@dataclass(frozen=True)
class ClassifierReport:
    """Mesures d'un classifieur sur une série d'épisodes."""

    classifier: str
    encounters: EncounterSummary
    episodes: EpisodeSummary


def build_report(classifier: str, results: Sequence[EpisodeResult],
                 rules: Mapping[str, Action] = DECISION_RULES) -> ClassifierReport:
    logs = [log for r in results for log in r.encounters]
    return ClassifierReport(classifier, summarize_encounters(logs, rules), summarize_episodes(results))
