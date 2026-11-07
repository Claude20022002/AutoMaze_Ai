"""Tests de metrics.py : agrégation des EncounterLog et des résultats d'épisode."""

import csv
from dataclasses import asdict, dataclass, field, fields

import pytest

from simulator.contracts import CLASSES, UNKNOWN, EncounterLog
from simulator.metrics import (
    build_report,
    confusion_matrix,
    expected_action,
    is_inadequate,
    read_encounters_csv,
    summarize_encounters,
    summarize_episodes,
)


def log(true_label: str, predicted: str, infer_ms: float = 1.0, action: str | None = None) -> EncounterLog:
    from simulator.config import DECISION_RULES

    action = action or DECISION_RULES[predicted].name
    return EncounterLog(0, 1, "test", true_label, predicted, 0.9, action, infer_ms)


@dataclass
class FakeResult:
    """Mêmes champs que ``simulator.engine.EpisodeResult``."""

    status: str
    distance_m: float
    planned_length_m: float
    replans: int = 0
    collisions: int = 0
    waiting_steps: int = 0
    encounters: list = field(default_factory=list)


LOGS = [
    log("person", "person", 2.0),
    log("person", "bicycle", 4.0),   # faute grave, décision inadaptée
    log("person", UNKNOWN, 6.0),     # faute grave (définition du contrat), décision inadaptée
    log("car", "car", 1.0),
    log("bus", "truck", 3.0),        # erreur, mais même action (REROUTE_WIDE) : décision adaptée
]


def test_summarize_encounters():
    s = summarize_encounters(LOGS)
    assert (s.encounters, s.correct, s.unknown, s.severe_errors, s.persons, s.inadequate) == (5, 2, 1, 2, 3, 2)
    assert s.accuracy == pytest.approx(0.4)
    assert s.person_recall == pytest.approx(1 / 3)
    assert s.mean_infer_ms == pytest.approx(3.2)
    assert s.median_infer_ms == pytest.approx(3.0)


def test_empty_summary_has_no_ratio():
    s = summarize_encounters([])
    assert s.encounters == 0 and s.accuracy is None and s.person_recall is None and s.mean_infer_ms is None


def test_inadequate_uses_true_class_rule():
    assert expected_action("person").name == "STOP_WAIT"
    assert not is_inadequate(log("bus", "truck"))
    assert is_inadequate(log("person", "car"))
    assert is_inadequate(log("car", "car", action="SLOW_DOWN"))  # action prise différente de la règle


def test_confusion_matrix():
    m = confusion_matrix(LOGS)
    assert set(m) == set(CLASSES)
    assert set(m["person"]) == set(CLASSES) | {UNKNOWN}
    assert (m["person"]["person"], m["person"]["bicycle"], m["person"][UNKNOWN], m["bus"]["truck"]) == (1, 1, 1, 1)
    assert sum(sum(row.values()) for row in m.values()) == len(LOGS)


def test_summarize_episodes():
    results = [
        FakeResult("arrivé", 2000, 1000, replans=2, collisions=0, waiting_steps=5),
        FakeResult("arrivé", 1500, 1500, replans=1, collisions=1),
        FakeResult("bloqué", 500, 1000, waiting_steps=30),
    ]
    s = summarize_episodes(results)
    assert s.episodes == 3
    assert s.statuses == {"arrivé": 2, "bloqué": 1}
    assert s.count("arrivé") == 2 and s.count("temps dépassé") == 0
    assert (s.collisions, s.replans, s.waiting_steps) == (1, 3, 35)
    assert s.mean_distance_m == pytest.approx(4000 / 3)
    assert s.mean_detour == pytest.approx((2.0 + 1.0 + 0.5) / 3)


def test_build_report_collects_encounters_of_all_episodes():
    results = [FakeResult("arrivé", 10, 10, encounters=LOGS[:2]), FakeResult("arrivé", 10, 10, encounters=LOGS[2:])]
    report = build_report("test", results)
    assert report.classifier == "test"
    assert report.encounters == summarize_encounters(LOGS)
    assert report.episodes.episodes == 2


def test_read_encounters_csv_round_trip(tmp_path):
    path = tmp_path / "encounters.csv"
    with path.open("w", newline="", encoding="utf-8") as f:   # même format que engine.save_encounters_csv
        writer = csv.DictWriter(f, fieldnames=[x.name for x in fields(EncounterLog)])
        writer.writeheader()
        writer.writerows(asdict(x) for x in LOGS)
    assert read_encounters_csv(path) == LOGS


def test_read_encounters_csv_rejects_other_columns(tmp_path):
    path = tmp_path / "autre.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_encounters_csv(path)
