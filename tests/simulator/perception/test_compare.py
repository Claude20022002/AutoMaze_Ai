"""Tests de compare.py.

La mise en forme se teste seule. Les tests qui lancent de vrais épisodes ont besoin de ``simulator/engine.py``
(équipe A) : ils sont ignorés tant que ce fichier n'est pas fusionné dans ``main``.
"""

import csv
from dataclasses import replace
from datetime import datetime

import pytest

from simulator.config import SimConfig
from simulator.contracts import EncounterLog
from simulator.compare import (
    ARRIVED,
    BLOCKED,
    MEASURES,
    SIMULATION_NOTE,
    TIMEOUT,
    Unavailable,
    render_markdown,
    write_summary_csv,
)
from simulator.metrics import ClassifierReport, EncounterSummary, EpisodeSummary

REPORT = ClassifierReport(
    "SIMULATION — modèle réel non chargé",
    EncounterSummary(encounters=10, correct=9, unknown=0, severe_errors=1, persons=4, inadequate=1,
                     mean_infer_ms=0.02, median_infer_ms=0.01),
    EpisodeSummary(episodes=5, statuses={ARRIVED: 4, BLOCKED: 1}, collisions=1, replans=7, waiting_steps=12,
                   mean_distance_m=1900.0, mean_detour=1.04),
)
NO_MODEL = Unavailable("ours", "le classifieur 'ours' n'est pas encore disponible")


def test_markdown_table():
    md = render_markdown([REPORT], [NO_MODEL], SimConfig(), 5, datetime(2026, 10, 19, 18, 0))
    assert "| Mesure | SIMULATION — modèle réel non chargé |" in md
    assert "| Arrivés | 4 / 5 (80 %) |" in md
    assert "| Fautes graves (personne non reconnue) | 1 |" in md
    assert "| Rappel « personne » | 3 / 4 (75 %) |" in md
    assert "| Temps moyen par décision (CPU) | 0.02 ms |" in md
    assert "`ours` : le classifieur 'ours' n'est pas encore disponible" in md   # signalé, jamais remplacé
    assert SIMULATION_NOTE in md
    assert "19/10/2026 à 18:00" in md
    assert len([line for line in md.splitlines() if line.startswith("| ")]) == len(MEASURES) + 2  # + en-tête, séparateur


def test_markdown_without_any_classifier():
    md = render_markdown([], [NO_MODEL], SimConfig(), 5)
    assert "Aucun classifieur n'a pu être évalué." in md
    assert SIMULATION_NOTE not in md


def test_summary_csv(tmp_path):
    path = write_summary_csv([REPORT], tmp_path / "comparaison.csv")
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0] == ["classifieur", *(label for label, _ in MEASURES)]
    assert rows[1][0] == REPORT.classifier and len(rows) == 2


# ---------------------------------------------------------------------- avec engine.py (équipe A)
@pytest.fixture(scope="module")
def engine():
    return pytest.importorskip("simulator.engine", reason="simulator/engine.py (équipe A) pas encore dans main")


def test_status_labels_match_engine(engine):
    assert (ARRIVED, BLOCKED, TIMEOUT) == (engine.ARRIVED, engine.BLOCKED, engine.TIMEOUT)


def test_run_comparison_end_to_end(engine, tmp_path):
    from simulator.compare import main
    from simulator.metrics import read_encounters_csv

    main(["--episodes", "3", "--out", str(tmp_path)])
    md = (tmp_path / "comparaison_seed7_3ep.md").read_text(encoding="utf-8")
    assert "| Épisodes | 3 |" in md
    assert "`reference`" in md and "`ours`" in md          # pas encore disponibles : signalés
    logs = read_encounters_csv(tmp_path / "encounters_simulation_seed7_3ep.csv")
    assert logs and all(isinstance(x, EncounterLog) for x in logs)


def test_same_results_when_run_twice(engine, tmp_path):
    from simulator.compare import run_comparison

    def without_timing(reports):
        # Les temps d'inférence sont réellement mesurés : ils varient d'une exécution à l'autre.
        return [replace(r, encounters=replace(r.encounters, mean_infer_ms=None, median_infer_ms=None)) for r in reports]

    first, _ = run_comparison(["simulation"], 3, SimConfig())
    second, _ = run_comparison(["simulation"], 3, SimConfig())
    assert without_timing(first) == without_timing(second)
