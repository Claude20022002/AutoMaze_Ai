"""Comparaison des classifieurs dans le simulateur (équipe B), protocole de la section 6 du plan.

    python -m simulator.compare --episodes 50                              # les trois classifieurs
    python -m simulator.compare --episodes 50 --classifiers simulation ours

Chaque classifieur affronte exactement les mêmes épisodes (même carte, mêmes obstacles, mêmes images).
Un classifieur indisponible (poids absents, accord non obtenu…) est signalé dans le tableau, jamais remplacé.

Fichiers écrits dans ``reports/simulator/`` :
- ``comparaison_seed<graine>_<n>ep.md`` : le tableau, à reprendre dans le rapport ;
- ``comparaison_seed<graine>_<n>ep.csv`` : les mêmes mesures, une ligne par classifieur ;
- ``encounters_<classifieur>_seed<graine>_<n>ep.csv`` : une ligne par obstacle rencontré (lu par metrics.py).
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from simulator.config import REPORTS_DIR, SimConfig
from simulator.metrics import ClassifierReport, build_report
from simulator.perception import CLASSIFIER_MODES

# Libellés des statuts d'épisode : ceux de engine.py (ARRIVED, BLOCKED, TIMEOUT).
ARRIVED, BLOCKED, TIMEOUT = "arrivé", "bloqué", "temps dépassé"

SIMULATION_NOTE = ("Le mode SIMULATION n'est pas un modèle : ses chiffres vérifient la logique du simulateur "
                   "(taux d'erreur réglé à la main), ils ne mesurent aucun classifieur.")


@dataclass(frozen=True)
class Unavailable:
    """Classifieur demandé mais impossible à charger : il apparaît dans le tableau avec la raison."""

    mode: str
    reason: str


# ---------------------------------------------------------------------- exécution
def run_comparison(modes: Sequence[str], episodes: int, config: SimConfig, graph=None, images=None,
                   csv_dir: Path | None = None) -> tuple[list[ClassifierReport], list[Unavailable]]:
    """Lance ``episodes`` épisodes avec chaque classifieur. ``csv_dir`` : où écrire les journaux par obstacle."""
    from simulator.engine import Episode, save_encounters_csv
    from simulator.perception import make_classifier, make_image_provider
    from simulator.world.road_map import load_map

    graph = load_map() if graph is None else graph
    images = make_image_provider(config) if images is None else images  # mêmes images pour tous
    reports: list[ClassifierReport] = []
    unavailable: list[Unavailable] = []
    for mode in modes:
        try:
            classifier = make_classifier(mode, config)
        except (NotImplementedError, FileNotFoundError, ValueError) as error:
            unavailable.append(Unavailable(mode, str(error)))
            continue
        results = [Episode(graph, classifier, config, e, image_for=images).run() for e in range(episodes)]
        if csv_dir is not None:
            save_encounters_csv(results, csv_dir / f"encounters_{mode}_seed{config.seed}_{episodes}ep.csv")
        reports.append(build_report(classifier.name, results))
    return reports, unavailable


# ---------------------------------------------------------------------- mise en forme
def _pct(part: int, total: int) -> str:
    return f"{part} / {total} ({100 * part / total:.0f} %)" if total else "—"


def _num(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


# (libellé, fonction qui donne la valeur affichée). L'ordre suit le protocole de la section 6.
MEASURES: list[tuple[str, Callable[[ClassifierReport], str]]] = [
    ("Épisodes", lambda r: str(r.episodes.episodes)),
    ("Arrivés", lambda r: _pct(r.episodes.count(ARRIVED), r.episodes.episodes)),
    ("Bloqués", lambda r: str(r.episodes.count(BLOCKED))),
    ("Temps dépassé", lambda r: str(r.episodes.count(TIMEOUT))),
    ("Collisions", lambda r: str(r.episodes.collisions)),
    ("Recalculs d'itinéraire", lambda r: str(r.episodes.replans)),
    ("Distance moyenne parcourue", lambda r: _num(r.episodes.mean_distance_m, 0, " m")),
    ("Détour moyen (parcouru / prévu)", lambda r: _num(r.episodes.mean_detour, 2, " ×")),
    ("Pas d'attente", lambda r: str(r.episodes.waiting_steps)),
    ("Obstacles rencontrés", lambda r: str(r.encounters.encounters)),
    ("Bien classés", lambda r: _pct(r.encounters.correct, r.encounters.encounters)),
    ("Classés « Inconnu »", lambda r: str(r.encounters.unknown)),
    ("Fautes graves (personne non reconnue)", lambda r: str(r.encounters.severe_errors)),
    ("Rappel « personne »", lambda r: _pct(r.encounters.persons - r.encounters.severe_errors, r.encounters.persons)),
    ("Décisions inadaptées", lambda r: _pct(r.encounters.inadequate, r.encounters.encounters)),
    ("Temps moyen par décision (CPU)", lambda r: _num(r.encounters.mean_infer_ms, 2, " ms")),
    ("Temps médian par décision (CPU)", lambda r: _num(r.encounters.median_infer_ms, 2, " ms")),
]


def render_markdown(reports: Sequence[ClassifierReport], unavailable: Sequence[Unavailable],
                    config: SimConfig, episodes: int, generated_at: datetime | None = None) -> str:
    generated_at = generated_at or datetime.now()
    lines = [
        "# Comparaison des classifieurs dans le simulateur",
        "",
        f"{episodes} épisodes par classifieur, graine {config.seed}, itinéraire « {config.start} » → "
        f"« {config.goal} », {config.n_obstacles} obstacles par épisode. "
        f"Généré le {generated_at:%d/%m/%Y à %H:%M} par `python -m simulator.compare`.",
        "",
    ]
    if reports:
        lines += ["| Mesure | " + " | ".join(r.classifier for r in reports) + " |",
                  "| --- |" + " --- |" * len(reports)]
        lines += [f"| {label} | " + " | ".join(value(r) for r in reports) + " |" for label, value in MEASURES]
        lines.append("")
    else:
        lines += ["Aucun classifieur n'a pu être évalué.", ""]
    if unavailable:
        lines.append("**Non évalués :**")
        lines += [f"- `{u.mode}` : {u.reason}" for u in unavailable]
        lines.append("")
    if any("SIMULATION" in r.classifier.upper() for r in reports):
        lines += [f"> {SIMULATION_NOTE}", ""]
    lines += [
        "Définitions : *décision inadaptée* = action différente de celle que la vraie classe demandait "
        "(`config.DECISION_RULES`) ; *faute grave* = personne classée autrement que « personne » "
        "(y compris « Inconnu ») ; *rappel « personne »* = personnes reconnues / personnes rencontrées.",
        "",
    ]
    return "\n".join(lines)


def write_summary_csv(reports: Sequence[ClassifierReport], path: Path) -> Path:
    """Les mêmes mesures que le tableau, une ligne par classifieur."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["classifieur", *(label for label, _ in MEASURES)])
        for r in reports:
            writer.writerow([r.classifier, *(value(r) for _, value in MEASURES)])
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--classifiers", nargs="+", default=list(CLASSIFIER_MODES), choices=CLASSIFIER_MODES)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="dossier de sortie")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes doit valoir au moins 1")

    config = SimConfig(seed=args.seed)
    reports, unavailable = run_comparison(args.classifiers, args.episodes, config, csv_dir=args.out)
    stem = f"comparaison_seed{config.seed}_{args.episodes}ep"
    markdown = render_markdown(reports, unavailable, config, args.episodes)
    args.out.mkdir(parents=True, exist_ok=True)
    md_path = args.out / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
    write_summary_csv(reports, args.out / f"{stem}.csv")
    print(markdown)
    print(f"Tableau enregistré : {md_path}")


if __name__ == "__main__":
    main()
