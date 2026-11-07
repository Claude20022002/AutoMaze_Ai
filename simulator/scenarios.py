"""Scénarios de conduite : les 4 scénarios du dossier Go/NoGo et des trajets quotidiens dans Casablanca.

Un scénario fixe le départ, l'arrivée, les arrêts prévus et les obstacles d'un épisode, puis dit ce qui doit se passer.
``check`` compare ensuite l'épisode joué à ces attentes : on vérifie ainsi que chaque scénario se réalise vraiment.

- **Go/NoGo** (chapitre 2, section « Scénarios de fonctionnement ») : route libre, présence d'un piéton,
  route bloquée, contournement. Les obstacles y sont placés à un endroit choisi, le même à chaque épisode.
- **Trajets quotidiens** : de vrais lieux de Casablanca, avec des arrêts (déposer les enfants, faire les courses)
  et des obstacles tirés au hasard selon le moment de la journée. Chaque épisode donne une autre journée.

Les attentes supposent que chaque obstacle est bien classé. Avec un vrai modèle, une erreur de classification peut
faire échouer un scénario : c'est voulu, et ``check`` le montre (par exemple une personne prise pour un vélo).

Le module ne dépend d'aucun affichage. Sans fenêtre :
    python -m simulator.scenarios                              # tous les scénarios, classifieur « simulation »
    python -m simulator.scenarios --scenario pieton --episodes 5
    python -m simulator.scenarios --classifier ours --famille gonogo
Dans les vues : ``python -m simulator.main --scenario pieton``, ``python -m simulator.render3d.app --scenario ecole``.
"""

from __future__ import annotations

import argparse
import random
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from simulator.config import DECISION_RULES, SimConfig
from simulator.contracts import CLASSES, CLASSES_FR, Action, Classifier, Obstacle
from simulator.engine import ARRIVED, Episode, ImageProvider, Stop, gray_placeholder, join_routes
from simulator.world.decisions import apply_action
from simulator.world.obstacles import place_obstacles
from simulator.world.pathfinding import CostModel, NoPathError, Route, dijkstra

GONOGO, QUOTIDIEN = "gonogo", "quotidien"


@dataclass(frozen=True)
class ObstacleSpec:
    """Obstacle placé à un endroit choisi de l'itinéraire prévu.

    ``kind`` :
    - ``"milieu"`` : au milieu d'une rue d'au moins 40 m ;
    - ``"rue_suivante"`` : au début de la rue suivante ; la voiture le voit avant le carrefour et prend une autre rue ;
    - ``"rue_actuelle"`` : loin dans une longue rue ; la voiture le voit alors qu'elle y roule déjà et fait demi-tour.
    ``after`` : l'emplacement est cherché à partir de cette fraction des tronçons de l'itinéraire.
    """

    label: str
    kind: str = "milieu"
    after: float = 0.35


@dataclass(frozen=True)
class StopSpec:
    """Arrêt prévu sur un trajet quotidien. ``place`` : clé de ``config.LANDMARKS``."""

    place: str
    dwell_s: float
    why: str


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    family: str                                 # GONOGO ou QUOTIDIEN
    description: str
    start: str                                  # clés de config.LANDMARKS
    goal: str
    expect: tuple[str, ...]                     # clés de CHECKS, vérifiées après l'épisode
    stops: tuple[StopSpec, ...] = ()
    obstacles: tuple[ObstacleSpec, ...] = ()    # obstacles placés (scénarios Go/NoGo)
    random_obstacles: int = 0                   # obstacles tirés au hasard (trajets quotidiens)
    random_labels: tuple[str, ...] = CLASSES    # classes tirées ; une classe répétée sort plus souvent
    traffic: int = 80                           # ambiance de la vue 3D (décor, sans effet sur l'épisode)
    pedestrians: int = 40

    @property
    def route_text(self) -> str:
        return " → ".join([self.start, *(s.place for s in self.stops), self.goal])


# Attentes communes : arriver, sans collision.
_BASE = ("arrive", "sans_collision")

SCENARIOS: dict[str, Scenario] = {s.key: s for s in (
    # ------------------------------------------------------------------ dossier Go/NoGo
    Scenario(
        "route_libre", "Scénario 1 · Route libre", GONOGO,
        "Aucun obstacle : la voiture suit le chemin de Dijkstra jusqu'à la destination, sans s'arrêter.",
        "Twin Center (Maârif)", "Place Mohammed V (centre-ville)",
        expect=(*_BASE, "aucun_obstacle", "trajet_inchange", "sans_attente"),
    ),
    Scenario(
        "pieton", "Scénario 2 · Présence d'un piéton", GONOGO,
        "Une personne traverse devant la voiture : arrêt, attente de la traversée, puis même trajet.",
        "Twin Center (Maârif)", "Place Mohammed V (centre-ville)",
        expect=(*_BASE, "arret_pieton", "trajet_inchange"),
        obstacles=(ObstacleSpec("person", "milieu", 0.35),),
        pedestrians=60,
    ),
    Scenario(
        "route_bloquee", "Scénario 3 · Route bloquée", GONOGO,
        "Un camion de livraison bloque la rue suivante : elle est marquée bloquée et Dijkstra calcule "
        "un autre chemin. La voiture tourne au carrefour, sans demi-tour.",
        "Twin Center (Maârif)", "Place Mohammed V (centre-ville)",
        expect=(*_BASE, "recalcul", "obstacle_evite", "sans_demi_tour"),
        obstacles=(ObstacleSpec("truck", "rue_suivante", 0.35),),
    ),
    Scenario(
        "contournement", "Scénario 4 · Contournement", GONOGO,
        "Une voiture en panne au milieu de la rue où l'on roule : demi-tour, nouvelle rue, puis la voiture "
        "reprend sa progression vers la destination.",
        "Twin Center (Maârif)", "Place Mohammed V (centre-ville)",
        expect=(*_BASE, "recalcul", "obstacle_evite", "demi_tour"),
        obstacles=(ObstacleSpec("car", "rue_actuelle", 0.35),),
    ),
    # ------------------------------------------------------------------ trajets quotidiens
    Scenario(
        "heure_de_pointe", "Aller au travail à l'heure de pointe", QUOTIDIEN,
        "Du quartier Gauthier au centre-ville vers 8 h : beaucoup de voitures, de bus et de camions de livraison.",
        "Quartier Gauthier", "Place Mohammed V (centre-ville)",
        expect=_BASE,
        random_obstacles=8,
        random_labels=("car",) * 4 + ("bus",) * 2 + ("truck",) * 2 + ("motorcycle",) * 2 + ("bicycle", "person"),
        traffic=120, pedestrians=50,
    ),
    Scenario(
        "ecole", "Déposer les enfants au lycée", QUOTIDIEN,
        "Gauthier → lycée Lyautey (dépose) → bureau au Twin Center. Autour de l'école : piétons et vélos.",
        "Quartier Gauthier", "Twin Center (Maârif)",
        expect=(*_BASE, "arrets"),
        stops=(StopSpec("Lycée Lyautey", 40, "déposer les enfants"),),
        random_obstacles=6,
        random_labels=("person",) * 4 + ("bicycle",) * 2 + ("car",) * 2 + ("motorcycle", "bus"),
        traffic=70, pedestrians=90,
    ),
    Scenario(
        "gare", "Attraper le train à Casa-Port", QUOTIDIEN,
        "Du Twin Center à la gare Casa-Port, en traversant le centre-ville.",
        "Twin Center (Maârif)", "Gare Casa-Port",
        expect=_BASE,
        random_obstacles=7,
    ),
    Scenario(
        "hopital", "Rendez-vous à l'hôpital Ibn Rochd", QUOTIDIEN,
        "Du boulevard d'Anfa à l'hôpital Ibn Rochd, en milieu de matinée.",
        "Boulevard d'Anfa", "Hôpital Ibn Rochd",
        expect=_BASE,
        random_obstacles=6,
        random_labels=("car",) * 3 + ("person",) * 2 + ("bus", "truck", "motorcycle", "bicycle"),
    ),
    Scenario(
        "courses", "Courses du samedi", QUOTIDIEN,
        "Gauthier → marché du Maârif → parc de la Ligue Arabe → retour à Gauthier. Beaucoup de piétons et de motos.",
        "Quartier Gauthier", "Quartier Gauthier",
        expect=(*_BASE, "arrets"),
        stops=(StopSpec("Marché du Maârif", 60, "faire les courses"),
               StopSpec("Parc de la Ligue Arabe", 30, "promenade au parc")),
        random_obstacles=8,
        random_labels=("person",) * 3 + ("motorcycle",) * 2 + ("car",) * 2 + ("bicycle", "truck"),
        traffic=60, pedestrians=80,
    ),
    Scenario(
        "journee", "Une journée complète", QUOTIDIEN,
        "Gauthier → lycée Lyautey → bureau au Twin Center → marché du Maârif → retour à Gauthier.",
        "Quartier Gauthier", "Quartier Gauthier",
        expect=(*_BASE, "arrets"),
        stops=(StopSpec("Lycée Lyautey", 30, "déposer les enfants"),
               StopSpec("Twin Center (Maârif)", 60, "journée de travail (abrégée)"),
               StopSpec("Marché du Maârif", 45, "faire les courses")),
        random_obstacles=12,
        traffic=90, pedestrians=60,
    ),
)}


# ---------------------------------------------------------------------- construction d'un épisode
def build_episode(
    scenario: Scenario | str,
    graph: nx.MultiDiGraph,
    classifier: Classifier,
    config: SimConfig,
    episode: int = 0,
    image_for: ImageProvider = gray_placeholder,
) -> Episode:
    """Épisode du scénario. Mêmes obstacles pour tous les classifieurs ; les trajets quotidiens changent d'un
    épisode à l'autre (graine du simulateur, du scénario et de l'épisode)."""
    from simulator.world.road_map import landmark_node

    scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    start, goal = landmark_node(graph, scenario.start), landmark_node(graph, scenario.goal)
    stops = [Stop(landmark_node(graph, s.place), round(s.dwell_s / config.step_seconds), f"{s.place} ({s.why})")
             for s in scenario.stops]
    targets = [s.node for s in stops] + [goal]
    legs = [dijkstra(graph, a, b) for a, b in zip([start, *targets[:-1]], targets)]

    obstacles = [place_obstacle(graph, join_routes(legs), spec, goal, config) for spec in scenario.obstacles]
    if scenario.random_obstacles:
        rng = random.Random(config.seed * 100_003 + episode * 7_919 + zlib.crc32(scenario.key.encode()))
        drawn = _random_obstacles(graph, legs, scenario.random_obstacles, rng, scenario.random_labels)
        obstacles += _keep_feasible(graph, config, start, goal, stops, obstacles, drawn)
    return Episode(graph, classifier, config, episode, start=start, goal=goal, image_for=image_for,
                   obstacles=obstacles, stops=stops)


class _PerfectClassifier(Classifier):
    """Sert seulement à vérifier qu'un tirage d'obstacles laisse le trajet possible. Jamais affiché ni mesuré."""

    name = "vérification du tirage"

    def predict(self, image_rgb, true_label=None):
        from simulator.contracts import Prediction

        return Prediction(true_label, 1.0, tuple(float(c == true_label) for c in CLASSES), 0.0)


def _keep_feasible(graph, config: SimConfig, start: int, goal: int, stops: Sequence[Stop],
                   fixed: list[Obstacle], drawn: list[Obstacle]) -> list[Obstacle]:
    """Retire les obstacles tirés qui rendraient le trajet impossible même s'ils étaient tous bien classés.

    Le moteur considère qu'une voiture, un bus ou un camion ne repart jamais. Sur la seule voie d'une rue à sens
    unique (les boulevards à chaussées séparées le sont dans OSM), il fermerait l'accès à la destination, alors
    qu'en vrai on le dépasserait. On rejoue donc l'épisode avec un classifieur parfait et l'on retire, à chaque
    blocage, le dernier véhicule immobile rencontré. Le tirage reste le même pour tous les classifieurs.
    """
    kept = list(drawn)
    for _ in range(len(drawn)):
        trial = Episode(graph, _PerfectClassifier(), config, start=start, goal=goal,
                        obstacles=[*fixed, *kept], stops=stops)
        if trial.run().status == ARRIVED:
            break
        immobile = [s for s in trial.obstacles if s.handled and s.obstacle in kept
                    and config.clear_steps[s.obstacle.true_label] is None]
        if not immobile:
            break
        kept.remove(max(immobile, key=lambda s: s.detected_step).obstacle)
    return kept


def place_obstacle(graph: nx.MultiDiGraph, route: Route, spec: ObstacleSpec, goal: int, config: SimConfig) -> Obstacle:
    """Cherche sur l'itinéraire le premier tronçon où l'obstacle produit le comportement du scénario."""
    lengths = [float(graph.edges[e]["length"]) for e in route.edges]
    first = max(2, int(spec.after * len(route.edges)))
    for i in range(first, len(route.edges) - 1):
        edge, length = route.edges[i], lengths[i]
        if spec.kind == "milieu" and length >= 40:
            return Obstacle(edge, spec.label, None, length / 2)
        if spec.kind == "rue_suivante" and length >= 30 and lengths[i - 1] >= 45:
            # Vu à 35 m au plus, 8 m après le carrefour : la voiture est encore dans la rue précédente.
            if _detour_from(graph, edge, spec.label, edge[0], goal, config, avoid_back=route.edges[i - 1][0]):
                return Obstacle(edge, spec.label, None, 8.0)
        if spec.kind == "rue_actuelle" and length >= 60:
            # Au-delà de 35 m + un pas (8 m), la voiture est déjà dans la rue quand elle le voit.
            offset = min(max(50.0, 0.6 * length), length - 5)
            if _detour_from(graph, edge, spec.label, edge[0], goal, config):
                return Obstacle(edge, spec.label, None, offset)
    raise ValueError(f"aucun tronçon ne convient à l'obstacle {spec} sur cet itinéraire")


def _detour_from(graph, edge, label, start, goal, config: SimConfig, avoid_back: int | None = None) -> bool:
    """Un autre chemin existe-t-il depuis ``start`` si l'obstacle est bien classé ? Sans revenir sur ``avoid_back``."""
    costs = CostModel(graph)
    apply_action(DECISION_RULES[label], edge, graph, costs, config)
    try:
        route = dijkstra(graph, start, goal, costs)
    except NoPathError:
        return False
    return bool(route.edges) and (avoid_back is None or route.edges[0][1] != avoid_back)


def _random_obstacles(graph, legs: Sequence[Route], count: int, rng: random.Random, labels: tuple[str, ...]) -> list[Obstacle]:
    """Répartit ``count`` obstacles entre les étapes, selon leur longueur ; jamais tout près d'un arrêt."""
    total = sum(leg.length_m for leg in legs) or 1.0
    obstacles: list[Obstacle] = []
    for leg in legs:
        share = max(1, round(count * leg.length_m / total)) if count else 0
        obstacles += place_obstacles(graph, leg, share, rng, labels=labels)
    seen, unique = set(), []
    for obstacle in obstacles:                 # deux étapes peuvent emprunter la même rue
        if obstacle.edge not in seen:
            seen.add(obstacle.edge)
            unique.append(obstacle)
    return unique


# ---------------------------------------------------------------------- vérification
@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str = ""


def check(scenario: Scenario | str, episode: Episode) -> list[Check]:
    """Compare un épisode terminé (ou en cours) aux attentes du scénario."""
    scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    return [CHECKS[key](episode, scenario) for key in scenario.expect]


def realized(checks: Sequence[Check]) -> bool:
    return all(c.ok for c in checks)


def _obstacle_edges(episode: Episode) -> set:
    return {state.obstacle.edge for state in episode.obstacles}


def _arrive(ep: Episode, sc: Scenario) -> Check:
    return Check("Arrivé à destination", ep.result.status == ARRIVED, f"statut : {ep.result.status or 'en cours'}")


def _sans_collision(ep: Episode, sc: Scenario) -> Check:
    return Check("Aucune collision", ep.result.collisions == 0, f"{ep.result.collisions} collision(s)")


def _aucun_obstacle(ep: Episode, sc: Scenario) -> Check:
    return Check("Aucun obstacle rencontré", not ep.result.encounters, f"{len(ep.result.encounters)} rencontre(s)")


def _sans_attente(ep: Episode, sc: Scenario) -> Check:
    return Check("Aucun arrêt", ep.result.waiting_steps == 0, f"{ep.result.waiting_steps} pas d'attente")


def _trajet_inchange(ep: Episode, sc: Scenario) -> Check:
    same = ep.result.replans == 0 and tuple(ep.car.driven_edges) == ep.planned.edges
    return Check("Chemin de Dijkstra suivi jusqu'au bout", same, f"{ep.result.replans} recalcul(s)")


def _arret_pieton(ep: Episode, sc: Scenario) -> Check:
    persons = [log for log in ep.result.encounters if log.true_label == "person"]
    stopped = bool(persons) and all(log.action == Action.STOP_WAIT.name for log in persons)
    waited = ep.result.waiting_steps >= ep.config.wait_steps
    detail = ", ".join(f"personne classée {CLASSES_FR.get(log.predicted_label, 'Inconnu')} → "
                       f"{Action[log.action].value}" for log in persons)
    return Check("Arrêt devant la personne, puis départ", stopped and waited, detail or "aucune personne rencontrée")


def _recalcul(ep: Episode, sc: Scenario) -> Check:
    return Check("Rue bloquée et nouveau chemin calculé", ep.result.replans >= 1 and bool(ep.costs.blocked),
                 f"{ep.result.replans} recalcul(s), {len(ep.costs.blocked)} tronçon(s) bloqué(s)")


def _obstacle_evite(ep: Episode, sc: Scenario) -> Check:
    driven = _obstacle_edges(ep) & set(ep.car.driven_edges)
    return Check("La rue de l'obstacle n'est pas traversée", not driven, f"{len(driven)} traversée(s)")


def _demi_tour(ep: Episode, sc: Scenario) -> Check:
    return Check("Demi-tour, puis autre rue", ep.result.u_turns >= 1, f"{ep.result.u_turns} demi-tour(s)")


def _sans_demi_tour(ep: Episode, sc: Scenario) -> Check:
    return Check("Changement de rue au carrefour, sans demi-tour", ep.result.u_turns == 0,
                 f"{ep.result.u_turns} demi-tour(s)")


def _arrets(ep: Episode, sc: Scenario) -> Check:
    reached = [name for name, _ in ep.result.stops_reached]
    expected = [stop.name for stop in ep.stops]
    return Check("Arrêts prévus effectués dans l'ordre", reached == expected, f"{len(reached)} / {len(expected)}")


CHECKS = {
    "arrive": _arrive, "sans_collision": _sans_collision, "aucun_obstacle": _aucun_obstacle,
    "sans_attente": _sans_attente, "trajet_inchange": _trajet_inchange, "arret_pieton": _arret_pieton,
    "recalcul": _recalcul, "obstacle_evite": _obstacle_evite, "demi_tour": _demi_tour,
    "sans_demi_tour": _sans_demi_tour, "arrets": _arrets,
}


# ---------------------------------------------------------------------- ligne de commande
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=["tous", *SCENARIOS], default="tous")
    parser.add_argument("--famille", choices=("toutes", GONOGO, QUOTIDIEN), default="toutes")
    parser.add_argument("--classifier", default="simulation", help="simulation, reference ou ours")
    parser.add_argument("--episodes", type=int, default=1, help="épisodes par scénario")
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # console Windows sans UTF-8

    from simulator.perception import make_classifier, make_image_provider
    from simulator.world.road_map import load_map

    config = SimConfig(seed=args.seed)
    graph = load_map()
    classifier = make_classifier(args.classifier, config)
    chosen = [s for s in SCENARIOS.values()
              if args.scenario in ("tous", s.key) and args.famille in ("toutes", s.family)]
    total = ok = 0
    print(f"Classifieur : {classifier.name}\n")
    for scenario in chosen:
        print(f"{scenario.title}  [{scenario.key}]\n  {scenario.route_text}")
        for e in range(args.episodes):
            episode = build_episode(scenario, graph, classifier, config, e, make_image_provider(config, episode=e))
            episode.run()
            checks = check(scenario, episode)
            total, ok = total + 1, ok + realized(checks)
            r = episode.result
            print(f"  épisode {e} : {'RÉALISÉ' if realized(checks) else 'NON RÉALISÉ'} · {r.distance_m:.0f} m, "
                  f"{r.steps} pas, {len(r.encounters)} obstacle(s) classé(s)")
            for c in checks:
                print(f"    [{'ok' if c.ok else 'NON'}] {c.label} ({c.detail})")
        print()
    print(f"{ok} / {total} épisodes conformes à leur scénario.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
