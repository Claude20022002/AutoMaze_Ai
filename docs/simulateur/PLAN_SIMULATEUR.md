# AutoMaze AI — Plan de conception du simulateur

> Version du 19 octobre 2026. Copie de travail du plan partagé :
> https://claude.ai/code/artifact/fdff8b06-b1c0-48c8-a481-d572f4a29f6c
> Ce fichier fait foi pour les deux agents. Toute modification passe par une PR relue par l'autre équipe.

## 0. Qui fait quoi

| Équipe | Agent | Branche | Dossier de travail |
| --- | --- | --- | --- |
| **A — Monde et navigation** | Claude Code (VS Code) | `sim/monde` | `AutoMaze_Ai/` (dépôt principal) |
| **B — Perception et comparaison** | Agent Claude dans Cursor | `sim/perception` | `AutoMaze_Ai-perception/` (git worktree) |
| Modèle (notebooks 01 à 05) | L'équipe étudiante (3 personnes) | `main` | Colab + Drive |

**Pourquoi un git worktree pour l'équipe B ?** Deux éditeurs ouverts sur le même dossier partagent la même branche :
un `git checkout` fait par un agent changerait les fichiers sous les pieds de l'autre. Le worktree donne
à l'agent Cursor son propre dossier, sur sa propre branche, avec le même historique Git.

Mise en place (une seule fois, depuis le dossier `AutoMaze_Ai/`) :

```bash
git switch -c sim/monde                                   # branche de l'équipe A (Claude Code)
git worktree add ../AutoMaze_Ai-perception -b sim/perception main   # dossier + branche de l'équipe B
```

Ouvrez ensuite `AutoMaze_Ai-perception/` dans Cursor, et `AutoMaze_Ai/` dans VS Code.

## 1. Objectif et périmètre

Deux équipes construisent le simulateur en parallèle. Une version de démonstration doit tourner pour la séance 2
(25 novembre), et la version branchée sur notre CNN pour la soutenance (16 décembre). Le simulateur montre une voiture
qui roule dans les vraies rues d'une partie de Casablanca (du Maârif au centre-ville, carte OpenStreetMap),
suit le plus court chemin calculé par Dijkstra, classe chaque obstacle rencontré et adapte son trajet.

Les deux équipes ne dépendent l'une de l'autre que par `simulator/contracts.py`, figé le 21 octobre (section 3).

**Trois classifieurs interchangeables**, choisis au lancement :

| Mode | Rôle | Disponible |
| --- | --- | --- |
| `simulation` | Vraie classe, plus un taux d'erreur réglable. Teste la logique sans aucun modèle | Dès la première semaine |
| `reference` | Modèle préentraîné, affiné sur nos 6 classes. Sert à tester le simulateur avec un vrai modèle et de point de comparaison | Mi-novembre, une fois les recadrages prêts |
| `ours` | Notre CNN entraîné de zéro, chargé depuis `models/metadata.json` | Version finale le 9 décembre |

**Hors périmètre :** détection d'objets dans une image entière, vidéo, physique réaliste. Le simulateur connaît
la position des obstacles ; le modèle ne fait que les classer.

> **Règle du cours à faire valider.** Les consignes interdisent les modèles préentraînés pour *notre* modèle.
> Le modèle de référence reste donc séparé : il est étiqueté « Référence préentraînée » partout et n'est jamais
> présenté comme notre travail. Accord de l'enseignant à demander en séance 1 (4 novembre). Sans accord, le mode
> `reference` est retiré et le reste du plan ne change pas.

## 2. Architecture

```text
simulator/
├── contracts.py          PARTAGÉ  Prediction, Classifier, Obstacle, Action, EncounterLog
├── config.py             PARTAGÉ  zone de la carte, lieux, graines, table classe -> Action, seuils
│
├── world/                ÉQUIPE A
│   ├── road_map.py       carte OSM de Casablanca (OSMnx), lieux de départ et d'arrivée
│   ├── pathfinding.py    Dijkstra écrit à la main, coûts dynamiques par tronçon
│   ├── obstacles.py      placement des obstacles sur le chemin
│   ├── car.py            position, capteur (portée), déplacement
│   └── decisions.py      Action -> pénalités / tronçons bloqués -> recalcul
├── engine.py             ÉQUIPE A  boucle d'un épisode, mode --headless, écriture des EncounterLog
├── main.py               ÉQUIPE A  point d'entrée : python -m simulator.main
├── ui/
│   ├── map_view.py       ÉQUIPE A  rues, itinéraire, voiture, obstacles (pygame)
│   ├── hud.py            ÉQUIPE A  nom du classifieur, compteurs
│   └── camera_panel.py   ÉQUIPE B  image de l'obstacle, prédiction, vraie classe, probabilités
│
├── perception/           ÉQUIPE B
│   ├── obstacle_bank.py  images de data/crops/test/<classe>/ ou images de remplacement
│   ├── simulated.py      mode SIMULATION
│   ├── ours.py           adaptateur de notre CNN (metadata.json)
│   ├── reference.py      adaptateur du modèle préentraîné (reference_metadata.json)
│   └── threshold.py      seuil « Inconnu »
├── metrics.py            ÉQUIPE B  agrégation des EncounterLog
└── compare.py            ÉQUIPE B  python -m simulator.compare --episodes 50

tests/simulator/world/        ÉQUIPE A
tests/simulator/perception/   ÉQUIPE B
notebooks/06_reference_pretrained.ipynb   ÉQUIPE B (affinage de la référence sur Colab)
```

À chaque obstacle, `engine.py` (A) demande une image à `obstacle_bank` (B) et la passe au classifieur choisi (B).
La `Prediction` renvoyée détermine une `Action` (A), et Dijkstra recalcule le trajet (A).

## 3. Contrats d'interface (figés le 21 octobre)

`simulator/contracts.py` est écrit en premier, par l'équipe A, puis relu et approuvé par l'équipe B avant tout autre code.
**Le fichier `simulator/contracts.py` fait foi** ; l'extrait ci-dessous en donne l'essentiel. Le code réel ajoute :

- une validation dans `Prediction` (label connu, 6 probabilités de somme 1, score entre 0 et 1) : un classifieur
  qui ne respecte pas le contrat échoue immédiatement, au lieu de produire une décision fausse ;
- `Obstacle.image_path` peut valoir `None` (image de remplacement générée tant que les recadrages n'existent pas) ;
- `EncounterLog.correct` et `EncounterLog.severe_error` (personne non reconnue), utilisés par `metrics.py` ;
- `config.check_metadata_classes()`, qui refuse un modèle dont l'ordre des classes diffère
  (piège classique : `ImageFolder` trie les classes par ordre alphabétique).

Tests : `python -m pytest tests/simulator/test_contracts.py`.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

# Ordre = indices de sortie des modèles. Lu depuis models/metadata.json quand il existe.
CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
UNKNOWN = "unknown"  # affiché quand le score est sous le seuil ; jamais une classe entraînée


@dataclass(frozen=True)
class Prediction:
    label: str                 # une valeur de CLASSES, ou UNKNOWN
    score: float               # score softmax maximal (pas une probabilité calibrée)
    probs: tuple[float, ...]   # 6 valeurs, dans l'ordre de CLASSES
    infer_ms: float            # temps d'inférence mesuré


class Classifier(ABC):
    name: str  # affiché dans l'interface : "SIMULATION", "Référence préentraînée", "Notre CNN"

    @abstractmethod
    def predict(self, image_rgb: np.ndarray, true_label: str | None = None) -> Prediction:
        """image_rgb : uint8, forme (H, W, 3), RGB, taille quelconque.
        Chaque classifieur applique lui-même son prétraitement.
        true_label n'est utilisé que par le mode simulation."""


@dataclass(frozen=True)
class Obstacle:
    edge: tuple[int, int, int]  # tronçon de la carte OSM : (intersection de départ, d'arrivée, clé)
    true_label: str             # vraie classe, connue du simulateur
    image_path: Path | None     # image du jeu de test ; None = image de remplacement
    offset_m: float = 0.0       # position sur le tronçon, pour l'affichage


class Action(Enum):
    STOP_WAIT = "s'arrêter et attendre"
    SLOW_DOWN = "ralentir"
    REROUTE = "contourner"
    REROUTE_WIDE = "contourner largement"
    CAUTIOUS_STOP = "arrêt prudent"


@dataclass(frozen=True)
class EncounterLog:            # une ligne par obstacle rencontré, écrite en CSV
    episode: int
    step: int
    classifier: str
    true_label: str
    predicted_label: str
    score: float
    action: str
    infer_ms: float
```

**Point d'entrée de l'équipe B** (accord du 19 octobre) : `simulator/perception/__init__.py` expose deux fonctions,
utilisées par `engine.py` et `main.py` de l'équipe A :

```python
def make_classifier(name: str, config: SimConfig) -> Classifier:
    """name : "simulation", "reference" ou "ours". Erreur claire si le modèle demandé n'est pas disponible."""

def make_image_provider(config: SimConfig) -> Callable[[Obstacle], np.ndarray]:
    """Renvoie une fonction qui donne l'image RGB uint8 (H, W, 3) d'un obstacle : une image de
    data/crops/test/<vraie classe>/ si elle existe, sinon une image de remplacement. Même obstacle = même image."""
```

**Vue caméra** (`simulator/ui/camera_panel.py`, équipe B) : `main.py` l'utilise dès qu'elle existe, sinon il affiche
une vue provisoire. Interface attendue :

```python
class CameraPanel:
    def __init__(self, rect: pygame.Rect) -> None: ...          # zone de 408 × 230 px à droite de la fenêtre
    def draw(self, surface: pygame.Surface, detection: Detection | None) -> None: ...
    # Detection (simulator/engine.py) : obstacle, image_rgb, prediction (avec probs), action
```

Couleurs et polices communes : `simulator/ui/theme.py` (`CLASS_COLORS`, `font`, `draw_text`).

**Ce que l'équipe A fournit à l'équipe B** (`simulator/engine.py`) : `Episode(graph, classifier, config, episode,
image_for=...)` puis `.run()` renvoie un `EpisodeResult` (statut, distance, recalculs, **collisions**, attente,
liste des `EncounterLog`). `save_encounters_csv(results, path)` écrit le CSV lu par `metrics.py`. Les obstacles
se comportent selon leur **vraie** classe (`SimConfig.clear_steps`) : une personne finit de traverser, un véhicule
reste arrêté. Une mauvaise classification peut donc provoquer une collision, ce qui rend la comparaison parlante.

**Qui fait quoi avec ces contrats :**

- L'équipe A appelle `Classifier.predict()` quand la voiture arrive près d'un `Obstacle`, applique l'`Action` correspondante et écrit une `EncounterLog`.
- L'équipe B fournit les classifieurs, la banque d'images d'obstacles et l'analyse des `EncounterLog`.
- La table classe → `Action` est dans `simulator/config.py`. Les règles finales sont validées avec l'équipe modèle.

Table proposée (à valider) :

| Classe prédite | Action | Effet sur Dijkstra |
| --- | --- | --- |
| person | `STOP_WAIT` | Tronçon bloqué pendant quelques pas, puis la voiture repart |
| bicycle | `SLOW_DOWN` | Coût du tronçon multiplié (la voiture ralentit) |
| motorcycle | `SLOW_DOWN` | Coût du tronçon multiplié (la voiture ralentit) |
| car | `REROUTE` | Tronçon bloqué, nouveau calcul |
| bus, truck | `REROUTE_WIDE` | Tronçon bloqué et tronçons voisins pénalisés, nouveau calcul |
| unknown | `CAUTIOUS_STOP` | Tronçon bloqué, nouveau calcul |

## 4. Équipe A — Monde et navigation (Claude Code)

L'équipe A peut tout tester avec le mode `simulation`, ou avec un faux classifieur de 5 lignes dans ses tests, sans attendre aucun modèle.

| Tâche | Fichier | Terminée quand |
| --- | --- | --- |
| Carte réelle de Casablanca | `world/road_map.py`, `assets/casablanca_drive.graphml` | Carte OSM téléchargée une fois et versionnée ; sens uniques respectés ; plus grande composante fortement connexe |
| Dijkstra avec coûts dynamiques | `world/pathfinding.py` | Écrit à la main (tas binaire) ; même coût que `networkx.dijkstra_path_length` sur 20 graphes aléatoires et sur la carte |
| Obstacles et voiture | `world/obstacles.py`, `world/car.py` | Obstacles placés sur le chemin ; la voiture détecte celui qui est dans sa portée (par exemple 80 m le long de l'itinéraire) |
| Règles de décision | `world/decisions.py`, `config.py` | Chaque `Action` pénalise ou bloque des tronçons, puis relance Dijkstra depuis la position actuelle |
| Boucle de simulation | `engine.py` | Un épisode tourne aussi sans affichage (`--headless`) et écrit les `EncounterLog` |
| Affichage | `ui/map_view.py`, `ui/hud.py` | Rues, itinéraire prévu, nom de la rue, voiture, obstacles, nom du classifieur et compteurs à l'écran |
| Lancement | `main.py` | `python -m simulator.main --classifier simulation --seed 7` ouvre la fenêtre ; `--screenshot capture.png` produit une image sans fenêtre |

> **Piège.** Une carte découpée dans une ville contient des impasses et des sens uniques qui sortent de la zone :
> certains itinéraires deviennent impossibles. `road_map.py` ne garde que la plus grande composante fortement
> connexe, où chaque intersection peut rejoindre toutes les autres.

**Cas limites :** aucun chemin restant (la voiture attend, puis l'épisode se termine avec le statut « bloqué »),
obstacle sur l'arrivée, personne qui libère son tronçon après quelques pas.

**Terminé pour A :** sur 100 graines, chaque épisode se termine à l'arrivée ou avec le statut « bloqué », sans erreur,
et `pytest tests/simulator/world` passe.

## 5. Équipe B — Perception et comparaison (agent Cursor)

L'équipe B peut tout tester sans carte, en appelant `predict()` sur des images.

| Tâche | Fichier | Terminée quand |
| --- | --- | --- |
| Mode simulation | `perception/simulated.py` | Renvoie la vraie classe sauf dans X % des cas (réglable, avec graine) ; nom affiché **SIMULATION — modèle réel non chargé** |
| Banque d'obstacles | `perception/obstacle_bank.py` | Tire une image de `data/crops/test/<classe>/` ; sans recadrages, génère des images de remplacement (fond de couleur + nom de la classe) |
| Adaptateur de notre CNN | `perception/ours.py` | Lit `models/metadata.json` et applique **exactement** le prétraitement du notebook 04 ; erreur claire si les poids manquent |
| Référence préentraînée | `perception/reference.py`, `notebooks/06_reference_pretrained.ipynb` | Affinée sur nos recadrages (section 6), chargée depuis `models/reference_metadata.json` |
| Seuil « Inconnu » | `perception/threshold.py` | Sous le seuil lu dans les métadonnées, le label devient `UNKNOWN` |
| Vue caméra | `ui/camera_panel.py` | Image de l'obstacle, classe prédite, vraie classe, score, barres des 6 probabilités |
| Comparaison | `metrics.py`, `compare.py` | `python -m simulator.compare --episodes 50` exécute les mêmes épisodes avec chaque classifieur et écrit un tableau dans `reports/` |

> **Piège.** Si le prétraitement du simulateur diffère de celui du notebook (BGR ou RGB, taille, normalisation),
> le modèle se dégrade sans aucun message d'erreur. Ajoutez un test qui compare les sorties du simulateur et du
> notebook sur 10 images fixées.

**Terminé pour B :** les trois classifieurs passent les mêmes tests de contrat (forme de `probs`, label valide,
temps mesuré), et `pytest tests/simulator/perception` passe.

## 6. Modèle préentraîné de référence

Recommandation : **ResNet-18 préentraîné sur ImageNet** (torchvision), dont on affine la dernière couche sur **nos**
recadrages et **notre** découpage (`data/splits.csv`). Seule l'initialisation change : la comparaison est équitable.

| Option | Atouts | Limites |
| --- | --- | --- |
| ResNet-18 ImageNet, affiné (recommandé) | Déjà dans torchvision, environ 45 Mo, affinage de quelques minutes sur le GPU de Colab | Attend des images plus grandes (224 px) : il faut agrandir nos recadrages |
| MobileNetV3-Small, affiné | Très rapide sur CPU, donc fluide dans le simulateur | Souvent un peu moins précis |
| ImageNet sans affinage | Aucun entraînement | Pas de classe « personne » dans ImageNet |
| CLIP en zéro-shot | Fonctionne sans nos données | Dépendance lourde ; comparaison peu parlante |

Avant que les recadrages existent (jusqu'à mi-novembre), le simulateur est testé avec le mode `simulation`.

**Protocole de comparaison** (même jeu de test, mêmes graines) :

1. **Hors simulateur** (notebook 05) : F1 macro, rappel « personne », matrice de confusion, temps d'inférence CPU, nombre de paramètres.
2. **Dans le simulateur** (`compare.py`, 50 épisodes) : part des obstacles bien classés, décisions inadaptées, fautes graves (une personne classée autrement), épisodes arrivés, longueur du trajet, nombre de recalculs.

| Mesure | Notre CNN | Référence préentraînée |
| --- | --- | --- |
| F1 macro (test) |  |  |
| Rappel « personne » (test) |  |  |
| Fautes graves sur 50 épisodes |  |  |
| Temps moyen par décision (ms, CPU) |  |  |
| Nombre de paramètres |  |  |

La référence sera probablement meilleure : elle a appris sur environ 1,2 million d'images avant d'être affinée.
Le résultat intéressant est l'**écart** et son explication. Fichiers nommés `reference_*`, nom affiché
« Référence préentraînée », jamais dans `metadata.json`.

## 7. Planning

| Date | Jalon | Équipe | Vérification |
| --- | --- | --- | --- |
| 21 oct. | `contracts.py` et `config.py` écrits et approuvés | A écrit, B relit | PR fusionnée dans `main` |
| 21 → 30 oct. | Carte de Casablanca, Dijkstra, tests | A | `pytest tests/simulator/world` |
| 21 → 30 oct. | Mode simulation, banque d'obstacles avec images de remplacement | B | `pytest tests/simulator/perception` |
| 30 oct. | **Intégration 1** : la voiture va du Twin Center à la place Mohammed V en mode simulation (sans vue caméra) | A + B | Démo de 5 minutes |
| 31 oct. → 13 nov. | Obstacles dynamiques, décisions, rendu pygame, HUD | A | Tests + démo |
| 31 oct. → 13 nov. | Vue caméra, métriques, `compare.py` | B | Tests + démo |
| 4 nov. | **Séance 1** : demander l'accord pour la référence préentraînée | Étudiants | Réponse notée dans le journal |
| 13 nov. | **Intégration 2** : démo complète en mode simulation | A + B | Démo de 5 minutes |
| 14 → 20 nov. | Référence préentraînée (notebook 06 + `reference.py`), si accord | B | Tests de contrat |
| 25 nov. | **Séance 2** : démo du simulateur | Étudiants | — |
| 26 nov. → 11 déc. | Branchement de notre CNN, comparaison finale | B (A en soutien) | `compare.py` sur 50 épisodes |
| 9 déc. | Modèle des étudiants figé | Étudiants | `models/metadata.json` final |
| 16 déc. | **Soutenance** | Tous | — |

Une intégration en retard décale la suivante, jamais les séances.

## 8. Organisation Git, intégration et tests

| Élément | Équipe A (Claude Code) | Équipe B (agent Cursor) |
| --- | --- | --- |
| Branche | `sim/monde` | `sim/perception` |
| Dossier | `AutoMaze_Ai/` | `AutoMaze_Ai-perception/` (worktree) |
| Fichiers possédés | `simulator/world/`, `engine.py`, `main.py`, `ui/map_view.py`, `ui/hud.py` | `simulator/perception/`, `ui/camera_panel.py`, `metrics.py`, `compare.py`, notebook 06 |
| Tests | `tests/simulator/world/` | `tests/simulator/perception/` |

**Fichiers partagés**, modifiés seulement par une PR approuvée par les deux équipes : `simulator/contracts.py`,
`simulator/config.py`, `requirements.txt`, ce plan.

**Règles :**

- PR petites et fréquentes (au moins une par semaine) vers `main`, plutôt qu'une grosse fusion la veille d'une séance.
- `pytest` passe avant chaque push ; une fusion dans `main` exige que tous les tests passent.
- Avant de commencer une tâche, chaque agent fait `git fetch` puis `git merge origin/main` dans sa branche pour récupérer les contrats à jour.
- Le simulateur tourne **en local** : pygame ouvre une fenêtre, ce que Colab ne permet pas. Les poids sont copiés depuis le Drive dans `models/` (ignoré par Git).
- `pygame` est ajouté à `requirements.txt` (installation locale uniquement).

## 9. Consignes pour les deux agents

Ces règles s'appliquent à Claude Code comme à l'agent Cursor.

1. **Ne jamais modifier les fichiers de l'autre équipe.** Un besoin de changement chez l'autre équipe = une note dans la PR ou dans la section 11 de ce fichier.
2. **Ne jamais remplir les `TODO(équipe)` des notebooks 01 à 05.** Le choix des données, le prétraitement, l'entraînement et l'évaluation sont le travail noté des étudiants.
3. **Aucun poids préentraîné hors du mode `reference`.** `ours.py` ne charge que `models/metadata.json` ; aucun modèle de secours.
4. **Ne jamais inventer de résultats.** Pas de métrique, de poids ou de prédiction fabriqués ; sans modèle, l'interface affiche **SIMULATION — modèle réel non chargé**.
5. **Ne pas commiter de données, de poids ni de vidéos** (`data/`, `models/*.pt`, `videos/` sont ignorés par Git).
6. **Aucune commande Git destructive** (`reset --hard`, `push --force`, réécriture d'historique) sans l'accord de l'utilisateur.
7. **Styles communs :** Python 3.11+, annotations de type, docstrings et commentaires en français, `pytest` pour les tests.

## 10. Risques

Données cartographiques : © contributeurs OpenStreetMap, licence ODbL. La mention doit apparaître dans l'interface et le rapport.


| Risque | Effet | Parade |
| --- | --- | --- |
| L'enseignant refuse le modèle préentraîné | Le mode `reference` disparaît | Question posée en séance 1 ; le reste du plan n'en dépend pas |
| Les recadrages arrivent tard | Pas de vraies images dans le simulateur | Images de remplacement et mode `simulation` |
| Prétraitement différent entre notebook et simulateur | Le modèle se dégrade sans message d'erreur | Test de concordance sur 10 images |
| Un contrat change en cours de route | Une équipe est bloquée | Contrats figés le 21 octobre, toute modification par PR commune |
| Les deux agents modifient le même fichier | Conflit de fusion | Propriété des fichiers (section 8) et worktree séparé |
| Le simulateur prend du temps sur le travail du modèle | Notebooks en retard pour la séance 2 | Le modèle reste prioritaire ; le simulateur s'arrête à sa version de démonstration si besoin |

## 11. Questions ouvertes et journal de coordination

- [x] Composition des équipes : A = Claude Code, B = agent Claude dans Cursor (19 oct. 2026)
- [ ] Accord de l'enseignant sur le modèle de référence (4 nov.)
- [ ] Itinéraire et nombre d'obstacles par épisode pour la démo
- [ ] Règles de décision finales par classe, à valider avec l'équipe modèle

| Date | Équipe | Note |
| --- | --- | --- |
| 19 oct. 2026 | A | Interface pygame (`ui/map_view.py`, `ui/hud.py`, `ui/theme.py`) et `main.py` terminés. `python -m simulator.main` fonctionne avec un classifieur de démonstration tant que `simulator.perception` n'existe pas. **B : interface `CameraPanel` décrite section 3** |
| 19 oct. 2026 | A | Carte, Dijkstra, obstacles, voiture, décisions et `engine.py` terminés (52 tests). Sur 100 épisodes : classifieur parfait 96 arrivés, 4 bloqués, 0 collision ; classifieur « toujours vélo » 131 collisions. **B : fournir `make_classifier` et `make_image_provider` (section 3)** |
| 19 oct. 2026 | A | **Changement : grille remplacée par la carte OSM de Casablanca.** `Obstacle.cell` devient `Obstacle.edge` (+ `offset_m`). Aucun impact sur les tâches de B, à confirmer à la relecture |
| 19 oct. 2026 | A | `contracts.py`, `config.py` et leurs tests sur `sim/monde` : **à relire par B** avant fusion dans `main` |
| 19 oct. 2026 | — | Plan créé |
