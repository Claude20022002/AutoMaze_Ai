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
qui traverse un labyrinthe avec Dijkstra, classe chaque obstacle rencontré et adapte son trajet.

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
├── config.py             PARTAGÉ  tailles, graines, table classe -> Action, seuils
│
├── world/                ÉQUIPE A
│   ├── maze.py           grille, murs, départ, arrivée (avec boucles)
│   ├── pathfinding.py    Dijkstra avec coûts par case
│   ├── obstacles.py      placement des obstacles sur le chemin
│   ├── car.py            position, capteur (portée), déplacement
│   └── decisions.py      Action -> coûts / cases bloquées -> recalcul
├── engine.py             ÉQUIPE A  boucle d'un épisode, mode --headless, écriture des EncounterLog
├── main.py               ÉQUIPE A  point d'entrée : python -m simulator.main
├── ui/
│   ├── grid_view.py      ÉQUIPE A  grille, chemin, voiture, obstacles (pygame)
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
    cell: tuple[int, int]      # (ligne, colonne) dans la grille
    true_label: str            # vraie classe, connue du simulateur
    image_path: Path           # image du jeu de test (ou image de remplacement)


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

**Qui fait quoi avec ces contrats :**

- L'équipe A appelle `Classifier.predict()` quand la voiture arrive près d'un `Obstacle`, applique l'`Action` correspondante et écrit une `EncounterLog`.
- L'équipe B fournit les classifieurs, la banque d'images d'obstacles et l'analyse des `EncounterLog`.
- La table classe → `Action` est dans `simulator/config.py`. Les règles finales sont validées avec l'équipe modèle.

Table proposée (à valider) :

| Classe prédite | Action | Effet sur Dijkstra |
| --- | --- | --- |
| person | `STOP_WAIT` | Case bloquée pendant quelques pas |
| bicycle | `SLOW_DOWN` | Coût de la case augmenté |
| motorcycle | `SLOW_DOWN` | Coût de la case augmenté |
| car | `REROUTE` | Case bloquée, nouveau calcul |
| bus, truck | `REROUTE_WIDE` | Case et voisines bloquées, nouveau calcul |
| unknown | `CAUTIOUS_STOP` | Case bloquée, nouveau calcul |

## 4. Équipe A — Monde et navigation (Claude Code)

L'équipe A peut tout tester avec le mode `simulation`, ou avec un faux classifieur de 5 lignes dans ses tests, sans attendre aucun modèle.

| Tâche | Fichier | Terminée quand |
| --- | --- | --- |
| Labyrinthe avec graine aléatoire | `world/maze.py` | Reproductible à partir de sa graine ; toujours au moins deux chemins de A à B |
| Dijkstra avec coût par case | `world/pathfinding.py` | Les tests sur de petites grilles donnent la même longueur qu'un BFS quand tous les coûts valent 1 |
| Obstacles et voiture | `world/obstacles.py`, `world/car.py` | Obstacles placés sur le chemin ; la voiture détecte celui qui est dans sa portée (par exemple 2 cases) |
| Règles de décision | `world/decisions.py`, `config.py` | Chaque `Action` modifie les coûts ou bloque des cases, puis relance Dijkstra depuis la position actuelle |
| Boucle de simulation | `engine.py` | Un épisode tourne aussi sans affichage (`--headless`) et écrit les `EncounterLog` |
| Affichage | `ui/grid_view.py`, `ui/hud.py` | Grille, chemin prévu, voiture, obstacles, nom du classifieur et compteurs à l'écran |
| Lancement | `main.py` | `python -m simulator.main --classifier simulation --seed 7` ouvre la fenêtre |

> **Piège.** Un labyrinthe « parfait » (généré par backtracking) n'a qu'un seul chemin entre deux cases : la voiture
> ne peut jamais contourner un obstacle. Retirez quelques murs au hasard pour créer des boucles.

**Cas limites :** aucun chemin restant (la voiture attend, puis l'épisode se termine avec le statut « bloqué »),
obstacle sur l'arrivée, personne qui libère sa case après quelques pas.

**Terminé pour A :** sur 100 graines, chaque épisode se termine à l'arrivée ou avec le statut « bloqué », sans erreur,
et `pytest tests/simulator/world` passe.

## 5. Équipe B — Perception et comparaison (agent Cursor)

L'équipe B peut tout tester sans labyrinthe, en appelant `predict()` sur des images.

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
| 21 → 30 oct. | Labyrinthe, Dijkstra, tests | A | `pytest tests/simulator/world` |
| 21 → 30 oct. | Mode simulation, banque d'obstacles avec images de remplacement | B | `pytest tests/simulator/perception` |
| 30 oct. | **Intégration 1** : la voiture traverse le labyrinthe en mode simulation (sans vue caméra) | A + B | Démo de 5 minutes |
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
| Fichiers possédés | `simulator/world/`, `engine.py`, `main.py`, `ui/grid_view.py`, `ui/hud.py` | `simulator/perception/`, `ui/camera_panel.py`, `metrics.py`, `compare.py`, notebook 06 |
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
- [ ] Taille de la grille et nombre d'obstacles par épisode pour la démo
- [ ] Règles de décision finales par classe, à valider avec l'équipe modèle

| Date | Équipe | Note |
| --- | --- | --- |
| 19 oct. 2026 | A | `contracts.py`, `config.py` et leurs tests sur `sim/monde` : **à relire par B** avant fusion dans `main` |
| 19 oct. 2026 | — | Plan créé |
