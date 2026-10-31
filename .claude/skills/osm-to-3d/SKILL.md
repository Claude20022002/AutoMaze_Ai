---
name: osm-to-3d
description: Construire la scène 3D (chaussées, trottoirs, marquages, carrefours, feux, passages piétons, panneaux stop, bâtiments) à partir de la carte OpenStreetMap de Casablanca déjà utilisée par le simulateur (graphe OSMnx projeté en UTM). À utiliser pour écrire ou corriger simulator/render3d/world_mesh.py, placer des éléments sur la route, ajouter les bâtiments OSM ou changer la zone de la carte.
---

# De la carte OSM à la scène 3D

**Implémenté dans `simulator/render3d/world_mesh.py` et `buildings.py`** (31 octobre 2026). Ce skill décrit les choix
et les pièges. Pour toute modification de géométrie, incrémenter `MESH_VERSION`, sinon l'ancien cache .bam est réutilisé.

## Pièges rencontrés

| Symptôme | Cause | Correction en place |
| --- | --- | --- |
| Hachures sur les routes vues de loin | Z-fighting entre le sol et la chaussée à 2 cm (précision du tampon de profondeur) | Sol dans le bin `background` sans écriture de profondeur ; marquages avec `setDepthOffset` |
| Construction de la ville en 112 s | `roads.buffer()` recalculé pour chaque bâtiment | Zone d'exclusion calculée une fois et `shapely.prepare` |
| Feux et panneaux sur une autre chaussée | Grands carrefours, boulevards à deux chaussées | `push_off_road` : vers le bord le plus proche puis 0,8 m au-delà, répété |
| Overpass `502` ou délai dépassé | Serveurs saturés | Miroirs à la suite (`road_map.OVERPASS_SERVERS`), `requests_timeout = 300` |

Équivalent de l'import OpenDRIVE de CARLA, mais à partir de **notre** graphe. Il n'y a pas de seconde carte :
la voiture 3D, Dijkstra et les obstacles partagent exactement les mêmes tronçons (`EdgeKey`).

## Données disponibles (relevées dans `simulator/assets/casablanca_drive.graphml`)

- Chargement : `simulator.world.road_map.load_map()`, qui renvoie un `MultiDiGraph` **projeté en UTM** (mètres, CRS dans
  `graph.graph["crs"]`). Zone : `config.MAP_BBOX`, du Maârif au centre-ville.
- Attributs des tronçons :
  - `length` ;
  - `geometry` (LineString), **seulement pour les tronçons courbes** ; sinon, ligne droite entre `x,y` des nœuds ;
  - `highway`, `lanes`, `width`, `oneway`, `maxspeed`, `name`, `junction`, `tunnel`, `reversed`.
  Une valeur peut être une **liste** quand OSMnx a fusionné plusieurs voies (comme `name` dans `street_name`).
  `lanes` et `width` sont des chaînes.
- Attributs des nœuds : `x`, `y`, `street_count`, `highway`, `ref`, `railway`. Valeurs `highway` présentes dans
  la carte : `traffic_signals`, `crossing`, `stop`. Elles ne sont que quelques-unes : ne pas compter dessus pour
  remplir la ville.

## Repère local : obligatoire

Les coordonnées UTM valent environ 3,7 millions de mètres. En float32 (le format des sommets dans Panda3D), la
précision tombe à environ 0,25 m : les maillages tremblent. Toujours passer par un repère local :

```python
@dataclass(frozen=True)
class WorldFrame:
    origin_x: float          # centre de la carte en UTM (moyenne des x des nœuds, par exemple)
    origin_y: float

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        return x - self.origin_x, y - self.origin_y      # X Panda3D = est, Y Panda3D = nord, Z = haut
```

- Cap : `H = degrees(atan2(dy, dx)) - 90`, puisque H = 0 regarde vers +Y.
- On roule **à droite** (Maroc). La voiture et les obstacles sont décalés de `LANE_OFFSET_M` (1,8 m, comme dans
  `ui/driver_view.py`) vers la droite du sens de marche.
- Le sol est plat (z = 0). Chaussée à z = 0,02 m et marquages à z = 0,03 m, pour éviter le scintillement (z-fighting).

## Chaussées et carrefours

1. **Un seul ruban par rue** : une rue à double sens apparaît deux fois (u→v et v→u). Dédoublonner avec la paire
   `frozenset({u, v})` et la géométrie, puis construire une seule fois.
2. Largeur : `width` si présent, sinon `lanes` × 3,2 m, sinon une valeur par type de route, par exemple
   primary 12 m, secondary 10 m, tertiary 8 m, residential 6,5 m. Prendre la première valeur si c'est une liste.
3. Polygones avec shapely, déjà installé avec OSMnx :
   - `line.buffer(width / 2, cap_style="flat")` pour chaque rue ;
   - puis `unary_union` par tuile : les carrefours se forment tout seuls.
4. Trottoirs : `buffer(width / 2 + 2.5)` moins la chaussée, légèrement surélevés (z = 0,12 m).
5. Triangulation : `shapely.constrained_delaunay_triangles` (shapely ≥ 2.1, trous respectés), en Python pur, donc
   testable sans GPU. Triangles remis dans le sens trigonométrique pour faire face à +Z.
6. Marquages : ligne centrale en tirets (`line.interpolate` tous les 6 m), uniquement si la rue est à double sens
   et a au moins 2 voies. Les couper à 8 m des carrefours.

## Éléments de signalisation

| Nœud OSM | Placement |
| --- | --- |
| `highway=traffic_signals` | Un feu par tronçon entrant, au coin droit, tourné vers la voiture qui arrive |
| `highway=crossing` | Bandes de passage piéton en travers de la chaussée, et un panneau |
| `highway=stop` | Panneau stop à droite du tronçon entrant, à 2 m avant le carrefour |

Les modèles viennent de `catalog.json`, section `props` (skill `asset-pipeline`). Les feux sont **décoratifs** :
ils ne changent pas la logique de `engine.py`, sauf décision explicite notée dans le plan.

## Bâtiments (facultatif, coûteux)

- Téléchargement une seule fois, comme la carte, puis travail hors ligne :
  `ox.features_from_bbox(MAP_BBOX, tags={"building": True})`. En OSMnx 2, la bbox est (ouest, sud, est, nord).
  Projeter dans le CRS du graphe, ne garder que les polygones, enregistrer à côté de la graphml. Vérifier
  la taille du fichier avant de le commiter.
- Hauteur : la balise `height`, sinon `building:levels` × 3,2 m. Sinon, une valeur tirée avec la graine, à
  **marquer comme estimée** (comme les façades générées de la vue pseudo-3D).
- Murs : un quadrilatère par côté de l'anneau extérieur. Toit : polygone triangulé. Couleurs par sommet, sans texture.
- Sans bâtiments, une bande de façades génériques le long des rues suffit pour la démo.
- Afficher la mention : « Carte : © contributeurs OpenStreetMap (ODbL) ».

## Performances et cache

- Découper en tuiles de 200 m. Une tuile = un `NodePath` = `flattenStrong()` après construction.
- La construction prend du temps : mettre en cache dans un `.bam` dont le nom contient le hash de la graphml et une
  constante `MESH_VERSION`. Cache dans un dossier ignoré par Git, reconstruit automatiquement si le hash change.
- Code de géométrie **pur**, sans Panda3D : fonctions qui renvoient sommets et triangles en numpy. Seule une petite
  couche les transforme en `GeomNode`. La géométrie se teste alors sans GPU.

## Tests attendus

- Repère local : un aller-retour `to_local` donne les mêmes points ; toutes les coordonnées locales restent sous 5 km.
- Chaque `EdgeKey` de l'itinéraire par défaut est recouvert par la chaussée (point à mi-tronçon dans l'union des polygones).
- Un tronçon à double sens ne produit qu'un seul ruban.
- La position 3D d'un obstacle (`point_on_edge` + décalage de voie) tombe sur la chaussée.
