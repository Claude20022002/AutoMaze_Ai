---
name: sim3d-architecture
description: Architecture du simulateur 3D d'AutoMaze AI (Panda3D, inspiré de CARLA, carte OSM de Casablanca, assets Kenney et Quaternius). À lire avant de concevoir, modifier ou relire tout code du dossier simulator/render3d/, avant de brancher la 3D sur engine.py ou la perception, et avant toute décision sur ce que reçoit le classifieur dans la vue 3D.
---

# Simulateur 3D AutoMaze AI : architecture

## Décision (31 octobre 2026)

- **CARLA n'est pas utilisé.** Le PC de développement a un GPU AMD Radeon intégré et 12 Go de RAM. CARLA 0.9.16
  demande un GPU dédié de niveau RTX 2070 avec 8 Go de VRAM. Son client Python n'existe pas pour Python 3.13
  (wheels 3.10 à 3.12). Importer des assets demande une compilation d'Unreal Engine (plus de 130 Go).
- **Moteur retenu : Panda3D 1.10.16** (wheel `cp313-win_amd64`), avec `panda3d-gltf` 1.3 et
  `panda3d-simplepbr` 0.13. Testé sur ce PC : OpenGL 4.6 et GLSL 4.60, capteur caméra hors écran en 224×224
  en moins de 1 ms par image, chargement glTF et OBJ fonctionnels.
- **On reprend les idées de CARLA, pas son code** (tableau ci-dessous). Ursina n'est pas utilisé : c'est une
  surcouche de Panda3D qui cache les buffers hors écran dont le capteur caméra a besoin.
- **Cette session peut modifier les fichiers des équipes A et B** (choix de l'utilisateur). Chaque changement
  important est tout de même noté dans le journal de `docs/simulateur/PLAN_SIMULATEUR.md` (section 11).

## Concepts de CARLA, version AutoMaze

| CARLA | Chez nous | Où |
| --- | --- | --- |
| Serveur Unreal + client Python | Un seul processus Python. La logique (`engine.py`) ne connaît pas la 3D | `simulator/engine.py`, `simulator/render3d/` |
| Mode synchrone, `world.tick()` | `Episode.step()` à pas fixe (`SimConfig.step_seconds`). Le rendu lit l'état et interpole entre deux pas | tâche Panda3D `sim_tick` |
| Carte OpenDRIVE | Graphe OSMnx déjà projeté en mètres (UTM), transformé en maillages | skill `osm-to-3d` |
| Blueprint library | Catalogue `classe -> modèles glTF` avec échelle et orientation | `simulator/assets/3d/catalog.json` |
| Actor, `spawn_actor` | `Actor3D` : un `NodePath` relié à un `ObstacleState` ou à la voiture | `render3d/actors.py` |
| Capteur caméra RGB, `sensor.listen` | Buffer hors écran copié en RAM, converti en `np.ndarray` uint8 RGB | skill `panda3d-recipes` |
| Vérité terrain (boîtes, segmentation) | Boîte 3D de l'obstacle projetée par la lentille, donc recadrage exact | `render3d/sensors.py` |
| Traffic Manager | Trafic d'ambiance qui suit des itinéraires Dijkstra, sans effet sur les métriques | `render3d/traffic.py` |
| Météo (`WeatherParameters`) | Heure du jour, brouillard, lumière : réglages simples | `render3d/environment.py` |
| Spectateur | Caméra de poursuite, vue conducteur, vue de dessus (touche V) | `render3d/cameras.py` |
| Recorder | `EncounterLog` en CSV + graine : un épisode se rejoue à l'identique | existant |

Les chemins `render3d/*` sont la cible : ces fichiers sont à créer.

## Règles non négociables

1. **La logique ne dépend jamais du rendu.** `engine.py`, `world/` et `perception/` n'importent pas Panda3D.
   `python -m simulator.engine --episodes 50` et `python -m simulator.compare` restent sans affichage, avec des
   résultats identiques avec ou sans 3D. La 3D est une *vue* : elle lit `Episode` et ne le modifie pas.
2. **Ce que reçoit le classifieur.** Par défaut (`--perception crop`), il reçoit l'image de la banque
   d'obstacles (`make_image_provider`) : un vrai recadrage du jeu de test, ou une image de remplacement.
   Le modèle 3D low-poly de l'obstacle sert *seulement à l'affichage*. La vue montre l'image classée en
   vignette (« entrée du modèle »).
   Un mode expérimental `--perception render` peut donner au classifieur le recadrage rendu par la caméra 3D.
   Ses résultats sont toujours étiquetés « rendu 3D », jamais mélangés aux autres dans `compare.py`. Notre CNN
   apprend sur des photos : un écart sur des modèles low-poly mesure un changement de domaine, pas le modèle.
3. **Règles du cours :** aucun détecteur (YOLO ou autre), aucun poids préentraîné hors du mode `reference`.
   Le simulateur connaît la position des obstacles (vérité terrain) et le modèle ne fait que classer.
4. **Ne jamais inventer de résultats.** Sans modèle chargé, la vue affiche « SIMULATION — modèle réel non chargé ».
5. **Ne jamais modifier les `TODO(équipe)` des notebooks** ni les données. Ne pas commiter de poids, de zips ni de caches.
6. **Code et commentaires en français**, Python 3.11+ (3.13 sur ce PC), annotations de type, `pytest`.

## Ce qui existe (31 octobre 2026)

```text
simulator/
├── render3d/                 vue 3D (seul dossier qui importe panda3d)
│   ├── app.py                Simulator3D : mise en page, boucle à pas fixe, caméras, touches, --screenshot
│   ├── world_mesh.py         sol, chaussées, trottoirs et bordures, marquages et zébras, îlots, bâtiments,
│   │                         mobilier, signalisation (skill osm-to-3d)
│   ├── textures.py           textures procédurales : façades par étage, vitrines, bitume, dalles, sol
│   ├── street_furniture.py   palmiers, arbres, lampadaires (maillages procéduraux) et leurs emplacements
│   ├── environment.py        dôme de ciel, soleil et ombres autour de la voiture, lumière d'appoint, brume
│   ├── buildings.py          bâtiments OSM : téléchargement et lecture de assets/casablanca_buildings.npz
│   ├── actors.py             EgoCar, ObstacleView (comportement selon la vraie classe), RouteLine
│   ├── hud.py                SidePanel : classifieur, état, vue caméra (image classée), probabilités, crédits
│   ├── dashboard.py          tableau de bord conducteur : compteur, état de conduite, objet devant, itinéraire
│   ├── minimap.py            minicarte : seconde caméra orthographique, orientée dans le sens de marche
│   ├── masks.py              masques de caméra : vue principale, minicarte, carte d'ombre
│   ├── catalog.py            catalogue des modèles et normalisation (skill asset-pipeline)
│   ├── build_assets.py       production des .glb depuis les téléchargements
│   ├── blender_to_glb.py     script exécuté par Blender
│   └── contact_sheet.py      planches de contrôle des modèles
├── assets/3d/                .glb, catalog.json, CREDITS.md ; _cache/ (ville en .bam), ignoré par Git
└── ...                       inchangé : engine.py, world/, perception/, metrics.py, compare.py, main.py (vue 2D)
tests/simulator/render3d/     catalogue, géométrie pure, vraie carte, capture 3D dans un sous-processus
```

Pas encore faits : capteur caméra RGB 3D et recadrage (mode `--perception render`, recettes dans
`panda3d-recipes`), trafic d'ambiance, réglages d'heure et de météo, parcs OSM (`leisure=park`).

## Mise en page et interface

- La vue 3D n'occupe que sa zone (`DisplayRegion` de la caméra principale) : colonne de droite pour le panneau
  (`hud.PANEL_W` = 400 px), bas de la zone 3D pour le tableau de bord en vue conducteur (`app.DASHBOARD_H`).
  Rien ne recouvre la scène. `compute_layout` (testée) calcule les zones ; `apply_layout` les réapplique à chaque
  image si la fenêtre change de taille, et corrige le format de la lentille (ShowBase le remet à celui de la
  fenêtre entière).
- Interface en `pixel2d` (pixels, origine en haut à gauche, z négatif vers le bas), police DejaVu Sans (accents).
- Vue conducteur : carrosserie de l'ego cachée pour la caméra principale seulement (elle reste sur la minicarte),
  caméra à la place du conducteur. Le tableau de bord lit l'épisode (`driving_state`, `object_ahead`,
  `remaining_m`, testées sans affichage).
- Masques (`masks.py`) : la ligne d'itinéraire, les étiquettes et les repères ne projettent pas d'ombre ; les
  repères ne sont visibles que sur la minicarte ; les étiquettes et le ciel seulement dans la vue principale.

## Direction visuelle (références)

- Casablanca, « ville blanche » : façades blanches et crème Art Déco et mauresques, quelques pastels, boulevards
  bordés de palmiers, petits taxis rouges. D'où `world_mesh.BUILDING_COLORS` (pondérées), palmiers sur les axes
  `primary` et `secondary`, et la variante `petit-taxi-rouge.glb`.
- Villes procédurales low-poly : façades texturées par étage et par travée (motifs répétés, jamais coupés),
  rez-de-chaussée commerçant, ombrage doux. Rester low-poly et non photoréaliste : le GPU est intégré.
- Simulateurs de conduite : combiné d'instruments (compteur analogique), distance au véhicule devant, consigne
  en cours. Le verdict est toujours écrit en toutes lettres, pas seulement en couleur.

```powershell
python -m simulator.render3d.app                                   # fenêtre, classifieur simulation
python -m simulator.render3d.app --classifier ours --episode 3
python -m simulator.render3d.app --view conducteur --time-factor 4
python -m simulator.render3d.app --screenshot reports/simulator/vue3d.png --steps 60   # sans fenêtre
```

Touches : Espace pause · → pas à pas · ↑/↓ vitesse · V vue (poursuite, conducteur, dessus) · R · N · Échap.

## Synchronisation logique et rendu

- `Simulator3D.tick` accumule le temps réel × `time_factor`. Chaque fois que `step_seconds` s'est écoulé, elle
  appelle `episode.step()`, comme `main.py`. Le rendu interpole la voiture entre l'état précédent et l'état
  courant (`Car.position_xy()`, `Car.heading()`). Au-delà de 60 m d'écart, il n'y a pas d'interpolation : c'est
  le cas du demi-tour du moteur, qui **ramène la voiture au début du tronçon** (simplification de `engine.py`) et
  la fait sauter à l'écran.
- Les obstacles (`ObstacleView`) sont créés à chaque épisode depuis `episode.obstacles`. Le modèle est choisi avec
  `catalog.pick` et la graine de l'obstacle. Le comportement affiché suit la **vraie** classe :
  - la personne traverse entre `detected_step` et `cleared_step` ;
  - le vélo ou la moto repart après `cleared_step` ;
  - les autres véhicules restent immobiles.
- Le moteur traite voiture et obstacles comme des points. Le point de l'ego est son pare-chocs avant, celui d'un
  véhicule obstacle est son arrière : les modèles ne se chevauchent pas quand le moteur arrête la voiture 1 m avant.
- `episode.last_detection` alimente la vue caméra du HUD : l'image de la banque d'obstacles, jamais le rendu 3D.
- Nouvel épisode : `ObstacleView.destroy()` (y compris `Actor.cleanup()`), la ville reste en place.

## Performances mesurées (GPU AMD intégré, 1440×860, hors écran, ombres activées)

- Environ 30 s de construction de la ville au premier lancement, puis 5 s avec le cache .bam (environ 85 Mo,
  textures comprises). Cache invalidé par le hash de la carte, des bâtiments et `world_mesh.MESH_VERSION`, à
  incrémenter à chaque changement de géométrie, de couleur ou de texture.
- Environ 85 images/s en vue poursuite, 75 en vue conducteur, 150 vue de dessus. Éclairage par pixel
  (`setShaderAuto`), ombres 2048 px sur 180 m autour de la voiture. `--no-shadows` si la machine est plus faible.
- Les objets mobiles (`dynamique`) gardent l'éclairage classique (`setShaderOff`) : l'ombrage par pixel faisait des
  rayures (« shadow acne ») sur leurs petites facettes. Ils projettent quand même leur ombre.

## Vérifier une modification

1. `python -m pytest tests/simulator` : tous les tests passent, y compris les tests 3D en mode hors écran.
2. `python -m simulator.engine --classifier simulation --episodes 20` donne les mêmes chiffres qu'avant le changement.
3. Lancer la vue 3D, puis vérifier visuellement :
   - la voiture reste sur la route ;
   - les obstacles sont sur la voie ;
   - la vignette affichée est bien l'image classée.
