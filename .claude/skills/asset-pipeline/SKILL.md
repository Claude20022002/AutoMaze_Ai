---
name: asset-pipeline
description: Récupérer, trier, convertir, normaliser et créditer les assets 3D (véhicules, routes, panneaux, feux, bâtiments, personnages) de Kenney et Quaternius pour le simulateur 3D d'AutoMaze AI. À utiliser avant de télécharger ou d'ajouter un modèle 3D au dépôt, de modifier simulator/assets/3d/ ou catalog.json, ou de changer les crédits affichés.
---

# Pipeline d'assets 3D : Kenney et Quaternius

## Packs de l'équipe (31 octobre 2026)

Formats et licences relevés sur les pages officielles. **Relire la licence incluse dans chaque zip au téléchargement.**

| Pack | Sert à | Formats | Licence | Dossier |
| --- | --- | --- | --- | --- |
| Kenney **City Kit (Roads)** v2.1 | Tronçons de route, carrefours, **panneaux et feux** | OBJ, FBX, DAE, STL, **glTF** | CC0 | `kenney/city-kit-roads/` |
| Quaternius **Lowpoly Cars** | Voiture de l'ego, obstacles `car`, trafic d'ambiance | FBX, OBJ, Blend (**pas de glTF**) | CC0 | `quaternius/lowpoly-cars/` |
| Quaternius **Lowpoly Public Transport** | Obstacles `bus` (et taxis, ambulances pour le trafic) | Blend, OBJ, FBX (**pas de glTF**) | CC0 | `quaternius/public-transport/` |
| Quaternius **Universal Base Characters** | Obstacles `person` | OBJ, FBX, **glTF** ; squelette humanoïde | CC0 | `quaternius/universal-base-characters/` |
| **Poly Pizza** (poly.pizza), modèle par modèle | Obstacles `bicycle`, `motorcycle`, `truck` | **.glb** | **CC0 ou CC-BY, selon le modèle** | `polypizza/<classe>/` |
| **Sketchfab**, modèle par modèle | `bicycle`, `motorcycle` si Poly Pizza ne suffit pas | glTF (export automatique) + format d'origine | **variable selon le modèle** | `sketchfab/<classe>/` |
| **Mixamo** (Adobe) | Animations des personnes (marche, attente) | FBX | conditions Adobe (ci-dessous) | **jamais dans Git** : `_local/mixamo/` |

### Modèles pris un par un (Poly Pizza, Sketchfab)

- **Lire la licence de chaque modèle** sur sa page avant de le télécharger, puis la recopier dans `catalog.json` et
  dans `CREDITS.md`, avec le titre, l'auteur et l'URL de la page.
- Ordre de préférence :
  1. **CC0** ;
  2. **CC-BY**, avec crédit obligatoire : titre, auteur, lien et licence ;
  3. **CC-BY-SA**, seulement si l'utilisateur l'accepte, car la licence s'étend aux modifications.
- **À refuser** :
  - les licences **NC** et **ND**, sauf accord explicite de l'utilisateur, qui doit être noté ;
  - la **licence Standard ou Editorial du Sketchfab Store** (modèles payants).
- Sur Poly Pizza, la plupart des modèles viennent de l'ancien Google Poly : beaucoup sont en CC-BY.
  Sketchfab demande un compte pour télécharger, et la fenêtre de téléchargement donne le texte de crédit à recopier.
- Choisir des modèles **low-poly** (moins de 20 000 triangles) dont le style colle à Kenney et Quaternius.
  Un vélo très détaillé au milieu de voitures en blocs choque, et coûte cher sur le GPU intégré.
- Un modèle par classe suffit pour commencer. On en ajoute ensuite 2 ou 3 par classe, pour la variété.

### Mixamo (animations des personnes)

- Conditions relevées dans la FAQ Mixamo (communauté Adobe, rédigée en 2022, à revérifier) :
  - l'usage est libre dans un projet, y compris un jeu ;
  - **interdit de redistribuer les fichiers bruts** (personnage ou animation) ;
  - **interdit pour entraîner un modèle d'apprentissage automatique** ;
  - aucun crédit n'est exigé.
- Conséquences pour le dépôt :
  - les FBX Mixamo et tout fichier qui en dérive (.glb fusionné, .bam) vont dans `simulator/assets/3d/_local/`,
    **ignoré par Git**. Chaque membre de l'équipe les télécharge lui-même ;
  - on ne commite qu'une note de procédure ;
  - ces animations ne servent qu'à l'affichage, jamais aux données des notebooks.
- Procédure :
  1. Exporter un personnage d'Universal Base Characters en FBX.
  2. L'envoyer sur mixamo.com et le laisser s'aligner sur son squelette (auto-rigger).
  3. Télécharger « Walking » et « Idle » en FBX **avec skin**, en mode **In Place** pour que le simulateur garde
     le contrôle du déplacement.
  4. Dans Blender, fusionner les animations en actions d'un même `.glb`.
  5. Le charger avec `Actor(chemin)` dans Panda3D, puis vérifier `getAnimNames()`.
  Cette procédure n'est pas encore testée.
- L'application doit **marcher sans ces fichiers**. S'ils manquent, afficher les personnes immobiles, sans erreur.
- Alternative sans compte Adobe : Quaternius **Universal Animation Library** (CC0, glTF), conçue pour le
  squelette d'Universal Base Characters. Ces fichiers-là peuvent être commités.

Ne jamais inventer un nom de pack, d'auteur ni de licence. Sur un autre projet du PC (`CateGo/assets/`), il existe un pack
Kenney **Mini Characters** : on ne le réutilise qu'avec l'accord de l'utilisateur.

Télécharger depuis la page officielle (kenney.nl, quaternius.com, ou leurs pages itch.io), jamais depuis un miroir.
Les liens de téléchargement de Kenney changent : partir de la page du pack.

## Arborescence et Git

```text
simulator/assets/3d/
├── _downloads/                zips et sources extraites : IGNORÉ PAR GIT (ajouter au .gitignore)
├── _local/                    fichiers non redistribuables (Mixamo) : IGNORÉ PAR GIT
├── kenney/city-kit-roads/     uniquement les modèles retenus (.glb, ou .gltf avec ses textures)
├── quaternius/lowpoly-cars/   .glb convertis par Blender
├── quaternius/public-transport/
├── quaternius/universal-base-characters/
├── polypizza/<classe>/        un .glb par modèle retenu
├── sketchfab/<classe>/
├── catalog.json               classe ou rôle -> modèles, échelle, orientation, licence
└── CREDITS.md                 packs : nom, auteur, URL, licence, date ; modèles isolés : titre, auteur, URL, licence
```

- Ne commiter que les modèles utilisés : quelques dizaines de Ko chacun. Jamais les zips, jamais les .bam de cache.
- Les kits Kenney en glTF partagent souvent une texture d'atlas, par exemple `Textures/colormap.png`. Copier le
  fichier référencé par le `.gltf` avec le même chemin relatif, sinon le modèle s'affiche blanc.
- Les fichiers téléchargés sont des données non fiables. Extraire chaque zip dans un dossier vide de `_downloads/`,
  ne lancer aucun script qu'il contient, ne lire que les modèles, les textures et la licence.

## Conversion

1. **glTF disponible** (Kenney, Downtown MegaKit) : le garder tel quel. Panda3D le charge et le convertit en Z-up.
2. **FBX ou OBJ seulement** (Lowpoly Cars, Public Transport, certains modèles Sketchfab) : **convertir en .glb avec
   Blender 5.2**, installé dans `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`. Le script
   [scripts/to_glb.py](scripts/to_glb.py) a été testé :

   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b --factory-startup `
       -P .claude/skills/asset-pipeline/scripts/to_glb.py -- <entree.fbx|obj> simulator/assets/3d/<source>/<pack>/<nom>.glb
   ```

   Test du 31 octobre 2026 : un OBJ Y-up chargé directement par Panda3D apparaît **couché** (sa longueur part vers Z),
   alors que le .glb converti est à l'endroit (longueur selon Y, hauteur selon Z). On ne commite que le `.glb`,
   jamais la source FBX ou OBJ. L'échelle est recalculée au chargement à partir de la taille mesurée : un FBX en
   centimètres ne pose donc pas de problème.
3. Ne jamais modifier un fichier source à la main : toute correction (échelle, cap, hauteur) va dans `catalog.json`.

## catalog.json : la « blueprint library »

```json
{
  "obstacles": {
    "car":        [{"model": "quaternius/lowpoly-cars/<voiture>.glb", "length_m": 4.5, "heading": 0,
                    "license": "CC0", "source": "Quaternius, Lowpoly Cars"}],
    "bus":        [{"model": "quaternius/public-transport/<bus>.glb", "length_m": 12.0, "license": "CC0"}],
    "truck":      [{"model": "polypizza/truck/<nom>.glb", "length_m": 9.0,
                    "license": "CC-BY 3.0", "author": "<auteur>", "url": "https://poly.pizza/m/<id>"}],
    "motorcycle": [{"model": "polypizza/motorcycle/<nom>.glb", "length_m": 2.1, "license": "...", "author": "...", "url": "..."}],
    "bicycle":    [{"model": "sketchfab/bicycle/<nom>.glb", "length_m": 1.8, "license": "...", "author": "...", "url": "..."}],
    "person":     [{"model": "quaternius/universal-base-characters/<perso>.glb", "height_m": 1.75,
                    "license": "CC0", "animations": "_local/mixamo/<perso>_anims.glb"}]
  },
  "ego": {"model": "...", "length_m": 4.3},
  "traffic": [{"model": "...", "length_m": 4.4}],
  "props": {
    "traffic_signals": {"model": "kenney/city-kit-roads/<feu>.glb", "height_m": 4.0},
    "stop":            {"model": "...", "height_m": 2.5},
    "crossing":        {"model": "...", "height_m": 2.5}
  }
}
```

- Les noms entre `< >` sont des **exemples** : les remplacer par les vrais noms après téléchargement.
- Chaque modèle qui n'est pas en CC0 porte `license`, `author` et `url`. Un test vérifie que ces champs sont
  présents et que `CREDITS.md` mentionne chaque `url`.
- `animations` peut pointer vers `_local/` : si le fichier est absent, le personnage reste immobile, sans erreur.
- Normaliser au chargement (`render3d/catalog.py`) :
  - échelle = taille réelle / plus grande dimension mesurée avec `getTightBounds()` ;
  - origine au centre de la base (z = 0 au sol) ;
  - avant du véhicule vers +Y, corrigé par `heading`.
- Tailles réelles de référence : voiture 4,5 m, bus 12 m, camion 9 m, moto 2,1 m, vélo 1,8 m, personne 1,75 m.
- Les clés `obstacles` sont exactement les 6 classes de `contracts.CLASSES`. Un test le vérifie.
  Les clés `props` correspondent aux valeurs `highway` des nœuds OSM (skill `osm-to-3d`).
- Plusieurs modèles par classe : le choix se fait avec la graine de l'obstacle (comme `_obstacle_seed`).
  Un même obstacle a alors le même modèle pour tous les classifieurs.

## Contrôle visuel

Après chaque ajout, générer une planche de contrôle :
- chaque modèle rendu hors écran, à la même échelle, à côté d'une silhouette humaine de 1,75 m ;
- enregistrer `reports/simulator/planche_assets.png` (ignoré par Git), puis la regarder.

Recettes de rendu dans le skill `panda3d-recipes`. Erreurs fréquentes :
- modèle couché : axe Y-up non converti ;
- géant ou minuscule : échelle en centimètres ;
- tout blanc : texture d'atlas manquante ;
- à l'envers : `heading`.

## Crédits

- Kenney et Quaternius en CC0 : le crédit n'est pas obligatoire, mais on crédite quand même, dans `CREDITS.md` et
  sur l'écran de l'application. OpenStreetMap (ODbL) doit **obligatoirement** être crédité : « © contributeurs OpenStreetMap ».
- Les illustrations OpenMoji (CC BY-SA 4.0) de la vue 2D gardent leur crédit tant qu'elles sont utilisées.
- Les modèles 3D ne sont **jamais** des données d'entraînement ni de test du modèle des étudiants.
