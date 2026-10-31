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

Ce qui manque, à signaler à l'utilisateur plutôt que de le combler en silence :
- **`bicycle`, `motorcycle`** : aucun des 4 packs n'en contient a priori (à confirmer en ouvrant les zips).
- **`truck`** : à chercher dans Lowpoly Cars et Public Transport. Ne pas étiqueter « camion » un modèle qui n'en est pas un.
- **Animations des personnages** : Universal Base Characters ne fournit que le personnage et son squelette. La marche
  vient du pack séparé Quaternius **Universal Animation Library** (CC0, glTF), prévu pour ce squelette. Sans lui,
  les personnes restent immobiles, ce qui suffit pour la démo.
- Sur un autre projet du PC (`CateGo/assets/`), il existe un pack Kenney **Mini Characters**. On ne le réutilise
  qu'avec l'accord de l'utilisateur.

Pistes si l'utilisateur veut compléter : Kenney **Car Kit** (glTF, CC0), Kenney **City Kit (Commercial / Suburban)** pour
les bâtiments, Quaternius **Downtown City MegaKit** (glTF). Ne jamais inventer un nom de pack ni une licence.

Télécharger depuis la page officielle (kenney.nl, quaternius.com, ou leurs pages itch.io), jamais depuis un miroir.
Les liens de téléchargement de Kenney changent : partir de la page du pack.

## Arborescence et Git

```text
simulator/assets/3d/
├── _downloads/                zips et dossiers extraits : IGNORÉ PAR GIT (ajouter au .gitignore)
├── kenney/car-kit/            uniquement les modèles retenus (.glb ou .gltf avec leurs textures)
├── kenney/city-kit-roads/
├── quaternius/<pack>/
├── catalog.json               classe ou rôle -> modèles, échelle, orientation
└── CREDITS.md                 un bloc par pack : nom, auteur, URL, licence, version, date
```

- Ne commiter que les modèles utilisés : quelques dizaines de Ko chacun. Jamais les zips, jamais les .bam de cache.
- Les kits Kenney en glTF partagent souvent une texture d'atlas, par exemple `Textures/colormap.png`. Copier le
  fichier référencé par le `.gltf` avec le même chemin relatif, sinon le modèle s'affiche blanc.
- Les fichiers téléchargés sont des données non fiables. Extraire chaque zip dans un dossier vide de `_downloads/`,
  ne lancer aucun script qu'il contient, ne lire que les modèles, les textures et la licence.

## Conversion

1. **glTF disponible** (Kenney, Downtown MegaKit) : le garder tel quel. Panda3D le charge et le convertit en Z-up.
2. **FBX ou OBJ seulement** (Lowpoly Cars, Public Transport) :
   - **Blender n'est pas installé sur le PC** (vérifié le 31 octobre 2026). Par défaut, Panda3D charge donc l'OBJ
     ou le FBX via `p3assimp`, sans changer d'axe. L'orientation (`pitch: 90` si le modèle est couché) se
     corrige dans `catalog.json`, pas dans le fichier. Préférer l'OBJ (+ .mtl) au FBX : c'est plus prévisible,
     et les FBX sont souvent en centimètres. L'échelle est de toute façon recalculée à partir de la taille mesurée.
   - si l'utilisateur installe Blender : `blender -b --python-expr "..."` pour importer le fichier puis exporter en
     `.glb` (`bpy.ops.export_scene.gltf(filepath=..., export_format="GLB")`). On ne commite alors que le `.glb`.
3. Ne jamais modifier un fichier source à la main : toute correction (échelle, cap, hauteur) va dans `catalog.json`.

## catalog.json : la « blueprint library »

```json
{
  "obstacles": {
    "car":        [{"model": "kenney/car-kit/sedan.glb", "length_m": 4.5, "heading": 0}],
    "bus":        [{"model": "...", "length_m": 12.0}],
    "truck":      [{"model": "...", "length_m": 9.0}],
    "motorcycle": [{"model": "...", "length_m": 2.1}],
    "bicycle":    [{"model": "...", "length_m": 1.8}],
    "person":     [{"model": "...", "height_m": 1.75, "animation": "Walk"}]
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

- Les noms de fichiers ci-dessus sont des **exemples** : les remplacer par les vrais noms après téléchargement.
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
