---
name: asset-pipeline
description: Récupérer, trier, convertir, normaliser et créditer les assets 3D (véhicules, routes, panneaux, feux, bâtiments, personnages) de Kenney et Quaternius pour le simulateur 3D d'AutoMaze AI. À utiliser avant de télécharger ou d'ajouter un modèle 3D au dépôt, de modifier simulator/assets/3d/ ou catalog.json, ou de changer les crédits affichés.
---

# Pipeline d'assets 3D : Kenney et Quaternius

## Sources retenues

Licences et formats relevés le 31 octobre 2026. **Relire la licence incluse dans chaque zip au téléchargement.**

| Besoin | Pack | Formats | Licence |
| --- | --- | --- | --- |
| Voitures, bus, camions, taxi, police… | Kenney **Car Kit** (kenney.nl/assets/car-kit) | OBJ, FBX, **glTF** | CC0 |
| Routes, carrefours, **panneaux et feux** (depuis la v2.1) | Kenney **City Kit (Roads)** | OBJ, FBX, DAE, STL, **glTF** | CC0 |
| Bâtiments, décor urbain | Kenney **City Kit (Commercial)** et **(Suburban)** | dont **glTF** | CC0 |
| Ville modulaire plus détaillée | Quaternius **Downtown City MegaKit** | OBJ, FBX, **glTF** | usage libre, y compris éducatif : vérifier le fichier de licence |
| Voitures supplémentaires | Quaternius **Lowpoly Cars** | FBX, OBJ, Blend (**pas de glTF**) | CC0 (le SUV de police est réservé aux abonnés Patreon) |
| Bus, taxis, ambulances | Quaternius **Lowpoly Public Transport** | Blend, OBJ, FBX | CC0 |
| Personnes animées, vélos, motos | Rechercher sur kenney.nl/assets (filtre 3D) et quaternius.com (packs de personnages) | à vérifier | à vérifier |

Les packs vélo, moto et personnage ne sont pas encore identifiés avec certitude. Vérifier sur les sites avant de les
citer, et ne jamais inventer un nom de pack ni une licence. S'il en manque un, le signaler à l'utilisateur
plutôt que de le remplacer en silence.

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
2. **FBX ou OBJ seulement** (Quaternius Lowpoly Cars) :
   - si Blender est installé : `blender -b --python-expr "..."` pour importer le fichier puis exporter en
     `.glb` (`bpy.ops.export_scene.gltf(filepath=..., export_format="GLB")`) ;
   - sinon : Panda3D charge OBJ et FBX via `p3assimp` sans changer d'axe. Corriger l'orientation dans
     `catalog.json` (`pitch: 90`), pas en modifiant le fichier.
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
