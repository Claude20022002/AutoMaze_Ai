# Données du simulateur

## casablanca_drive.graphml

Réseau routier carrossable d'une partie de Casablanca, du Maârif au centre-ville
(zone `MAP_BBOX` de `simulator/config.py` : 7,645° O – 7,612° O, 33,578° N – 33,600° N).

- Source : OpenStreetMap, téléchargé avec OSMnx 2.1 le 19 octobre 2026.
- Contenu : 1 387 intersections et 2 907 tronçons orientés (sens uniques respectés), plus grande composante
  fortement connexe uniquement.
- Régénérer : `python -m simulator.world.road_map --download`.

Données © contributeurs OpenStreetMap, disponibles sous licence ODbL :
https://www.openstreetmap.org/copyright. Cette mention doit apparaître dans l'interface du simulateur et dans le rapport.

## casablanca_buildings.npz

Contours des bâtiments de la même zone, pour la vue 3D (`simulator/render3d/`) : 4 609 contours, projetés dans le
repère de la carte (UTM 29N, EPSG:32629), avec une hauteur par bâtiment.

- Source : OpenStreetMap (balise `building`), téléchargé avec OSMnx 2.1 le 31 octobre 2026.
- Hauteur : balise `height`, sinon `building:levels` × 3,2 m, sinon **estimée** (3 à 7 étages tirés avec une graine) :
  ce ne sont pas toujours les vraies hauteurs.
- Régénérer : `python -m simulator.render3d.buildings --download`.

Données © contributeurs OpenStreetMap, licence ODbL (même mention que la carte).
