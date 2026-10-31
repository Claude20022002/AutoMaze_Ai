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
