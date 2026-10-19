# AutoMaze AI

Classification de 6 usagers de la route (personne, vélo, voiture, moto, bus, camion) par un modèle entraîné de zéro,
utilisé dans un simulateur de voiture qui traverse un labyrinthe avec Dijkstra.

Règles du cours : pas de YOLO, pas de détecteur, pas de poids préentraînés.

| Séance | Date | Livrable |
| --- | --- | --- |
| 1 | 4 novembre 2026 | Plan, choix du dataset, objectifs |
| 2 | 25 novembre 2026 | Notebooks commentés : prétraitement et évaluations |
| Soutenance | 16 décembre 2026 | Démo du simulateur et rapport final |

## Organisation

```text
notebooks/   00 à 05, exécutés sur Google Colab depuis VS Code
data/        données locales, ignorées par Git (sur Colab : dans Google Drive)
models/      poids (.pt, ignorés par Git) et metadata.json
reports/     figures pour le rapport
videos/      module vidéo, partie avancée ultérieure (ignoré par Git)
```

Le code et les notebooks sont versionnés sur GitHub. Les données, les poids et les figures sont stockés
dans le dossier Google Drive partagé `AutoMaze_Ai`.

## Installation (une fois par membre)

1. **Cloner le dépôt :** `git clone git@github.com:Claude20022002/AutoMaze_Ai.git`
2. **Ouvrir le dossier dans VS Code** et accepter les extensions recommandées (Python, Jupyter, Colab).
3. **Drive :** le dossier `AutoMaze_Ai` vous est partagé. Dans Google Drive, faites un clic droit dessus,
   puis *Organiser > Ajouter un raccourci > Mon Drive*. Le chemin `MyDrive/AutoMaze_Ai` doit exister chez chacun.
4. **Colab :** ouvrez `notebooks/00_verification_colab.ipynb`, puis *Select Kernel > Colab > New Colab Server*
   et connectez-vous avec votre compte Google.
5. **Exécutez le notebook 00.** Toutes les cellules doivent afficher `OK`.

Pour travailler en local sans Colab : `pip install -r requirements.txt`. Attention, les données sont alors écrites
dans `data/` sur votre PC et non dans le Drive.

## Règles de travail

- Exécutez tous les notebooks sur Colab : CPU pour 01 à 03, GPU T4 pour 04. Ainsi, il n'y a qu'un seul jeu de données.
- Une seule personne modifie un notebook à la fois : faites un `git pull` avant, puis un commit et un push après.
- Gardez un seul serveur Colab par compte et fermez-le après usage : *Ctrl+Shift+P > Colab: Remove Server*.
- Si le message « too many servers » apparaît, fermez les sessions dans *Exécution > Gérer les sessions* sur colab.research.google.com.
- Ne modifiez pas l'ordre de `CLASSES` ni les clés de `models/metadata.json` : le simulateur en dépend.

## Rapport

Le rapport final est rédigé dans un document partagé (lien envoyé à l'équipe).
