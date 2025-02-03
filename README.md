# Lisez-moi : Projet - Bird Species Observation Dashboard

## Description
Cette application Dash affiche les observations d'oiseaux au Canada sous deux formes :
- Un **graphique temporel** montrant l'évolution du nombre d'observations par année.
- Une **carte interactive** représentant les localisations des observations sur le territoire canadien.

## Fonctionnalités
- **Filtrage des données** : Sélection des espèces d'oiseaux.
- **Visualisation interactive** : Zoom et navigation sur la carte.
- **Graphique dynamique** : Mise à jour automatique en fonction de l'espèce sélectionnée.

## Installation
### Prérequis
- Python 3.10+
- pip (gestionnaire de paquets Python)

### Étapes
1. **Cloner le dépôt** (ou télécharger les fichiers) :
   ```sh
   git clone https://github.com/carl-genest/SCI1402-Projet.git
   ```
2. **Créer un environnement virtuel** et l'activer :
   ```sh
   python -m venv dash_env
   source venv/bin/activate  # macOS/Linux
   dash_env\Scripts\activate  # Windows
   ```
3. **Installer les dépendances** :
   ```sh
   pip install -r requirements.txt
   ```
4. **Lancer l'application** :
   ```sh
   python dashboard/main.py
   ```
5. **Accéder à l'application** :
   Ouvrir un navigateur et aller à `http://127.0.0.1:8050/`

## Structure du projet
```
dashboard/
│-- main.py               # Fichier principal de l'application Dash
data/
│-- bbs50-can_naturecounts_filtered_data.txt # Observations
│-- Birds_by_Report_Group.csv # Groupes d'oiseaux
│requirements.txt     # Liste des dépendances Python
│README.md            # Ce fichier
```

## Données utilisées
Le fichier `data/bbs50-can_naturecounts_filtered_data.txt` contient :
- `ScientificName` : Nom scientifique de l'espèce observée
- `CommonName` : Nom commun de l'oiseau observé
- `Order` : Ordre de l'espèce observée
- `DecimalLatitude` / `DecimalLongitude` : Coordonnées de l'observation
- `YearCollected` : Année de l'observation
- `ObservationCount` : Nombre d'individus observés

## Technologies utilisées
- **Dash** pour la création du tableau de bord
- **Plotly** pour les graphiques
- **Pandas** pour la gestion des données

## Auteurs
Développé par Carl Genest.

## Utilisation 
Seule une utilisation dans le cadre du cours SCI1402 de l'Université Téluq est permise.

