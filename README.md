# Lisez-moi : Projet de tableau de bord des observations d'oiseaux au Canada

## Description
Cette application Dash affiche les observations d'oiseaux au Canada sous deux formes :
- Un **graphique temporel** montrant l'évolution du nombre d'observations par année.
- Une **carte interactive** représentant les localisations des observations sur le territoire canadien.

## Fonctionnalités
- **Filtrage des données** : Sélection des espèces d'oiseaux, des groupes d'espèces et de l'année liée aux observations.
- **Visualisation des cartes interactives** : Zoom et navigation sur la carte.
- **Graphique interactif** : Affichage des tendances en fonction de l'espèce sélectionnée.
- **Statistiques** : Affichage de statistiques sur le groupe d'espèces choisi et application d'un test statistique pour déterminer l'aspect significatif des différences de tendances entre le groupe d'espèces choisi et le reste des espèces. Un diagramme Quantile-Quantile et un diagramme en boîte permettent de visualiser la distribution des groupes.  

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
│-- main.py                            # Fichier principal de l'application Dash
data/
│-- data_split_files/                  # Observations - fichiers découpés en raison de leur taille
│ │--data_split_file_1.txt
│ │--data_split_file_2.txt
│ │--data_split_file_3.txt
│ │--data_split_file_4.txt
│ │--data_split_file_5.txt
│ │--data_split_file_6.txt
│ │--data_split_file_7.txt
│ │--data_split_file_8.txt
│-- Birds_by_Report_Group.csv          # Groupes d'oiseaux pour chaque espèce
│-- Birds_by_Report_Group_Unique.csv   # Groupe unique sélectionné pour chaque espèce
│requirements.txt                      # Liste des dépendances Python
│README.md                             # Ce fichier
```

## Données utilisées
Les fichiers dans `data/data_split_files/` contiennent :
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

