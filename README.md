# Modèle de Prédiction de Survie pour Patients Atteints de Leucémie

Ce projet implémente un modèle d'ensemble pour la prédiction du risque de survie chez les patients atteints de leucémie, en utilisant des données cliniques, cytogénétiques et moléculaires.

## Structure du Projet

- `main.py` : Script principal qui orchestre l'ensemble du processus
- `utils.py` : Fonctions utilitaires pour le chargement et le prétraitement des données
- `feature_engineering.py` : Création de features expertes à partir des données brutes
- `models.py` : Implémentation des modèles de survie et de l'ensemble
- `feature_selection.py` : Définition et sélection des ensembles de features
- `visualization.py` : Fonctions pour visualiser les résultats

## Prérequis

Voir le fichier `requirements.txt` pour les dépendances.

## Utilisation

1. Placez les fichiers de données dans le répertoire courant:
   - `clinical_train.csv`
   - `clinical_test.csv`
   - `molecular_train.csv`
   - `molecular_test.csv`
   - `target_train.csv`

2. Exécutez le script principal:
   ```
   python main.py
   ```

3. Les résultats seront générés dans le répertoire courant:
   - `submission_ensemble.csv` : Prédictions finales
   - `cv_results.png` : Graphique des résultats de validation croisée
   - `risk_distribution.png` : Distribution des scores de risque

## Approche

Le modèle utilise une approche d'ensemble combinant:
- Modèle Cox proportional hazards
- Random Survival Forest
- Gradient Boosting Survival Analysis

Les features sont créées à partir de connaissances expertes en hématologie, incluant:
- Analyse cytogénétique selon les guidelines ELN 2022
- Profil mutationnel avec classification pronostique des gènes
- Biomarqueurs cliniques avancés