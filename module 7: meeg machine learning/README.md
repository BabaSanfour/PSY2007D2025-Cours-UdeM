# 🤖 Module 7 : Apprentissage automatique pour l'EEG/MEG

Ce module transpose l'AP4 en script afin d'illustrer un pipeline de classification supervisée, de l'extraction de caractéristiques à l'évaluation du modèle.

## Contenu
- `meeg_machine_learning.py` — script complet mêlant préparation des features, validation croisée et visualisations des performances.

## Objectifs clés
- Construire un pipeline scikit-learn sur des données M/EEG prétraitées.
- Explorer différentes représentations fréquentielles/temps-fréquence comme entrées du modèle.
- Discuter des bonnes pratiques de validation (séparation train/test, équilibrage, métriques).

## Ateliers suggérés
1. Tester d'autres estimateurs (SVM, RandomForest) et comparer les scores.
2. Introduire une étape de sélection de features ou de réduction de dimension (PCA) avant la classification.
3. Simuler un déséquilibre de classes et analyser l'impact sur les métriques.
