# 🧼 Module 5 : Prétraitement M/EEG

Ce module convertit l'AP2 en script Python et détaille un pipeline complet de nettoyage : filtrage, ICA et AutoReject appliqués au dataset de démonstration de MNE.

## Contenu
- `meeg_preprocessing.py` — script issu du notebook AP2, comprenant les questions guidées sous forme de commentaires ou de placeholders `...` à compléter en classe.

## Objectifs clés
- Comprendre l'impact des filtres notch, passe-bas et passe-haut sur les capteurs MEG / EEG.
- Configurer et interpréter une décomposition ICA pour retirer les artefacts oculaires et cardiaques.
- Utiliser AutoReject pour estimer automatiquement des seuils de rejet par canal.

## Conseils pédagogiques
- Invitez les étudiant·e·s à comparer visuellement les données avant/après chaque étape.
- Les sections marquées `...` servent de points d'arrêt pour coder en direct ou proposer des exercices.
