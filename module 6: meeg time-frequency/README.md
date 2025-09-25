# 📈 Module 6 : Temps-fréquence et potentiels évoqués

Adaptation scriptée de l'AP3 : ce module regroupe les analyses d'oscillations, la segmentation en `Epochs` et le calcul de réponses évoquées à partir des données nettoyées.

## Contenu
- `meeg_time_frequency.py` — script commenté couvrant la construction des epochs, la moyenne conditionnelle, la PSD et les cartes temps-fréquence.

## Objectifs clés
- Extraire et documenter les évènements comportementaux liés à un enregistrement M/EEG.
- Comparer potentiels évoqués et signatures oscillatoires entre conditions.
- Introduire les outils temps-fréquence de MNE (Morlet, TFR, topographies dynamiques).

## Idées d'activités
- Rejouer les étapes sur d'autres paires de conditions (`event_id`) pour tester des hypothèses alternatives.
- Faire implémenter aux étudiant·e·s leurs propres métriques de bande (puissance beta, gamma, etc.).
