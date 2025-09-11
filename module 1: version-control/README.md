# 🔄 Module 1 : Contrôle de version et collaboration (Optionel)

Bienvenue dans le monde du contrôle de version ! Ce module couvre Git et GitHub — essentiels pour collaborer sur des projets d’analyse M/EEG et partager des notebooks/rapports.

## Sommaire
1. [Introduction à Git](#introduction-à-git)
2. [Premiers pas avec GitHub](#premiers-pas-avec-github)
3. [Commandes Git de base](#commandes-git-de-base)
4. [Flux de collaboration](#flux-de-collaboration)

## Introduction à Git

### Qu’est-ce que Git ?
Un système de contrôle de version distribué pour suivre les modifications et travailler à plusieurs sans perdre l’historique.

### Installation
1. Télécharger : https://git-scm.com/downloads
2. Vérifier :
```bash
git --version
```
3. Configurer :
```bash
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

## Premiers pas avec GitHub

### Qu’est-ce que GitHub ?
Une plateforme d’hébergement Git pour partager le code, ouvrir des issues et collaborer via des pull requests.

### Mise en place
1. Créer un compte : https://github.com
2. (Optionnel) Configurer des clés SSH
3. Créer votre premier dépôt

## Commandes Git de base

### Initialiser / cloner
```bash
git init              # Initialiser un dépôt
git clone <url>       # Cloner un dépôt existant
git status            # État du dépôt
```

### Travailler sur des changements
```bash
git add .             # Stager les changements
git commit -m "msg"   # Commit avec message
git push              # Envoyer vers le remote
git pull              # Récupérer les changements
```

### Branches
```bash
git branch            # Lister les branches
git checkout -b feat  # Créer et basculer sur une branche
git merge branche     # Fusionner dans la branche courante
```

## Flux de collaboration

### Bonnes pratiques
1. Toujours `git pull` avant de commencer
2. Utiliser des branches par fonctionnalité
3. Commits clairs et atomiques
4. Revue de code avant fusion
5. Tenir un README et un `requirements.txt` à jour

### Travailler à plusieurs
1. Forker les dépôts si besoin
2. Ouvrir des pull requests
3. Discuter et réviser le code
4. Résoudre les conflits de fusion

## Projets M/EEG : structure type
```
projet/
├── data/              # Données brutes/derivées (attention au versioning)
├── notebooks/         # Notebooks Jupyter
├── src/               # Code source (prétraitement, features, modèles)
├── tests/             # Tests unitaires éventuels
├── requirements.txt   # Dépendances
└── README.md          # Documentation
```

Note : évitez de versionner des données volumineuses/confidentielles. Utilisez des liens, DVC, ou des scripts de téléchargement.

## Dépannage et ressources
- Docs Git : https://git-scm.com/doc
- Aide GitHub : https://help.github.com
- Pro Git (livre) : https://git-scm.com/book/en/v2

