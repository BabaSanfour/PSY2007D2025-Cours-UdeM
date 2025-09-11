# 🛠️ Module 0 : Outils essentiels

Bienvenue ! Ce module vous aide à préparer un environnement de travail adapté à l’analyse de données M/EEG (EEG/MEG) et aux exercices de programmation du cours.

## Deux façons de démarrer

### Option 1 : Google Colab (débutant recommandé)
Colab est un environnement Jupyter hébergé dans le cloud, sans installation locale.
- Zéro installation, fonctionne dans le navigateur
- Paquets IA/numériques faciles à ajouter
- Sauvegarde automatique

Pour installer des bibliothèques utiles au M/EEG dans un notebook Colab :
```python
!pip install numpy scipy mne matplotlib seaborn scikit-learn
```

### Option 2 : Installation locale (plus de contrôle)
Installez Python et VS Code pour travailler hors ligne et gérer finement votre environnement.

## Sommaire
1. [Préparer l’environnement Python](#préparer-lenvironnement-python)
2. [VS Code](#vs-code)
3. [Terminal: commandes de base](#terminal-commandes-de-base)

## Préparer l’environnement Python

### Python — de quoi s’agit-il ?
Un langage polyvalent, lisible et très utilisé en science des données et en neurosciences. L’écosystème open source (NumPy, SciPy, MNE, etc.) est un atout majeur.

### Installation de Python
1. Téléchargez Python : https://www.python.org/downloads/
2. Cochez « Add Python to PATH » pendant l’installation
3. Vérifiez la version :
```bash
python --version
```

### Environnements virtuels
Isolez les dépendances par projet (fortement recommandé pour l’analyse M/EEG).
```bash
# Créer un environnement
python -m venv env_meeg

# Activer
# Windows
env_meeg\Scripts\activate
# macOS/Linux
source env_meeg/bin/activate

# Désactiver
deactivate
```

Conservez un fichier `requirements.txt` par projet :
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

### Paquets recommandés pour M/EEG
```bash
pip install numpy scipy mne matplotlib seaborn scikit-learn
```

## VS Code
Téléchargez VS Code : https://code.visualstudio.com/

Extensions utiles :
- Python (Microsoft)
- Jupyter
- Pylance
- GitLens

Fonctionnalités clés : terminal intégré, debug, Git, IntelliSense.

## Terminal: commandes de base
Sous Windows, utilisez WSL2 ou Git Bash pour des commandes type Unix.

Exemples :
```bash
pwd        # Répertoire courant
ls         # Lister fichiers
cd         # Changer de dossier
mkdir      # Créer un dossier
python --version
pip list
```

Dans un notebook Jupyter/Colab :
```python
!ls
!pip install mne
```

## Dépannage
- Documentation Python : https://docs.python.org/
- VS Code : https://code.visualstudio.com/docs
- MNE-Python : https://mne.tools/stable/index.html

