# Kit de démarrage — Cours M/EEG et oscillations 🚀

## Bienvenue !

Ce kit vous accompagne pour la partie programmation du cours consacré aux données M/EEG (MEG/EEG) et aux oscillations neuronales.

## Contenu du kit

### 🛠️ Module 0 : Outils essentiels
- Terminal et environnements Python
- VS Code et Jupyter
- Démarrage rapide sur Google Colab
- Paquets utiles pour M/EEG (NumPy, SciPy, MNE, Matplotlib, scikit-learn)

### 🔄 Module 1 : Contrôle de version et collaboration (Optionel)
- Git et GitHub
- Dépôts, branches et pull requests
- Bonnes pratiques pour projets d’analyse M/EEG

### 🐍 Module 2 : Python pour les neurosciences
- Bases Python (types, boucles, fonctions)
- Manipulation de données et notebooks
- Signaux 1D : génération, filtrage, DSP (PSD)

### 🧰 Module 3 : Boîte à outils M/EEG
- Bibliothèques clés : NumPy, SciPy, MNE, Matplotlib, scikit-learn
- Prétraitement et caractéristiques (bandes de fréquences)
- Démo ML simple sur des caractéristiques oscillatoires
- Exemples pratiques (scripts prêts à exécuter)

## Par où commencer
1. Commencez par le Module 0 pour préparer l’environnement.
2. Parcourez les modules dans l’ordre.
3. Testez les scripts d’exemples et modifiez-les pour expérimenter.

## Séance 1 — Notebook Colab
- [Lien du notebook](https://colab.research.google.com/drive/1LKqnxEM3DMZoxsXgnRDSlC-tvCrBSdcC?usp=sharing)
- Couvre le Module 2 (bases Python, , bibliotheques, notebooks).
- Conseil: ouvrez-le dans Colab, faites une copie dans votre Drive, puis exécutez cellule par cellule en lisant les explications.

## Séance 2 — Fichier Notebook 
- [Lien du notebook](session2.ipynb)
- Couvre le Module 4 et 5 (Base de MNE et netoyage de données).

## Installer localement (clone ou fork) et utiliser avec Colab

### Option A — Fork puis clone (si vous comptez contribuer)
1. Forkez le dépôt sur GitHub (depuis l’interface du dépôt d’origine).
2. Clonez votre fork en local:
   ```bash
   git clone https://github.com/<votre-compte>/<nom-du-repo>.git
   cd <nom-du-repo>
   ```
3. Créez un environnement et installez les dépendances:
   ```bash
   python -m venv env_meeg
   # macOS/Linux
   source env_meeg/bin/activate
   # Windows
   # env_meeg\Scripts\activate

   pip install -r "module 3 - python for m_eeg/requirements.txt"
   ```
4. (Optionnel) Installez Jupyter et lancez-le:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

### Option B — Clone direct (lecture seule)
```bash
git clone https://github.com/BabaSanfour/PSY2007D2025-Cours-UdeM.git
cd PSY2007D2025-Cours-UdeM
python -m venv env_meeg
source env_meeg/bin/activate  # ou env_meeg\Scripts\activate (Windows)
pip install -r "module 3 - python for m_eeg/requirements.txt"
```

### Utiliser ce dépôt dans Google Colab
- Ouvrez un nouveau notebook Colab, puis clonez le dépôt:
  ```python
  !git clone https://github.com/BabaSanfour/PSY2007D2025-Cours-UdeM.git
  %cd PSY2007D2025-Cours-UdeM
  !pip install -r "module 3 - python for m_eeg/requirements.txt"
  ```
- Exécutez les scripts d’exemple directement dans Colab:
  ```python
  !python "module 3 - python for m_eeg/meeg_synthetic_psd.py"
  ```
- (Optionnel) Montez votre Google Drive pour sauvegarder/charger des fichiers:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

## Prérequis
- Un ordinateur avec accès à Internet
- Envie d’apprendre et d’explorer les données M/EEG ✨

## Besoin d’aide ?
- Chaque module contient des instructions et des exemples détaillés.
- Demandez conseil à l’équipe enseignante si besoin.

## Diapositives
- Mini-ensemble de diapositives : « Introduction à la programmation en Python ».
- Contenu : C’est quoi la programmation ? Pourquoi Python ? Configuration de l’environnement (VS Code, Colab).
- Conseil d’usage : survoler avant le Module 0, garder ouvert pendant la mise en place, réviser ensuite.

## Bon apprentissage ! 🌟

Rappel : la pratique régulière est la clé. Concentrez-vous sur les concepts M/EEG (prétraitement, oscillations, analyse temps-fréquence) tout en consolidant vos bases Python.
