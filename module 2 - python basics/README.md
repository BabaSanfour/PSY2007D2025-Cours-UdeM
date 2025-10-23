# 🐍 Module 2 : Python pour les neurosciences (M/EEG)

Bienvenue ! Ce module couvre les bases de Python en les reliant à l’analyse de signaux M/EEG : séries temporelles, filtrage, densité spectrale de puissance (DSP/PSD) et notebooks.

## Sommaire
1. [Bases de Python](#bases-de-python)
2. [Structures de données](#structures-de-données)
3. [Fonctions et classes](#fonctions-et-classes)
4. [Travailler avec les données](#travailler-avec-les-données)
5. [Notebooks Jupyter](#notebooks-jupyter)
6. [Exercice M/EEG](#exercice-meeg)

## Bases de Python

### Syntaxe de base
```python
nom = "Étudiant"  # chaîne
age = 22           # entier
taille = 1.72      # flottant
est_etudiant = True

print(f"Bonjour {nom} !")
```

### Contrôle de flux
```python
if age >= 18:
    print("Majeur")
else:
    print("Mineur")

for i in range(3):
    print(i)
```

## Structures de données
```python
# Liste
valeurs = [1, 2, 3]
valeurs.append(4)

# Dictionnaire
infos = {"nom": "EEG", "fs": 250}

# Tuple (immuable)
dimensions = (64, 1000)  # 64 canaux, 1000 échantillons
```

## Fonctions et classes
```python
def puissance_bande(freqs, psd, fmin, fmax):
    """Somme de la PSD entre fmin et fmax."""
    import numpy as np
    masque = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[masque], freqs[masque])
```

## Travailler avec les données
```python
import pandas as pd
df = pd.read_csv('donnees.csv')
df.head()
```

## Notebooks Jupyter
```bash
pip install jupyter
jupyter notebook
```

## Votre premier fichier Python
Créez `hello_fr.py` :
```python
print("Bienvenue au cours M/EEG et oscillations !")
```
Exécutez :
```bash
python hello_fr.py
```

## Exercice M/EEG

Objectif : générer un signal EEG synthétique avec une composante alpha (≈10 Hz), estimer sa PSD (Welch), appliquer un filtre passe-bande 8–12 Hz et visualiser un spectrogramme court.

- Script exemple : `signal_basics.py`
- Lancer : `python signal_basics.py`

Prolongez l’exercice :
- Modifiez l’amplitude du 10 Hz et observez la PSD
- Ajoutez une raie à 50/60 Hz et utilisez un notch
- Comparez STFT et ondelettes (si vous utilisez MNE)

## Ressources
- Python : https://docs.python.org/
- NumPy : https://numpy.org/doc/
- SciPy : https://docs.scipy.org/doc/scipy/
- MNE-Python : https://mne.tools/stable/index.html

