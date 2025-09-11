# 🧰 Module 3 : Boîte à outils pour l’analyse M/EEG

Bienvenue ! Ce module présente des bibliothèques essentielles pour l’analyse M/EEG et propose des exemples pratiques centrés sur les oscillations (alpha, beta, gamma), la PSD et une mini-pipeline d’apprentissage supervisé.

## Sommaire
1. [Bibliothèques essentielles](#bibliothèques-essentielles)
2. [Manipulation et visualisation](#manipulation-et-visualisation)
3. [Prétraitement et PSD](#prétraitement-et-psd)
4. [Apprentissage automatique](#apprentissage-automatique)
5. [Exemples pratiques](#exemples-pratiques)

## Bibliothèques essentielles
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

## Manipulation et visualisation
```python
# Trame temporelle
fs = 250
t = np.arange(0, 10, 1/fs)
alpha = 2.0 * np.sin(2*np.pi*10*t)
bruit = 0.5 * np.random.randn(t.size)
x = alpha + bruit

# Visualisation
plt.figure()
plt.plot(t, x)
plt.xlabel('Temps (s)'); plt.ylabel('Amplitude (µV)'); plt.title('EEG synthétique ~10 Hz')
plt.show()
```

## Prétraitement et PSD
```python
# PSD (Welch)
f, pxx = signal.welch(x, fs=fs, nperseg=fs*2)

# Filtrage passe-bande 8–12 Hz
sos = signal.butter(4, [8, 12], btype='bandpass', fs=fs, output='sos')
x_filt = signal.sosfiltfilt(sos, x)

plt.figure(figsize=(10,4))
plt.semilogy(f, pxx)
plt.xlim(1, 40); plt.xlabel('Fréquence (Hz)'); plt.ylabel('PSD (V²/Hz)'); plt.title('PSD (Welch)')
plt.show()
```

## Apprentissage automatique
Exemple d’objectif : distinguer deux conditions expérimentales à partir de la puissance alpha (8–12 Hz).

Pipeline type :
```python
X = features  # ex: puissance alpha par essai
y = labels    # 0/1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = ('scaler + logistic regression')
```

Voir le script d’exemple pour une implémentation complète.

## Exemples pratiques

1. `meeg_synthetic_psd.py`
   - Crée un `Raw` MNE à partir d’un signal synthétique
   - Compare signal brut vs filtré 8–12 Hz
   - Calcule la PSD et visualise un spectrogramme
   - Exécution : `python meeg_synthetic_psd.py`

2. `meeg_ml_pipeline.py`
   - Génère des essais synthétiques (différence de puissance alpha entre 2 classes)
   - Extrait la puissance de bande (Welch) par essai et entraîne une régression logistique
   - Affiche l’évaluation et un graphique des caractéristiques
   - Exécution : `python meeg_ml_pipeline.py`

3. `meeg_time_frequency.py`
   - Démo temps–fréquence (STFT) sur un signal à amplitude alpha modulée
   - Exécution : `python meeg_time_frequency.py`

Avant d’exécuter, installez les dépendances :
```bash
pip install -r requirements.txt
```

## Bonnes pratiques
- Documenter le prétraitement (filtres, notch, ICA si utilisé)
- Utiliser des époques bien définies (événements, fenêtres)
- Définir a priori les bandes d’intérêt (delta, thêta, alpha, bêta, gamma)
- Séparer entraînement/test sans fuite d’information
- Versionner les scripts et conserver les paramètres d’analyse

## Ressources
- MNE-Python : https://mne.tools/stable/index.html
- SciPy Signal : https://docs.scipy.org/doc/scipy/tutorial/signal.html
- scikit-learn : https://scikit-learn.org/stable/

