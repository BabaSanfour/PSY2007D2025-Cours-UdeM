"""
Démo temps–fréquence (STFT) sur un signal à amplitude alpha modulée.

Contenu :
- Générer un signal avec une fenêtre d'augmentation d'amplitude alpha
- Calculer un spectrogramme (STFT)
- Visualiser dynamique temps–fréquence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def signal_alpha_module(fs=250, duree=10.0, f_alpha=10.0):
    t = np.arange(0, duree, 1/fs)
    # Amplitude modulée : faible au début, forte au milieu, faible à la fin
    envelope = np.ones_like(t) * 0.8
    envelope[(t >= 3) & (t <= 7)] = 2.0
    x = envelope * np.sin(2*np.pi*f_alpha*t) + 0.4*np.random.randn(t.size)
    return t, x, envelope


def main():
    fs = 250
    t, x, envelope = signal_alpha_module(fs=fs)

    f, tt, Sxx = signal.stft(x, fs=fs, nperseg=int(0.5*fs), noverlap=int(0.25*fs))
    power = np.abs(Sxx)**2

    plt.figure(figsize=(12, 6))

    # Signal + enveloppe
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, x, label='Signal')
    ax1.plot(t, envelope, 'r--', label='Enveloppe alpha (indicative)')
    ax1.set_title('Signal à amplitude alpha modulée')
    ax1.set_xlabel('Temps (s)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.legend()

    # Spectrogramme
    ax2 = plt.subplot(2,1,2)
    im = ax2.pcolormesh(tt, f, power, shading='auto')
    ax2.set_ylim(1, 40)
    ax2.set_xlabel('Temps (s)'); ax2.set_ylabel('Fréquence (Hz)')
    ax2.set_title('Spectrogramme (STFT)')
    plt.colorbar(im, ax=ax2, label='Puissance')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

