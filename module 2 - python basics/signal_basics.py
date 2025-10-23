"""
Notions de base pour signaux M/EEG (script autonome)

Contenu :
- Génération d'un signal synthétique avec une composante alpha (~10 Hz)
- Estimation PSD (Welch)
- Filtre passe-bande 8–12 Hz
- Spectrogramme (STFT)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def generer_signal(fs: int = 250, duree_s: float = 10.0, f_alpha: float = 10.0,
                   amp_alpha: float = 2.0, bruit_ecart_type: float = 0.5):
    """Génère un signal EEG synthétique : alpha + bruit blanc.

    Paramètres
    ---------
    fs : int
        Fréquence d'échantillonnage (Hz).
    duree_s : float
        Durée du signal (secondes).
    f_alpha : float
        Fréquence centrale de la bande alpha (Hz).
    amp_alpha : float
        Amplitude du sinus alpha.
    bruit_ecart_type : float
        Écart-type du bruit gaussien.
    """
    t = np.arange(0, duree_s, 1 / fs)
    alpha = amp_alpha * np.sin(2 * np.pi * f_alpha * t)
    bruit = bruit_ecart_type * np.random.randn(t.size)
    x = alpha + bruit
    return t, x


def psd_welch(x: np.ndarray, fs: int):
    f, pxx = signal.welch(x, fs=fs, nperseg=fs * 2)
    return f, pxx


def filtrer_bande(x: np.ndarray, fs: int, fmin: float = 8.0, fmax: float = 12.0):
    sos = signal.butter(4, [fmin, fmax], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)


def spectrogramme_stft(x: np.ndarray, fs: int):
    f, t, Sxx = signal.stft(x, fs=fs, nperseg=int(0.5 * fs), noverlap=int(0.25 * fs))
    return f, t, np.abs(Sxx) ** 2


def main():
    fs = 250
    t, x = generer_signal(fs=fs)

    # PSD brute
    f, pxx = psd_welch(x, fs)

    # Filtrage 8–12 Hz
    x_filt = filtrer_bande(x, fs, 8, 12)
    f_filt, pxx_filt = psd_welch(x_filt, fs)

    # STFT
    f_s, t_s, Sxx = spectrogramme_stft(x, fs)

    # Tracés
    plt.figure(figsize=(14, 8))

    # Signal brut et filtré
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, x, label='Brut', alpha=0.7)
    ax1.plot(t, x_filt, label='Filtré 8–12 Hz', alpha=0.8)
    ax1.set_title('Signal temporel')
    ax1.set_xlabel('Temps (s)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.legend()

    # PSD brute
    ax2 = plt.subplot(2, 2, 2)
    ax2.semilogy(f, pxx, label='Brut')
    ax2.semilogy(f_filt, pxx_filt, label='Filtré')
    ax2.set_xlim(1, 40)
    ax2.set_title('PSD (Welch)')
    ax2.set_xlabel('Fréquence (Hz)'); ax2.set_ylabel('PSD (V²/Hz)')
    ax2.legend()

    # Zoom temporel 2 s
    ax3 = plt.subplot(2, 2, 3)
    n_zoom = int(2 * fs)
    ax3.plot(t[:n_zoom], x[:n_zoom], label='Brut', alpha=0.7)
    ax3.plot(t[:n_zoom], x_filt[:n_zoom], label='Filtré', alpha=0.8)
    ax3.set_title('Zoom 2 s')
    ax3.set_xlabel('Temps (s)'); ax3.set_ylabel('Amplitude (µV)')
    ax3.legend()

    # Spectrogramme (STFT)
    ax4 = plt.subplot(2, 2, 4)
    im = ax4.pcolormesh(t_s, f_s, Sxx, shading='auto')
    ax4.set_ylim(1, 40)
    ax4.set_title('Spectrogramme (STFT)')
    ax4.set_xlabel('Temps (s)'); ax4.set_ylabel('Fréquence (Hz)')
    plt.colorbar(im, ax=ax4, label='Puissance')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

