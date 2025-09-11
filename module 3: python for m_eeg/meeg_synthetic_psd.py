"""
M/EEG — Démo PSD et filtrage avec MNE sur un signal synthétique.

Étapes :
- Génère un EEG 1-canal avec composante alpha (~10 Hz)
- Crée un Raw MNE
- Compare signal brut vs filtré (8–12 Hz)
- Calcule et trace la PSD
- Affiche un spectrogramme (STFT)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne


def creer_raw_synthetique(fs=250, duree_s=10.0, f_alpha=10.0, amp_alpha=2.0, bruit_sigma=0.5):
    t = np.arange(0, duree_s, 1/fs)
    alpha = amp_alpha * np.sin(2*np.pi*f_alpha*t)
    bruit = bruit_sigma * np.random.randn(t.size)
    x = alpha + bruit

    data = x[np.newaxis, :]  # (n_channels, n_times)
    info = mne.create_info(ch_names=['EEG001'], sfreq=fs, ch_types=['eeg'])
    raw = mne.io.RawArray(data, info, verbose='ERROR')
    return t, x, raw


def psd_welch_array(x: np.ndarray, fs: int):
    """Retourne freqs, psd pour un array 1D."""
    freqs, psd = signal.welch(x, fs=fs, nperseg=fs*2)
    return freqs, psd


def main():
    fs = 250
    t, x, raw = creer_raw_synthetique(fs=fs)

    # Filtrage 8–12 Hz
    raw_filt = raw.copy().filter(8., 12., fir_design='firwin', verbose='ERROR')

    # Récupère les données filtrées
    x_filt = raw_filt.get_data(picks='eeg')[0]

    # PSD
    f, pxx = psd_welch_array(x, fs)
    f_filt, pxx_filt = psd_welch_array(x_filt, fs)

    # Spectrogramme
    f_s, t_s, Sxx = signal.stft(x, fs=fs, nperseg=int(0.5 * fs), noverlap=int(0.25 * fs))

    # Figures
    plt.figure(figsize=(14, 8))

    # Signal temporel (brut vs filtré)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, x, label='Brut', alpha=0.7)
    ax1.plot(t, x_filt, label='Filtré 8–12 Hz', alpha=0.9)
    ax1.set_title('Signal EEG (1 canal)')
    ax1.set_xlabel('Temps (s)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.legend()

    # PSD
    ax2 = plt.subplot(2, 2, 2)
    ax2.semilogy(f, pxx, label='Brut')
    ax2.semilogy(f_filt, pxx_filt, label='Filtré')
    ax2.set_xlim(1, 40)
    ax2.set_title('PSD (Welch)')
    ax2.set_xlabel('Fréquence (Hz)'); ax2.set_ylabel('PSD (V²/Hz)')
    ax2.legend()

    # Zoom temporel (2 s)
    ax3 = plt.subplot(2, 2, 3)
    n_zoom = int(2 * fs)
    ax3.plot(t[:n_zoom], x[:n_zoom], label='Brut', alpha=0.7)
    ax3.plot(t[:n_zoom], x_filt[:n_zoom], label='Filtré', alpha=0.9)
    ax3.set_title('Zoom 2 s')
    ax3.set_xlabel('Temps (s)'); ax3.set_ylabel('Amplitude (µV)')
    ax3.legend()

    # Spectrogramme
    ax4 = plt.subplot(2, 2, 4)
    im = ax4.pcolormesh(t_s, f_s, np.abs(Sxx)**2, shading='auto')
    ax4.set_ylim(1, 40)
    ax4.set_title('Spectrogramme (STFT)')
    ax4.set_xlabel('Temps (s)'); ax4.set_ylabel('Fréquence (Hz)')
    plt.colorbar(im, ax=ax4, label='Puissance')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

