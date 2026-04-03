#!/usr/bin/env python3
"""Compare the sampled Rhodes vs FM Rhodes to identify spectral differences."""

import numpy as np
import subprocess
import tempfile
import wave
import os

SAMPLE_RATE = 44100

def load_mp3_as_numpy(mp3_path):
    """Convert MP3 to WAV in memory, return numpy array."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        'ffmpeg', '-y', '-i', mp3_path, '-ar', str(SAMPLE_RATE),
        '-ac', '1', '-f', 'wav', tmp_path
    ], capture_output=True)
    with wave.open(tmp_path, 'r') as wf:
        frames = wf.readframes(wf.getnframes())
        signal = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    os.remove(tmp_path)
    return signal


def analyze_note(signal, label, note_name):
    """Analyze a note's spectral and temporal characteristics."""
    print(f"\n{'='*60}")
    print(f"  {label} — {note_name}")
    print(f"{'='*60}")

    # Trim to 4 seconds max
    n = min(len(signal), SAMPLE_RATE * 4)
    signal = signal[:n]

    # --- Amplitude envelope ---
    # Compute RMS in 10ms windows
    win = int(0.01 * SAMPLE_RATE)
    rms = []
    for i in range(0, n - win, win):
        rms.append(np.sqrt(np.mean(signal[i:i+win]**2)))
    rms = np.array(rms)

    peak_idx = np.argmax(rms)
    peak_time = peak_idx * 0.01
    peak_val = rms[peak_idx]

    # Find decay to 50% and 10%
    half_idx = next((i for i in range(peak_idx, len(rms)) if rms[i] < peak_val * 0.5), len(rms))
    tenth_idx = next((i for i in range(peak_idx, len(rms)) if rms[i] < peak_val * 0.1), len(rms))

    print(f"  Peak time:     {peak_time:.3f}s")
    print(f"  Decay to 50%:  {half_idx * 0.01:.3f}s  ({(half_idx - peak_idx) * 10}ms after peak)")
    print(f"  Decay to 10%:  {tenth_idx * 0.01:.3f}s  ({(tenth_idx - peak_idx) * 10}ms after peak)")

    # --- Spectral analysis at different time points ---
    for t_start, t_label in [(0.01, "Attack (10ms)"), (0.05, "Early (50ms)"),
                              (0.2, "Body (200ms)"), (1.0, "Sustain (1s)")]:
        start = int(t_start * SAMPLE_RATE)
        end = min(start + int(0.05 * SAMPLE_RATE), n)  # 50ms window
        if start >= n:
            continue

        chunk = signal[start:end]
        if len(chunk) < 256:
            continue

        # Apply window
        chunk = chunk * np.hanning(len(chunk))

        # FFT
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)

        # Find top 8 peaks
        # Smooth to avoid noise peaks
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(fft, 5)

        # Find local maxima
        peaks = []
        for i in range(2, len(smooth) - 2):
            if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > smooth[i-2] and smooth[i] > smooth[i+2]:
                if freqs[i] > 30:  # skip DC
                    peaks.append((smooth[i], freqs[i]))

        peaks.sort(reverse=True)
        top = peaks[:8]

        print(f"\n  {t_label}:")
        if not top:
            print(f"    (no significant peaks)")
            continue

        max_amp = top[0][0]
        for amp, freq in sorted(top, key=lambda x: x[1]):
            rel_db = 20 * np.log10(amp / max_amp + 1e-10)
            bar = '#' * max(1, int((amp / max_amp) * 30))
            print(f"    {freq:7.1f} Hz  {rel_db:+5.1f} dB  {bar}")

    # --- Overall harmonic content ---
    # Use first 0.5s for harmonic analysis
    chunk = signal[:min(int(0.5 * SAMPLE_RATE), n)]
    chunk = chunk * np.hanning(len(chunk))
    fft = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)

    # Energy in bands
    total = np.sum(fft**2)
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 20000)]
    print(f"\n  Energy distribution (first 0.5s):")
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energy = np.sum(fft[mask]**2) / total * 100
        bar = '#' * max(1, int(energy / 2))
        print(f"    {lo:5d}-{hi:5d} Hz:  {energy:5.1f}%  {bar}")


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    # Compare a few notes across the range
    notes = [
        ('D3', 50),
        ('D4', 62),
        ('B4', 71),
    ]

    for note_name, midi in notes:
        sampled_path = os.path.join(base, 'audio', 'rhodes', f'{note_name}.mp3')
        fm_path = os.path.join(base, 'audio', 'rhodes-fm', f'{note_name}.mp3')

        if not os.path.exists(sampled_path):
            print(f"Skipping {note_name} — sampled file not found")
            continue
        if not os.path.exists(fm_path):
            print(f"Skipping {note_name} — FM file not found")
            continue

        sampled = load_mp3_as_numpy(sampled_path)
        fm = load_mp3_as_numpy(fm_path)

        analyze_note(sampled, "SAMPLED Rhodes", note_name)
        analyze_note(fm, "FM Rhodes", note_name)

        print(f"\n{'~'*60}")


if __name__ == '__main__':
    main()
