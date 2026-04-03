#!/usr/bin/env python3
"""
Deep multi-dimensional comparison of sampled Rhodes vs FM Rhodes.
Analyzes: spectral centroid, spectral rolloff, MFCCs, attack transient shape,
harmonic-to-noise ratio, spectral flux, and temporal envelope.
"""

import numpy as np
import librosa
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def load_audio(path):
    y, sr = librosa.load(path, sr=44100, mono=True)
    return y, sr


def analyze(y, sr, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    duration = len(y) / sr

    # --- 1. Temporal envelope shape ---
    print(f"\n  [ENVELOPE]")
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)

    peak_idx = np.argmax(rms)
    peak_time = times[peak_idx]
    peak_val = rms[peak_idx]

    # Attack shape: measure RMS at specific early time points
    for ms in [1, 5, 10, 20, 50]:
        t_idx = np.argmin(np.abs(times - ms/1000))
        if t_idx < len(rms):
            pct = rms[t_idx] / peak_val * 100
            print(f"    At {ms:3d}ms: {pct:5.1f}% of peak")

    # Decay shape
    for pct_target in [75, 50, 25, 10]:
        idx = next((i for i in range(peak_idx, len(rms)) if rms[i] < peak_val * pct_target/100), len(rms)-1)
        print(f"    Decay to {pct_target:2d}%: {times[idx]:.3f}s")

    # --- 2. Spectral centroid over time (brightness tracker) ---
    print(f"\n  [SPECTRAL CENTROID] (higher = brighter)")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    c_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=512)
    for t_sec, t_label in [(0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (0.5, "500ms"), (1.0, "1s"), (2.0, "2s")]:
        idx = np.argmin(np.abs(c_times - t_sec))
        if idx < len(centroid):
            print(f"    {t_label:>5s}: {centroid[idx]:7.0f} Hz")

    # --- 3. Spectral rolloff (where 85% of energy is below) ---
    print(f"\n  [SPECTRAL ROLLOFF 85%] (high-frequency extent)")
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, hop_length=512)[0]
    for t_sec, t_label in [(0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (1.0, "1s")]:
        idx = np.argmin(np.abs(c_times - t_sec))
        if idx < len(rolloff):
            print(f"    {t_label:>5s}: {rolloff[idx]:7.0f} Hz")

    # --- 4. Spectral flatness (noise-like vs tonal) ---
    # Higher = more noise-like, lower = more tonal/harmonic
    print(f"\n  [SPECTRAL FLATNESS] (0=pure tone, 1=white noise)")
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
    for t_sec, t_label in [(0.005, "5ms"), (0.01, "10ms"), (0.03, "30ms"), (0.1, "100ms"), (0.5, "500ms"), (1.0, "1s")]:
        idx = np.argmin(np.abs(c_times - t_sec))
        if idx < len(flatness):
            print(f"    {t_label:>5s}: {flatness[idx]:.6f}")

    # --- 5. MFCCs (timbral fingerprint) ---
    print(f"\n  [MFCC MEANS] (timbral shape, first 0.5s)")
    y_short = y[:int(0.5 * sr)]
    mfccs = librosa.feature.mfcc(y=y_short, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)
    for i, val in enumerate(mfcc_means):
        bar = '#' * max(1, int(abs(val) / 5))
        sign = '+' if val >= 0 else '-'
        print(f"    MFCC {i:2d}: {val:+8.1f}  {bar}")

    # --- 6. Spectral bandwidth (spread of energy) ---
    print(f"\n  [SPECTRAL BANDWIDTH] (spread around centroid)")
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
    for t_sec, t_label in [(0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (1.0, "1s")]:
        idx = np.argmin(np.abs(c_times - t_sec))
        if idx < len(bandwidth):
            print(f"    {t_label:>5s}: {bandwidth[idx]:7.0f} Hz")

    # --- 7. Zero crossing rate (rough texture measure) ---
    print(f"\n  [ZERO CROSSING RATE] (higher = more high-freq content/noise)")
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=256)[0]
    z_times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=256)
    for t_sec, t_label in [(0.005, "5ms"), (0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (1.0, "1s")]:
        idx = np.argmin(np.abs(z_times - t_sec))
        if idx < len(zcr):
            print(f"    {t_label:>5s}: {zcr[idx]:.4f}")

    # --- 8. Harmonic vs percussive energy ---
    print(f"\n  [HARMONIC/PERCUSSIVE SPLIT]")
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm**2)
    perc_energy = np.sum(y_perc**2)
    total = harm_energy + perc_energy
    print(f"    Harmonic:   {harm_energy/total*100:.1f}%")
    print(f"    Percussive: {perc_energy/total*100:.1f}%")

    # Percussive energy in first 50ms vs total percussive
    cutoff = int(0.05 * sr)
    perc_attack = np.sum(y_perc[:cutoff]**2)
    print(f"    Percussive in first 50ms: {perc_attack/perc_energy*100:.1f}% of all percussive")

    # --- 9. Attack transient spectral snapshot (first 20ms) ---
    print(f"\n  [ATTACK SPECTRUM] (first 20ms, top peaks)")
    attack_y = y[:int(0.02 * sr)]
    attack_y = attack_y * np.hanning(len(attack_y))
    fft = np.abs(np.fft.rfft(attack_y))
    freqs = np.fft.rfftfreq(len(attack_y), 1.0/sr)
    # Find peaks
    peaks = []
    for i in range(3, len(fft)-3):
        if fft[i] > fft[i-1] and fft[i] > fft[i+1] and fft[i] > fft[i-2] and freqs[i] > 50:
            peaks.append((fft[i], freqs[i]))
    peaks.sort(reverse=True)
    if peaks:
        max_a = peaks[0][0]
        for amp, freq in sorted(peaks[:10], key=lambda x: x[1]):
            db = 20 * np.log10(amp/max_a + 1e-10)
            print(f"    {freq:7.0f} Hz  {db:+5.1f} dB")

    return {
        'mfcc_means': mfcc_means,
        'centroid_early': centroid[np.argmin(np.abs(c_times - 0.05))] if len(centroid) > 0 else 0,
        'centroid_late': centroid[np.argmin(np.abs(c_times - 1.0))] if len(centroid) > 0 else 0,
        'harm_pct': harm_energy/total*100,
        'perc_pct': perc_energy/total*100,
    }


def compare_mfccs(stats_a, stats_b, label_a, label_b):
    print(f"\n{'='*70}")
    print(f"  MFCC DISTANCE: {label_a} vs {label_b}")
    print(f"{'='*70}")
    diff = stats_a['mfcc_means'] - stats_b['mfcc_means']
    total_dist = np.sqrt(np.sum(diff**2))
    print(f"  Euclidean distance: {total_dist:.1f}")
    print(f"  Per-coefficient difference:")
    for i, d in enumerate(diff):
        bar = '#' * max(1, int(abs(d) / 3))
        direction = '→ FM brighter' if d < 0 else '→ FM duller'
        if i == 0:
            direction = '→ FM louder' if d < 0 else '→ FM quieter'
        print(f"    MFCC {i:2d}: {d:+7.1f}  {bar}  {direction if abs(d) > 5 else ''}")


def main():
    notes = [('D3', 50), ('D4', 62), ('B4', 71)]

    for note_name, midi in notes:
        sampled_path = os.path.join(BASE, 'audio', 'rhodes', f'{note_name}.mp3')
        fm_path = os.path.join(BASE, 'audio', 'rhodes-fm', f'{note_name}.mp3')

        if not os.path.exists(sampled_path) or not os.path.exists(fm_path):
            print(f"Skipping {note_name}")
            continue

        print(f"\n{'#'*70}")
        print(f"  NOTE: {note_name} (MIDI {midi})")
        print(f"{'#'*70}")

        y_s, sr = load_audio(sampled_path)
        y_f, sr = load_audio(fm_path)

        stats_s = analyze(y_s, sr, f"SAMPLED Rhodes — {note_name}")
        stats_f = analyze(y_f, sr, f"FM Rhodes — {note_name}")
        compare_mfccs(stats_s, stats_f, "Sampled", "FM")

        print(f"\n{'~'*70}")


if __name__ == '__main__':
    main()
