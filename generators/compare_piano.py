#!/usr/bin/env python3
"""Compare generated grand piano samples against the original recorded piano samples."""

import numpy as np
import subprocess
import os
import wave
import tempfile


def mp3_to_wav_array(mp3_path):
    """Decode MP3 to numpy array via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        'ffmpeg', '-y', '-i', mp3_path, '-ar', '44100', '-ac', '1', tmp_path
    ], capture_output=True)
    with wave.open(tmp_path, 'r') as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(float)
        data /= 32768.0
    os.remove(tmp_path)
    return data


def analyze_note(signal, sr=44100, label=""):
    """Analyze a piano note's characteristics."""
    n = len(signal)
    duration = n / sr

    # Find peak and measure attack time
    abs_sig = np.abs(signal)
    peak_idx = np.argmax(abs_sig)
    peak_time = peak_idx / sr
    peak_val = abs_sig[peak_idx]

    # Measure amplitude at key time points
    times = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    amp_at = {}
    window = int(sr * 0.02)  # 20ms RMS window
    for t_sec in times:
        idx = int(t_sec * sr)
        if idx + window < n:
            rms = np.sqrt(np.mean(signal[idx:idx+window]**2))
            amp_at[t_sec] = rms

    # Normalize amplitudes to peak RMS
    peak_rms = max(amp_at.values()) if amp_at else 1
    amp_db = {}
    for t_sec, rms in amp_at.items():
        if rms > 0 and peak_rms > 0:
            amp_db[t_sec] = 20 * np.log10(rms / peak_rms)
        else:
            amp_db[t_sec] = -100

    # Decay analysis: time to -10dB, -20dB, -40dB
    rms_env = []
    hop = int(sr * 0.01)  # 10ms hop
    for i in range(0, n - window, hop):
        rms_env.append(np.sqrt(np.mean(signal[i:i+window]**2)))
    rms_env = np.array(rms_env)
    if len(rms_env) == 0:
        return {}
    peak_rms_env = np.max(rms_env)
    if peak_rms_env > 0:
        rms_db_env = 20 * np.log10(np.maximum(rms_env / peak_rms_env, 1e-10))
    else:
        rms_db_env = np.full_like(rms_env, -100)

    decay_times = {}
    for threshold in [-10, -20, -40]:
        below = np.where(rms_db_env < threshold)[0]
        if len(below) > 0:
            decay_times[threshold] = below[0] * 0.01
        else:
            decay_times[threshold] = duration

    # Spectral analysis: first 200ms (attack) and 500ms-1500ms (sustain)
    def spectral_profile(start_s, end_s):
        s = int(start_s * sr)
        e = min(int(end_s * sr), n)
        chunk = signal[s:e]
        if len(chunk) < 1024:
            return None, None, None
        # Apply window
        w = np.hanning(len(chunk))
        spectrum = np.abs(np.fft.rfft(chunk * w))
        freqs = np.fft.rfftfreq(len(chunk), 1/sr)
        # Spectral centroid
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        else:
            centroid = 0
        # Energy in bands
        bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 20000)]
        band_energy = {}
        total_energy = np.sum(spectrum**2)
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            be = np.sum(spectrum[mask]**2)
            band_energy[f"{lo}-{hi}"] = be / total_energy * 100 if total_energy > 0 else 0
        return centroid, band_energy, spectrum

    attack_centroid, attack_bands, attack_spec = spectral_profile(0.0, 0.2)
    sustain_centroid, sustain_bands, sustain_spec = spectral_profile(0.5, 1.5)

    # Spectral slope (dB/octave) in sustain region
    if sustain_spec is not None:
        freqs = np.fft.rfftfreq(int(1.0 * sr), 1/sr)  # approx
        # Measure energy at fundamental vs 5x fundamental
        # (we don't know the fundamental, but centroid gives a rough sense)

    return {
        'duration': duration,
        'peak_time': peak_time,
        'decay_10dB': decay_times.get(-10, None),
        'decay_20dB': decay_times.get(-20, None),
        'decay_40dB': decay_times.get(-40, None),
        'amp_envelope_dB': amp_db,
        'attack_centroid': attack_centroid,
        'sustain_centroid': sustain_centroid,
        'attack_bands': attack_bands,
        'sustain_bands': sustain_bands,
    }


def format_bands(bands):
    if bands is None:
        return "N/A"
    parts = []
    for band, pct in sorted(bands.items(), key=lambda x: int(x[0].split('-')[0])):
        if pct > 0.5:
            parts.append(f"{band}:{pct:.1f}%")
    return "  ".join(parts)


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    # Original recorded piano uses different note names
    # Map: find common notes or closest matches
    # Original: C2(36), Ds2(39), Fs2(42), A2(45), C3(48), Ds3(51), Fs3(54), A3(57),
    #           C4(60), Ds4(63), Fs4(66), A4(69), C5(72), Ds5(75), Fs5(78), A5(81), C6(84)
    # Generated: B1(35), E2(40), A2(45), D3(50), G3(55), B3(59), D4(62), F4(65),
    #            B4(71), E5(76), A5(81), D6(86)

    # Common or very close notes for comparison
    comparisons = [
        # (midi, orig_file, gen_file, note_name)
        (45, 'A2', 'A2', 'A2 (midi 45)'),
        (81, 'A5', 'A5', 'A5 (midi 81)'),
    ]

    # Also compare nearby notes
    nearby = [
        (36, 'C2', 40, 'E2', 'Low register: C2 orig vs E2 gen'),
        (60, 'C4', 62, 'D4', 'Middle register: C4 orig vs D4 gen'),
        (72, 'C5', 76, 'E5', 'High register: C5 orig vs E5 gen'),
    ]

    orig_dir = os.path.join(base, '..', 'audio', 'piano')
    gen_dir = os.path.join(base, '..', 'audio', 'grand-piano')

    print("=" * 90)
    print("GRAND PIANO COMPARISON: Recorded vs Generated")
    print("=" * 90)

    # Exact matches
    for midi, orig_file, gen_file, name in comparisons:
        print(f"\n{'─' * 90}")
        print(f"  {name}")
        print(f"{'─' * 90}")

        orig_path = os.path.join(orig_dir, f'{orig_file}.mp3')
        gen_path = os.path.join(gen_dir, f'{gen_file}.mp3')

        if not os.path.exists(orig_path) or not os.path.exists(gen_path):
            print(f"  Missing files, skipping")
            continue

        orig_sig = mp3_to_wav_array(orig_path)
        gen_sig = mp3_to_wav_array(gen_path)

        orig_a = analyze_note(orig_sig, label=f"orig {name}")
        gen_a = analyze_note(gen_sig, label=f"gen {name}")

        print(f"\n  {'Metric':<30} {'RECORDED':>20} {'GENERATED':>20}")
        print(f"  {'─'*30} {'─'*20} {'─'*20}")
        print(f"  {'Duration':<30} {orig_a['duration']:>19.2f}s {gen_a['duration']:>19.2f}s")
        print(f"  {'Peak time':<30} {orig_a['peak_time']*1000:>18.1f}ms {gen_a['peak_time']*1000:>18.1f}ms")
        print(f"  {'Decay to -10dB':<30} {orig_a['decay_10dB']:>19.2f}s {gen_a['decay_10dB']:>19.2f}s")
        print(f"  {'Decay to -20dB':<30} {orig_a['decay_20dB']:>19.2f}s {gen_a['decay_20dB']:>19.2f}s")
        print(f"  {'Decay to -40dB':<30} {orig_a['decay_40dB']:>19.2f}s {gen_a['decay_40dB']:>19.2f}s")

        print(f"\n  Amplitude envelope (dB relative to peak):")
        for t_sec in sorted(orig_a['amp_envelope_dB'].keys()):
            o = orig_a['amp_envelope_dB'].get(t_sec, -100)
            g = gen_a['amp_envelope_dB'].get(t_sec, -100)
            print(f"    t={t_sec:>4.2f}s: {'RECORDED':>10} {o:>7.1f} dB   {'GENERATED':>10} {g:>7.1f} dB   Δ={g-o:>+.1f} dB")

        print(f"\n  Spectral centroid:")
        oc = orig_a['attack_centroid']
        gc = gen_a['attack_centroid']
        print(f"    Attack (0-200ms):   RECORDED {oc:>7.0f} Hz   GENERATED {gc:>7.0f} Hz   ratio={gc/oc:.2f}" if oc and gc else "    N/A")
        oc = orig_a['sustain_centroid']
        gc = gen_a['sustain_centroid']
        print(f"    Sustain (0.5-1.5s): RECORDED {oc:>7.0f} Hz   GENERATED {gc:>7.0f} Hz   ratio={gc/oc:.2f}" if oc and gc else "    N/A")

        print(f"\n  Spectral energy distribution (attack 0-200ms):")
        print(f"    RECORDED: {format_bands(orig_a['attack_bands'])}")
        print(f"    GENERATED: {format_bands(gen_a['attack_bands'])}")

        print(f"\n  Spectral energy distribution (sustain 0.5-1.5s):")
        print(f"    RECORDED: {format_bands(sustain_bands)}" if (sustain_bands := orig_a['sustain_bands']) else "    N/A")
        print(f"    GENERATED: {format_bands(sustain_bands)}" if (sustain_bands := gen_a['sustain_bands']) else "    N/A")

    # Nearby comparisons
    for orig_midi, orig_file, gen_midi, gen_file, desc in nearby:
        print(f"\n{'─' * 90}")
        print(f"  {desc}")
        print(f"{'─' * 90}")

        orig_path = os.path.join(orig_dir, f'{orig_file}.mp3')
        gen_path = os.path.join(gen_dir, f'{gen_file}.mp3')

        if not os.path.exists(orig_path) or not os.path.exists(gen_path):
            print(f"  Missing files, skipping")
            continue

        orig_sig = mp3_to_wav_array(orig_path)
        gen_sig = mp3_to_wav_array(gen_path)

        orig_a = analyze_note(orig_sig)
        gen_a = analyze_note(gen_sig)

        print(f"\n  {'Metric':<30} {'RECORDED':>20} {'GENERATED':>20}")
        print(f"  {'─'*30} {'─'*20} {'─'*20}")
        print(f"  {'Duration':<30} {orig_a['duration']:>19.2f}s {gen_a['duration']:>19.2f}s")
        print(f"  {'Peak time':<30} {orig_a['peak_time']*1000:>18.1f}ms {gen_a['peak_time']*1000:>18.1f}ms")
        print(f"  {'Decay to -10dB':<30} {orig_a['decay_10dB']:>19.2f}s {gen_a['decay_10dB']:>19.2f}s")
        print(f"  {'Decay to -20dB':<30} {orig_a['decay_20dB']:>19.2f}s {gen_a['decay_20dB']:>19.2f}s")
        print(f"  {'Decay to -40dB':<30} {orig_a['decay_40dB']:>19.2f}s {gen_a['decay_40dB']:>19.2f}s")

        print(f"\n  Amplitude envelope (dB relative to peak):")
        for t_sec in sorted(orig_a['amp_envelope_dB'].keys()):
            o = orig_a['amp_envelope_dB'].get(t_sec, -100)
            g = gen_a['amp_envelope_dB'].get(t_sec, -100)
            print(f"    t={t_sec:>4.2f}s: {'RECORDED':>10} {o:>7.1f} dB   {'GENERATED':>10} {g:>7.1f} dB   Δ={g-o:>+.1f} dB")

        print(f"\n  Spectral centroid:")
        oc = orig_a['attack_centroid']
        gc = gen_a['attack_centroid']
        if oc and gc:
            print(f"    Attack (0-200ms):   RECORDED {oc:>7.0f} Hz   GENERATED {gc:>7.0f} Hz   ratio={gc/oc:.2f}")
        oc = orig_a['sustain_centroid']
        gc = gen_a['sustain_centroid']
        if oc and gc:
            print(f"    Sustain (0.5-1.5s): RECORDED {oc:>7.0f} Hz   GENERATED {gc:>7.0f} Hz   ratio={gc/oc:.2f}")

        print(f"\n  Spectral energy distribution (attack 0-200ms):")
        print(f"    RECORDED:  {format_bands(orig_a['attack_bands'])}")
        print(f"    GENERATED: {format_bands(gen_a['attack_bands'])}")

        print(f"\n  Spectral energy distribution (sustain 0.5-1.5s):")
        print(f"    RECORDED:  {format_bands(orig_a['sustain_bands'])}")
        print(f"    GENERATED: {format_bands(gen_a['sustain_bands'])}")

    print(f"\n{'=' * 90}")
    print("SUMMARY OF KEY DIFFERENCES")
    print("=" * 90)


if __name__ == '__main__':
    main()
