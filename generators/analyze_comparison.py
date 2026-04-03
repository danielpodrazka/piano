#!/usr/bin/env python3
"""
Comprehensive comparison of generated grand piano vs Salamander (Yamaha C5) samples.

Analyzes across multiple dimensions:
1. Spectral envelope & rolloff
2. Inharmonicity (partial stretching)
3. Decay rates (per-partial and overall)
4. Attack transient character
5. Spectral centroid over time (brightness evolution)
6. Two-stage decay (prompt vs aftersound)
7. Dynamic range / loudness profile
8. Phantom partials / metallic content
"""

import numpy as np
import subprocess
import wave
import os
import json
import tempfile

SAMPLE_RATE = 44100

NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']

def name_to_midi(name):
    for i, n in enumerate(NOTE_NAMES):
        if name.startswith(n) and name[len(n):].lstrip('-').isdigit():
            octave = int(name[len(n):])
            return (octave + 1) * 12 + i
    return None

def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)

def load_mp3(path):
    """Load MP3 as mono float array via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(SAMPLE_RATE),
            '-sample_fmt', 's16', tmp_path
        ], capture_output=True, check=True)
        with wave.open(tmp_path, 'r') as wf:
            frames = wf.readframes(wf.getnframes())
            signal = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
            signal /= 32768.0
        return signal
    finally:
        os.unlink(tmp_path)


def spectral_analysis(signal, freq, label):
    """Compute spectral envelope, partial amplitudes, and inharmonicity."""
    # Use first 2 seconds for spectral content
    n = min(len(signal), int(2.0 * SAMPLE_RATE))
    seg = signal[:n]

    # Window and FFT
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)

    # Find partials (peaks near expected harmonic frequencies)
    partials = []
    for h in range(1, 33):  # up to 32 partials
        expected = h * freq
        if expected > SAMPLE_RATE / 2 - 200:
            break
        # Search window: ±3% of expected frequency (accounts for inharmonicity)
        search_width = max(expected * 0.03, 5.0)
        mask = (freqs > expected - search_width) & (freqs < expected + search_width)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        peak_idx = idx[np.argmax(spectrum[idx])]
        peak_freq = freqs[peak_idx]
        peak_amp = spectrum[peak_idx]

        # Inharmonicity: deviation from ideal harmonic
        ideal = h * freq
        cents_deviation = 1200 * np.log2(peak_freq / ideal) if peak_freq > 0 and ideal > 0 else 0

        partials.append({
            'harmonic': h,
            'freq': float(peak_freq),
            'amp': float(peak_amp),
            'cents_sharp': float(cents_deviation),
        })

    # Normalize amplitudes to fundamental
    if partials and partials[0]['amp'] > 0:
        fund_amp = partials[0]['amp']
        for p in partials:
            p['amp_db'] = float(20 * np.log10(max(p['amp'] / fund_amp, 1e-10)))

    return partials


def decay_analysis(signal, freq):
    """Analyze decay envelope: overall and per-partial."""
    duration = len(signal) / SAMPLE_RATE

    # Overall RMS envelope in 50ms windows
    win_size = int(0.05 * SAMPLE_RATE)
    hop = win_size // 2
    rms_env = []
    times = []
    for i in range(0, len(signal) - win_size, hop):
        rms = np.sqrt(np.mean(signal[i:i+win_size] ** 2))
        rms_env.append(float(rms))
        times.append(float((i + win_size // 2) / SAMPLE_RATE))

    rms_env = np.array(rms_env)
    times = np.array(times)

    # Find peak and measure decay from there
    peak_idx = np.argmax(rms_env)
    peak_time = times[peak_idx]
    peak_rms = rms_env[peak_idx]

    # Time to -6dB, -20dB, -40dB from peak
    decay_times = {}
    for db_drop in [6, 20, 40]:
        threshold = peak_rms * 10 ** (-db_drop / 20)
        below = np.where((rms_env[peak_idx:] < threshold))[0]
        if len(below) > 0:
            decay_times[f't_{db_drop}dB'] = float(times[peak_idx + below[0]] - peak_time)
        else:
            decay_times[f't_{db_drop}dB'] = float(duration - peak_time)

    # Two-stage decay detection: fit biexponential
    # Look at log envelope after peak
    post_peak = rms_env[peak_idx:]
    post_times = times[peak_idx:] - times[peak_idx]
    valid = post_peak > peak_rms * 0.001  # above -60dB
    if np.sum(valid) > 10:
        log_env = np.log(post_peak[valid] + 1e-10)
        t_valid = post_times[valid]

        # Early decay (first 0.5s after peak) vs late decay (1s-3s)
        early_mask = t_valid < 0.5
        late_mask = (t_valid > 1.0) & (t_valid < 3.0)

        early_rate = None
        late_rate = None
        if np.sum(early_mask) > 3:
            p = np.polyfit(t_valid[early_mask], log_env[early_mask], 1)
            early_rate = float(-p[0])  # decay rate in nepers/s
        if np.sum(late_mask) > 3:
            p = np.polyfit(t_valid[late_mask], log_env[late_mask], 1)
            late_rate = float(-p[0])

        decay_times['early_rate'] = early_rate
        decay_times['late_rate'] = late_rate
        if early_rate and late_rate and late_rate > 0:
            decay_times['prompt_aftersound_ratio'] = float(early_rate / late_rate)

    decay_times['peak_time'] = float(peak_time)
    return decay_times


def attack_analysis(signal):
    """Analyze attack transient: rise time, peak time, noise content."""
    # First 100ms
    n = min(len(signal), int(0.1 * SAMPLE_RATE))
    seg = signal[:n]

    # Envelope via Hilbert-like (rectify + smooth)
    rectified = np.abs(seg)
    smooth_size = int(0.002 * SAMPLE_RATE)  # 2ms smoothing
    kernel = np.ones(smooth_size) / smooth_size
    env = np.convolve(rectified, kernel, mode='same')

    peak_idx = np.argmax(env)
    peak_time = peak_idx / SAMPLE_RATE

    # Rise time: 10% to 90% of peak
    peak_val = env[peak_idx]
    t10 = np.where(env[:peak_idx+1] > 0.1 * peak_val)[0]
    t90 = np.where(env[:peak_idx+1] > 0.9 * peak_val)[0]
    rise_time = None
    if len(t10) > 0 and len(t90) > 0:
        rise_time = float((t90[0] - t10[0]) / SAMPLE_RATE)

    # Spectral centroid of attack (first 30ms) — indicates noise/brightness
    attack_n = min(len(signal), int(0.03 * SAMPLE_RATE))
    attack_seg = signal[:attack_n] * np.hanning(attack_n)
    spec = np.abs(np.fft.rfft(attack_seg))
    freqs = np.fft.rfftfreq(attack_n, 1.0 / SAMPLE_RATE)
    spec_sum = np.sum(spec)
    if spec_sum > 0:
        attack_centroid = float(np.sum(freqs * spec) / spec_sum)
    else:
        attack_centroid = 0.0

    # High-frequency energy ratio in attack (noise indicator)
    hf_mask = freqs > 4000
    hf_ratio = float(np.sum(spec[hf_mask] ** 2) / (np.sum(spec ** 2) + 1e-10))

    return {
        'peak_time_ms': float(peak_time * 1000),
        'rise_time_ms': float(rise_time * 1000) if rise_time else None,
        'attack_centroid_hz': attack_centroid,
        'hf_energy_ratio': hf_ratio,
    }


def brightness_evolution(signal, freq):
    """Track spectral centroid over time — how brightness evolves during the note."""
    win_size = int(0.1 * SAMPLE_RATE)  # 100ms windows
    hop = win_size // 2
    centroids = []
    times = []

    for i in range(0, len(signal) - win_size, hop):
        seg = signal[i:i+win_size] * np.hanning(win_size)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(win_size, 1.0 / SAMPLE_RATE)
        spec_sum = np.sum(spec)
        if spec_sum > 1e-10:
            centroid = float(np.sum(freqs * spec) / spec_sum)
        else:
            centroid = 0.0
        centroids.append(centroid)
        times.append(float((i + win_size // 2) / SAMPLE_RATE))

    centroids = np.array(centroids)
    times = np.array(times)

    # Summarize: centroid at 0.1s, 0.5s, 1s, 2s, 4s
    summary = {}
    for t_target in [0.1, 0.5, 1.0, 2.0, 4.0]:
        idx = np.argmin(np.abs(times - t_target))
        if idx < len(centroids):
            summary[f'centroid_{t_target}s'] = float(centroids[idx])

    # Brightness decay: how fast does centroid drop?
    if len(centroids) > 5:
        peak_c_idx = np.argmax(centroids[:min(10, len(centroids))])
        half_centroid = centroids[peak_c_idx] * 0.5
        below = np.where(centroids[peak_c_idx:] < half_centroid)[0]
        if len(below) > 0:
            summary['brightness_halflife_s'] = float(times[peak_c_idx + below[0]] - times[peak_c_idx])

    return summary


def phantom_partial_analysis(signal, freq):
    """Check for phantom partials (sum-frequency components) in bass notes."""
    if freq > 350:  # only relevant for bass/mid
        return {'relevant': False}

    n = min(len(signal), int(1.0 * SAMPLE_RATE))
    seg = signal[:n]
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(seg * win))
    freqs_arr = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)

    # Look for energy at h2+h1 = 3f, h3+h1 = 4f (but these overlap with harmonics)
    # Better: look at h2+h3 = 5f region for non-harmonic bumps
    # Actually, phantom partials are at sum frequencies which may not be exactly harmonic
    # due to inharmonicity. Check energy between harmonics.

    inter_harmonic_energy = 0.0
    harmonic_energy = 0.0

    for h in range(2, 16):
        expected = h * freq
        if expected > SAMPLE_RATE / 2 - 500:
            break
        # Harmonic region: ±1%
        h_mask = (freqs_arr > expected * 0.99) & (freqs_arr < expected * 1.01)
        harmonic_energy += np.sum(spectrum[h_mask] ** 2)

        # Inter-harmonic region: between h and h+1
        mid = (h + 0.5) * freq
        ih_mask = (freqs_arr > mid * 0.97) & (freqs_arr < mid * 1.03)
        inter_harmonic_energy += np.sum(spectrum[ih_mask] ** 2)

    ratio = float(inter_harmonic_energy / (harmonic_energy + 1e-10))

    return {
        'relevant': True,
        'inter_harmonic_ratio_db': float(10 * np.log10(ratio + 1e-10)),
    }


def analyze_note(grand_path, sala_path, note_name):
    """Full comparison for one note."""
    midi = name_to_midi(note_name)
    freq = midi_to_freq(midi)

    grand = load_mp3(grand_path)
    sala = load_mp3(sala_path)

    result = {
        'note': note_name,
        'midi': midi,
        'freq': round(freq, 2),
    }

    # 1. Spectral analysis
    g_partials = spectral_analysis(grand, freq, 'grand')
    s_partials = spectral_analysis(sala, freq, 'sala')

    # Compare partial amplitudes
    partial_diffs = []
    for gp in g_partials:
        sp = next((p for p in s_partials if p['harmonic'] == gp['harmonic']), None)
        if sp and 'amp_db' in gp and 'amp_db' in sp:
            partial_diffs.append({
                'h': gp['harmonic'],
                'grand_db': round(gp['amp_db'], 1),
                'sala_db': round(sp['amp_db'], 1),
                'diff_db': round(gp['amp_db'] - sp['amp_db'], 1),
                'grand_cents': round(gp['cents_sharp'], 2),
                'sala_cents': round(sp['cents_sharp'], 2),
            })
    result['partial_comparison'] = partial_diffs

    # Spectral rolloff summary
    if len(g_partials) > 4 and len(s_partials) > 4:
        g_amps = [p.get('amp_db', -60) for p in g_partials[:16]]
        s_amps = [p.get('amp_db', -60) for p in s_partials[:16]]
        # Linear fit to get rolloff slope
        g_slope = np.polyfit(range(len(g_amps)), g_amps, 1)[0] if len(g_amps) > 2 else 0
        s_slope = np.polyfit(range(len(s_amps)), s_amps, 1)[0] if len(s_amps) > 2 else 0
        result['spectral_rolloff'] = {
            'grand_slope_db_per_partial': round(float(g_slope), 2),
            'sala_slope_db_per_partial': round(float(s_slope), 2),
        }

    # 2. Decay analysis
    result['decay_grand'] = decay_analysis(grand, freq)
    result['decay_sala'] = decay_analysis(sala, freq)

    # 3. Attack analysis
    result['attack_grand'] = attack_analysis(grand)
    result['attack_sala'] = attack_analysis(sala)

    # 4. Brightness evolution
    result['brightness_grand'] = brightness_evolution(grand, freq)
    result['brightness_sala'] = brightness_evolution(sala, freq)

    # 5. Phantom partials
    result['phantoms_grand'] = phantom_partial_analysis(grand, freq)
    result['phantoms_sala'] = phantom_partial_analysis(sala, freq)

    return result


def print_summary(results):
    """Print a readable summary of the comparison."""
    print("=" * 80)
    print("GRAND PIANO vs SALAMANDER (Yamaha C5) — COMPREHENSIVE COMPARISON")
    print("=" * 80)

    # Group findings
    print("\n1. SPECTRAL ROLLOFF (dB/partial, first 16 harmonics)")
    print(f"   {'Note':<6} {'Grand':>8} {'Sala':>8} {'Diff':>8}")
    print(f"   {'----':<6} {'-----':>8} {'----':>8} {'----':>8}")
    for r in results:
        if 'spectral_rolloff' in r:
            sr = r['spectral_rolloff']
            diff = sr['grand_slope_db_per_partial'] - sr['sala_slope_db_per_partial']
            print(f"   {r['note']:<6} {sr['grand_slope_db_per_partial']:>7.2f} {sr['sala_slope_db_per_partial']:>7.2f} {diff:>+7.2f}")

    print("\n2. INHARMONICITY (cents sharp from ideal, selected partials)")
    for r in results:
        if not r['partial_comparison']:
            continue
        print(f"\n   {r['note']} ({r['freq']} Hz):")
        print(f"   {'H':>4} {'Grand ¢':>9} {'Sala ¢':>9}")
        for p in r['partial_comparison']:
            if p['h'] in [1, 2, 4, 8, 16]:
                print(f"   {p['h']:>4} {p['grand_cents']:>+8.2f} {p['sala_cents']:>+8.2f}")

    print("\n3. DECAY TIMES")
    print(f"   {'Note':<6} {'Metric':<22} {'Grand':>8} {'Sala':>8}")
    print(f"   {'----':<6} {'------':<22} {'-----':>8} {'----':>8}")
    for r in results:
        dg, ds = r['decay_grand'], r['decay_sala']
        for key in ['t_6dB', 't_20dB', 't_40dB']:
            gv = dg.get(key, None)
            sv = ds.get(key, None)
            gstr = f"{gv:.2f}s" if gv else "N/A"
            sstr = f"{sv:.2f}s" if sv else "N/A"
            print(f"   {r['note']:<6} {key:<22} {gstr:>8} {sstr:>8}")
        # Prompt/aftersound ratio
        gr = dg.get('prompt_aftersound_ratio')
        sr_val = ds.get('prompt_aftersound_ratio')
        if gr and sr_val:
            print(f"   {r['note']:<6} {'prompt/after ratio':<22} {gr:>7.1f}x {sr_val:>7.1f}x")
        print()

    print("\n4. ATTACK CHARACTER")
    print(f"   {'Note':<6} {'Metric':<22} {'Grand':>10} {'Sala':>10}")
    print(f"   {'----':<6} {'------':<22} {'-----':>10} {'----':>10}")
    for r in results:
        ag, a_s = r['attack_grand'], r['attack_sala']
        print(f"   {r['note']:<6} {'peak time':<22} {ag['peak_time_ms']:>8.1f}ms {a_s['peak_time_ms']:>8.1f}ms")
        rt_g = f"{ag['rise_time_ms']:.1f}ms" if ag['rise_time_ms'] else "N/A"
        rt_s = f"{a_s['rise_time_ms']:.1f}ms" if a_s['rise_time_ms'] else "N/A"
        print(f"   {r['note']:<6} {'rise time':<22} {rt_g:>10} {rt_s:>10}")
        print(f"   {r['note']:<6} {'attack centroid':<22} {ag['attack_centroid_hz']:>8.0f}Hz {a_s['attack_centroid_hz']:>8.0f}Hz")
        print(f"   {r['note']:<6} {'HF energy ratio':<22} {ag['hf_energy_ratio']:>9.4f} {a_s['hf_energy_ratio']:>9.4f}")
        print()

    print("\n5. BRIGHTNESS EVOLUTION (spectral centroid over time)")
    print(f"   {'Note':<6} {'Time':<8} {'Grand':>8} {'Sala':>8} {'Diff':>8}")
    print(f"   {'----':<6} {'----':<8} {'-----':>8} {'----':>8} {'----':>8}")
    for r in results:
        bg, bs = r['brightness_grand'], r['brightness_sala']
        for t in ['0.1', '0.5', '1.0', '2.0']:
            gk = f'centroid_{t}s'
            if gk in bg and gk in bs:
                diff = bg[gk] - bs[gk]
                print(f"   {r['note']:<6} {t+'s':<8} {bg[gk]:>7.0f} {bs[gk]:>7.0f} {diff:>+7.0f}")
        bhl_g = bg.get('brightness_halflife_s')
        bhl_s = bs.get('brightness_halflife_s')
        if bhl_g and bhl_s:
            print(f"   {r['note']:<6} {'halflife':<8} {bhl_g:>6.2f}s {bhl_s:>6.2f}s")
        print()

    print("\n6. PHANTOM PARTIALS (inter-harmonic energy, bass notes only)")
    print(f"   {'Note':<6} {'Grand dB':>10} {'Sala dB':>10}")
    for r in results:
        pg, ps = r['phantoms_grand'], r['phantoms_sala']
        if pg.get('relevant'):
            gdb = f"{pg['inter_harmonic_ratio_db']:.1f}" if 'inter_harmonic_ratio_db' in pg else "N/A"
            sdb = f"{ps['inter_harmonic_ratio_db']:.1f}" if 'inter_harmonic_ratio_db' in ps else "N/A"
            print(f"   {r['note']:<6} {gdb:>10} {sdb:>10}")

    # Per-partial amplitude comparison for a few representative notes
    print("\n7. DETAILED PARTIAL AMPLITUDES (dB relative to fundamental)")
    for r in results:
        if r['note'] in ['C2', 'C4', 'A4', 'C6']:
            print(f"\n   {r['note']} ({r['freq']} Hz):")
            print(f"   {'H':>4} {'Grand dB':>10} {'Sala dB':>10} {'Diff':>8}")
            for p in r['partial_comparison'][:16]:
                print(f"   {p['h']:>4} {p['grand_db']:>9.1f} {p['sala_db']:>9.1f} {p['diff_db']:>+7.1f}")

    print("\n" + "=" * 80)
    print("OVERALL OBSERVATIONS")
    print("=" * 80)

    # Aggregate stats
    all_rolloff_diffs = []
    all_decay_ratios = {'6dB': [], '20dB': [], '40dB': []}
    all_attack_centroid_diffs = []
    all_brightness_diffs = []

    for r in results:
        if 'spectral_rolloff' in r:
            sr = r['spectral_rolloff']
            all_rolloff_diffs.append(sr['grand_slope_db_per_partial'] - sr['sala_slope_db_per_partial'])

        dg, ds = r['decay_grand'], r['decay_sala']
        for db in ['6', '20', '40']:
            k = f't_{db}dB'
            if dg.get(k) and ds.get(k) and ds[k] > 0:
                all_decay_ratios[f'{db}dB'].append(dg[k] / ds[k])

        ag, a_s = r['attack_grand'], r['attack_sala']
        all_attack_centroid_diffs.append(ag['attack_centroid_hz'] - a_s['attack_centroid_hz'])

        bg, bs = r['brightness_grand'], r['brightness_sala']
        if 'centroid_0.5s' in bg and 'centroid_0.5s' in bs:
            all_brightness_diffs.append(bg['centroid_0.5s'] - bs['centroid_0.5s'])

    if all_rolloff_diffs:
        mean_rd = np.mean(all_rolloff_diffs)
        print(f"\n  Spectral rolloff: Grand is {'steeper' if mean_rd < 0 else 'gentler'} by {abs(mean_rd):.2f} dB/partial on average")

    for db, ratios in all_decay_ratios.items():
        if ratios:
            mean_r = np.mean(ratios)
            print(f"  Decay to -{db}: Grand is {mean_r:.2f}x Salamander ({('shorter' if mean_r < 1 else 'longer')})")

    if all_attack_centroid_diffs:
        mean_ac = np.mean(all_attack_centroid_diffs)
        print(f"  Attack brightness: Grand is {abs(mean_ac):.0f} Hz {'higher' if mean_ac > 0 else 'lower'} centroid on average")

    if all_brightness_diffs:
        mean_bd = np.mean(all_brightness_diffs)
        print(f"  Sustain brightness (0.5s): Grand is {abs(mean_bd):.0f} Hz {'higher' if mean_bd > 0 else 'lower'} centroid on average")

    print()


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audio')
    grand_dir = os.path.join(base, 'grand-piano')
    sala_dir = os.path.join(base, 'salamander')

    # Find overlapping notes
    grand_notes = {f.replace('.mp3', '') for f in os.listdir(grand_dir) if f.endswith('.mp3')}
    sala_notes = {f.replace('.mp3', '') for f in os.listdir(sala_dir) if f.endswith('.mp3')}
    common = sorted(grand_notes & sala_notes, key=lambda n: name_to_midi(n))

    print(f"Found {len(common)} overlapping notes: {', '.join(common)}")
    print()

    results = []
    for note in common:
        grand_path = os.path.join(grand_dir, f'{note}.mp3')
        sala_path = os.path.join(sala_dir, f'{note}.mp3')
        print(f"Analyzing {note}...", flush=True)
        result = analyze_note(grand_path, sala_path, note)
        results.append(result)

    print()
    print_summary(results)


if __name__ == '__main__':
    main()
