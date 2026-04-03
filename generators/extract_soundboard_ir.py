#!/usr/bin/env python3
"""
Extract per-note soundboard transfer functions by spectral division.

For each reference recording:
  1. Synthesize the same note with the dry physics model (IR disabled)
  2. Compute H(f) = FFT(recorded) / FFT(synthesized)
  3. Detrend to remove room/mic spectral tilt
  4. Save per-note magnitude responses as a lookup table

The generator interpolates between these for each note, then creates a
minimum-phase FIR filter on the fly. This models how the soundboard's
resonance structure varies across the bridge (different coupling for each note).

Output: soundboard_tf.npz containing:
  - midi_points: array of MIDI numbers with measured transfer functions
  - freqs: frequency axis
  - transfer_functions: [n_notes, n_bins] detrended magnitude responses
"""

import numpy as np
import subprocess
import tempfile
import os
import wave

SAMPLE_RATE = 44100
DURATION = 4.0
N_FFT = 8192
IR_LENGTH = 2048

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reference piano samples
REFERENCE_NOTES = {
    'C2': 36, 'A2': 45, 'C3': 48, 'A3': 57,
    'C4': 60, 'A4': 69, 'C5': 72, 'A5': 81,
}


def load_mp3(path):
    """Load an MP3 file as mono float32 numpy array."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        'ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE),
        '-ac', '1', '-f', 'wav', tmp_path
    ], capture_output=True)
    with wave.open(tmp_path, 'r') as wf:
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    os.unlink(tmp_path)
    return audio


def spectral_envelope(signal, n_fft=N_FFT, hop=2048):
    """Compute smoothed magnitude spectrum (average over time frames)."""
    n = len(signal)
    window = np.hanning(n_fft)
    mag_sum = np.zeros(n_fft // 2 + 1)
    count = 0
    for start in range(0, n - n_fft, hop):
        frame = signal[start:start + n_fft] * window
        spec = np.fft.rfft(frame)
        mag_sum += np.abs(spec)
        count += 1
    if count == 0:
        return np.ones(n_fft // 2 + 1)
    return mag_sum / count


def smooth_spectrum(spec, window_size=32):
    """Apply moving average smoothing to a spectrum."""
    kernel = np.ones(window_size) / window_size
    padded = np.pad(spec, window_size // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(spec)]


def main():
    # Import here to avoid circular dependency (generator loads the IR file)
    # We need to temporarily disable IR in the generator
    import generate_grand_piano as gen
    # Save and disable IR so we get dry synthesis
    saved_tf_data = gen._tf_data
    gen._tf_data = None

    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'audio', 'piano')

    n_bins = N_FFT // 2 + 1
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    idx_1k = np.argmin(np.abs(freqs - 1000))

    n_samples = int(SAMPLE_RATE * DURATION)

    # First pass: compute all per-note transfer functions
    midi_points = []
    raw_tfs = []

    for name, midi in sorted(REFERENCE_NOTES.items(), key=lambda x: x[1]):
        path = os.path.join(ref_dir, f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  {name}: not found, skipping")
            continue

        print(f"  {name} (MIDI {midi})...")

        recorded = load_mp3(path)
        if len(recorded) > n_samples:
            recorded = recorded[:n_samples]
        elif len(recorded) < n_samples:
            recorded = np.pad(recorded, (0, n_samples - len(recorded)))
        rec_peak = np.max(np.abs(recorded))
        if rec_peak > 0:
            recorded = recorded / rec_peak * 0.85

        synth, _ = gen.generate_grand_piano_note(midi, duration=DURATION)
        if len(synth) > n_samples:
            synth = synth[:n_samples]
        elif len(synth) < n_samples:
            synth = np.pad(synth, (0, n_samples - len(synth)))

        rec_spec = spectral_envelope(recorded, N_FFT)
        syn_spec = spectral_envelope(synth, N_FFT)

        syn_floor = np.max(syn_spec) * 0.001
        H = rec_spec / np.maximum(syn_spec, syn_floor)

        # Smooth per-note (less than before — keep more detail)
        H = smooth_spectrum(H, window_size=24)

        # Normalize at 1kHz
        H /= H[idx_1k]

        midi_points.append(midi)
        raw_tfs.append(H)
        print(f"    H range: {H.min():.3f} - {H.max():.3f}")

    if not raw_tfs:
        print("No reference samples found!")
        return

    midi_points = np.array(midi_points)
    raw_tfs = np.array(raw_tfs)  # [n_notes, n_bins]

    # Compute shared spectral tilt from average of all notes
    avg_log_H = np.mean(np.log(raw_tfs + 1e-10), axis=0)
    log_f = np.log(freqs + 1.0)
    mask = (freqs >= 50) & (freqs <= 8000)
    coeffs = np.polyfit(log_f[mask], avg_log_H[mask], 1)
    trend = np.exp(np.polyval(coeffs, log_f))
    trend /= trend[idx_1k]

    print(f"\nShared spectral tilt: slope = {coeffs[0]:.2f}")

    # Detrend each note's transfer function
    detrended_tfs = np.zeros_like(raw_tfs)
    for i in range(len(midi_points)):
        dt = raw_tfs[i] / trend
        detrended_tfs[i] = dt

    # Frequency-dependent clamp: ±12 dB below 2kHz, tapering to ±6 dB above 3kHz.
    for i_bin, f in enumerate(freqs):
        if f < 2000:
            max_db = 12.0
        elif f < 3000:
            max_db = 12.0 - 6.0 * (f - 2000) / 1000
        else:
            max_db = 6.0
        max_boost = 10 ** (max_db / 20)
        detrended_tfs[:, i_bin] = np.clip(detrended_tfs[:, i_bin],
                                          1.0 / max_boost, max_boost)

    # Above 10kHz: taper to unity
    for i_bin, f in enumerate(freqs):
        if f > 10000:
            t = min((f - 10000) / 5000, 1.0)
            detrended_tfs[:, i_bin] = 1.0 + (detrended_tfs[:, i_bin] - 1.0) * (1.0 - t)

    # Print per-note summaries
    for i, midi in enumerate(midi_points):
        name = [n for n, m in REFERENCE_NOTES.items() if m == midi][0]
        print(f"\n  {name} (MIDI {midi}) detrended:")
        for f_check in [200, 500, 1000, 2000, 4000, 8000]:
            idx = np.argmin(np.abs(freqs - f_check))
            db = 20 * np.log10(detrended_tfs[i, idx])
            print(f"    {f_check:5d} Hz: {db:+.1f} dB")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'soundboard_tf.npz')
    np.savez(out_path,
             midi_points=midi_points,
             freqs=freqs,
             transfer_functions=detrended_tfs,
             ir_length=IR_LENGTH)
    print(f"\nSaved {len(midi_points)} per-note transfer functions to {out_path}")

    # Restore
    gen._tf_data = saved_tf_data


if __name__ == '__main__':
    main()
