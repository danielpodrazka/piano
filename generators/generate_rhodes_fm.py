#!/usr/bin/env python3
"""
Generate Rhodes-style electric piano samples using DX7-style FM synthesis.

Based on:
- DX7 E.PIANO 1 patch (Algorithm 5): carrier-modulator FM pairs
  BODY: 1:1 ratio, moderate mod index, slow decay (warm fundamental)
  TINE: 9:1 ratio, fast decay (metallic bell attack)
- Rhodes Mark I physics:
  * Tine vibrates as near-perfect sine wave
  * Tonebar couples as resonator (slight detuning → shimmer)
  * Magnetic pickup is nonlinear → asymmetric distortion → even harmonics
- Deep spectral comparison against real 1977 Mark I samples:
  * Attack: near-instant, spectral flatness ~0 (pure tone, NO noise)
  * Centroid: only 1.5-3x fundamental even during attack
  * Harmonics persist through sustain (slow mod envelope decay)

Physical model per note:
  1. BODY FM: 1:1 carrier:modulator with decaying mod index (warm tone)
  2. TINE FM: 9:1 ratio metallic attack with fast decay (bell character)
  3. SUB-HARMONIC: subtle 0.5:1 undertone (key-dependent)
  4. ADDITIVE HARMONICS: h2, h3 for spectral fill
  5. MAGNETIC PICKUP: asymmetric tanh distortion (even harmonics, warmth)
  6. SMOOTH ATTACK: half-cosine rise (no derivative discontinuity)

Output: MP3 files ready for the learn-piano.html sampler.
"""

import numpy as np
import os
import subprocess

SAMPLE_RATE = 44100
DURATION = 5.0  # seconds per sample

# Per-note transfer functions extracted from real Rhodes recordings
# via spectral division (extract_rhodes_tf.py). Captures pickup/amp/cabinet
# character that varies across the keyboard.
_TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rhodes_tf.npz')
_TF_DATA = None
_IR_CACHE = {}

if os.path.exists(_TF_PATH):
    _TF_DATA = np.load(_TF_PATH)
    _TF_MIDI = _TF_DATA['midi_points']
    _TF_MAG = _TF_DATA['transfer_functions']
    _TF_IR_LEN = int(_TF_DATA['ir_length'])


def _minimum_phase_ir(magnitude_response, ir_length):
    """Create minimum-phase FIR from magnitude response (cepstral method)."""
    mag = np.maximum(magnitude_response, 1e-10)
    n_fft = (len(mag) - 1) * 2
    full_mag = np.concatenate([mag, mag[-2:0:-1]])
    log_mag = np.log(full_mag)
    cepstrum = np.fft.ifft(log_mag).real
    min_cep = np.zeros_like(cepstrum)
    min_cep[0] = cepstrum[0]
    min_cep[1:n_fft // 2] = 2 * cepstrum[1:n_fft // 2]
    min_cep[n_fft // 2] = cepstrum[n_fft // 2]
    ir = np.fft.ifft(np.exp(np.fft.fft(min_cep))).real
    ir = ir[:ir_length]
    ir *= np.hanning(ir_length * 2)[ir_length:]
    return ir


def _get_rhodes_ir(midi):
    """Get per-note Rhodes IR by interpolating reference transfer functions."""
    if _TF_DATA is None:
        return None
    if midi in _IR_CACHE:
        return _IR_CACHE[midi]

    log_tfs = np.log(_TF_MAG + 1e-10)
    interp_log = np.zeros(log_tfs.shape[1])
    for i_bin in range(log_tfs.shape[1]):
        interp_log[i_bin] = np.interp(midi, _TF_MIDI, log_tfs[:, i_bin])
    mag_response = np.exp(interp_log)

    ir = _minimum_phase_ir(mag_response, _TF_IR_LEN)
    ir_energy = np.sqrt(np.sum(ir ** 2))
    if ir_energy > 0:
        n_fft = (len(_TF_MAG[0]) - 1) * 2
        ir = ir / ir_energy * np.sqrt(_TF_IR_LEN) / np.sqrt(n_fft)

    _IR_CACHE[midi] = ir
    return ir


NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']


def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


# Every chromatic note from B1 (35) to D6 (86)
NOTES = [(m, midi_to_name(m)) for m in range(35, 87)]

# Per-component phase table from fixed seed (consistent timbre, varied onsets).
# Components: 0=body, 1=tine, 2=sub, 3=h2, 4=h3
_PHASE_TABLE = np.random.RandomState(7823).uniform(0, 2 * np.pi, (128, 5))


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


VELOCITY_LAYERS = [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 1.00]


def generate_rhodes_note(midi, duration=DURATION, velocity=0.75):
    """Generate a Rhodes note using DX7-style 3-component FM synthesis.

    Velocity model (Rhodes-specific):
      [1] FM mod index scales with velocity — brighter body at ff
      [2] Tine prominence increases with velocity — more bell/bark at ff
      [3] Pickup distortion drive scales with velocity — warm at pp, barky at ff
      [4] Tine decay slows at higher velocity — bell sustains longer at ff
      [5] Velocity amplitude — energy scaling
    """
    freq = midi_to_freq(midi)
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Key-dependent scaling (1.0 at low register, 0.25 at high)
    key_scale = np.clip(1.0 - (midi - 40) / 60, 0.25, 1.0)

    # Per-note initial phases (varied waveform onsets across keyboard)
    ph = _PHASE_TABLE[midi]

    # Velocity-dependent effective intensity (same curve as grand piano)
    vel_clamp = max(velocity, 0.05)
    effective_vel = 0.15 + 0.85 * vel_clamp ** 1.5

    # --- Amplitude envelope (near-instant attack, smooth decay) ---
    # Half-cosine attack (C¹-smooth, no derivative discontinuity at peak)
    attack_time = 0.002  # 2ms — Rhodes attack is near-instant
    attack = np.where(t < attack_time,
                      0.5 - 0.5 * np.cos(np.pi * t / attack_time),
                      1.0)
    # Calibrated from sampled Rhodes D3: 50% at 1.18s, 10% at 3.56s
    decay_rate = np.interp(midi, [35, 60, 86], [0.415, 0.54, 0.67])
    env = attack * np.exp(-t * decay_rate)

    # ═══ COMPONENT 1: BODY (1:1 ratio) ═══
    # [1] FM mod index scales with velocity — more harmonics at ff.
    # pp: mod_idx ≈ 0.6 (nearly pure sine), fff: mod_idx ≈ 2.0 (rich harmonics)
    body_mod_idx = 1.979 * effective_vel * key_scale
    mod_decay_rate = 0.305 + (1.0 - key_scale) * 1.579
    # Mod envelope decays slower at high velocity — brightness persists longer
    mod_decay_rate *= (1.3 - 0.4 * effective_vel)
    body_mod_env = body_mod_idx * np.exp(-t * mod_decay_rate)
    body_mod = body_mod_env * np.sin(2 * np.pi * freq * t + ph[0])
    body = np.sin(2 * np.pi * freq * t + body_mod + ph[0])

    # ═══ TINE (9:1 ratio — metallic bell attack) ═══
    # [2] Tine prominence scales with velocity — the "bell" that defines Rhodes
    # character becomes much more prominent at ff (the "bark").
    # [4] Tine carrier decays slower at high velocity — bell rings longer.
    tine_ratio = 9.0
    tine_mod_freq = freq * tine_ratio
    if tine_mod_freq < SAMPLE_RATE / 2 - 1000:
        tine_mod_idx = (0.5 + 0.3 * key_scale) * effective_vel
        tine_mod_decay = (14.0 - 8.0 * key_scale) * (1.2 - 0.3 * effective_vel)
        tine_mod_env = tine_mod_idx * np.exp(-t * tine_mod_decay)
        tine_mod = tine_mod_env * np.sin(2 * np.pi * tine_mod_freq * t + ph[1])
        tine_carrier_decay = (5.0 - 3.0 * key_scale) * (1.3 - 0.4 * effective_vel)
        tine_carrier_env = np.exp(-t * tine_carrier_decay)
        tine = tine_carrier_env * np.sin(2 * np.pi * freq * t + tine_mod + ph[1])
    else:
        tine = np.zeros(n_samples)

    # ═══ SUB-HARMONIC ═══
    sub_freq = freq * 0.5
    if sub_freq >= 20:
        sub = 0.023 * key_scale * np.sin(2 * np.pi * sub_freq * t + ph[2])
    else:
        sub = np.zeros(n_samples)

    # ═══ ADDITIVE HARMONICS ═══
    h2 = np.zeros(n_samples)
    h3 = np.zeros(n_samples)
    if freq * 2 < SAMPLE_RATE / 2:
        h2 = 0.008 * np.sin(2 * np.pi * freq * 2 * t + ph[3])
    if freq * 3 < SAMPLE_RATE / 2:
        h3 = 0.069 * np.sin(2 * np.pi * freq * 3 * t + ph[4])

    # ═══ MIX ═══
    # [2] Tine mix increases with velocity — pp is warm body, ff has prominent bell
    tine_mix = (0.20 + 0.15 * key_scale) * (0.5 + 0.8 * effective_vel)
    # pp: tine_mix ~0.13, fff: tine_mix ~0.38
    signal = body * 0.80 + tine * tine_mix + sub + h2 + h3

    # Apply amplitude envelope
    signal *= env

    # ═══ MAGNETIC PICKUP SIMULATION ═══
    # [3] Pickup distortion drive scales with velocity.
    # pp: gentle, clean sound. ff: pickup saturates → "bark" (even harmonics).
    # Real Rhodes: the tine moves closer to the pickup at higher velocity,
    # increasing the magnetic flux change → more distortion.
    drive = 0.10 + 0.40 * effective_vel  # pp: 0.12, fff: 0.50
    asym = 0.510
    pos = np.maximum(signal, 0)
    neg = np.minimum(signal, 0)
    signal = (np.tanh(pos * drive) / np.tanh(drive) +
              np.tanh(neg * drive * asym) / np.tanh(drive * asym))

    # ═══ PICKUP/AMP IR CONVOLUTION ═══
    rhodes_ir = _get_rhodes_ir(midi)
    if rhodes_ir is not None:
        wet = np.convolve(signal, rhodes_ir, mode='full')[:n_samples]
        dry_rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
        wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-10
        wet *= dry_rms / wet_rms
        signal = 0.3 * signal + 0.7 * wet

    # [5] Velocity amplitude — Rhodes has less dynamic range than grand piano
    vel_amp = vel_clamp ** 1.2  # pp(0.15)→0.11, mp(0.6)→0.54, fff(1.0)→1.0
    signal *= vel_amp

    # --- Fade out ---
    fade_samples = int(0.1 * SAMPLE_RATE)
    signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # Normalize — returns (signal, raw_peak) for relative normalization
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    return signal, peak


def write_wav(filename, signal):
    """Write a numpy array as a 16-bit WAV file."""
    pcm = (signal * 32767).astype(np.int16)
    import wave
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def wav_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '128k',
        '-ar', '44100', mp3_path
    ], capture_output=True)


def main():
    import sys
    velocity_layers = '--velocity-layers' in sys.argv

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audio', 'rhodes-fm')
    os.makedirs(out_dir, exist_ok=True)

    if velocity_layers:
        print(f"Generating {len(NOTES)} × {len(VELOCITY_LAYERS)} velocity layers = {len(NOTES) * len(VELOCITY_LAYERS)} Rhodes FM samples...")
        print(f"  Velocities: {VELOCITY_LAYERS}")
    else:
        print(f"Generating {len(NOTES)} Rhodes FM samples...")
    print()

    for midi, name in NOTES:
        freq = midi_to_freq(midi)

        if velocity_layers:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz", end='', flush=True)
            layer_signals = []
            layer_peaks = []
            for v_idx, vel in enumerate(VELOCITY_LAYERS):
                signal, raw_peak = generate_rhodes_note(midi, velocity=vel)
                signal = signal / 0.85 * raw_peak
                layer_signals.append(signal)
                layer_peaks.append(np.max(np.abs(signal)))

            max_peak = max(layer_peaks) if max(layer_peaks) > 0 else 1.0
            for v_idx, signal in enumerate(layer_signals):
                signal = signal / max_peak * 0.85
                v_dir = os.path.join(out_dir, f'v{v_idx + 1}')
                os.makedirs(v_dir, exist_ok=True)

                wav_path = os.path.join(v_dir, f'{name}.wav')
                mp3_path = os.path.join(v_dir, f'{name}.mp3')
                write_wav(wav_path, signal)
                wav_to_mp3(wav_path, mp3_path)
                if os.path.exists(mp3_path):
                    os.remove(wav_path)
                print(f" v{v_idx+1}", end='', flush=True)
            print()
        else:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz")
            signal, _ = generate_rhodes_note(midi)

            wav_path = os.path.join(out_dir, f'{name}.wav')
            mp3_path = os.path.join(out_dir, f'{name}.mp3')

            write_wav(wav_path, signal)
            wav_to_mp3(wav_path, mp3_path)

            if os.path.exists(mp3_path):
                os.remove(wav_path)
            else:
                print(f"    Warning: ffmpeg conversion failed, keeping WAV")

    print(f"\nDone! Samples written to {out_dir}/")
    print("Files:", ', '.join(f'{name}.mp3' for _, name in NOTES))


if __name__ == '__main__':
    main()
