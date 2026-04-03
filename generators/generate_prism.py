#!/usr/bin/env python3
"""
Generate "Prism Keys" — a retro-futuristic electronic piano.

Layers:
  1. UNISON BASE: 3 slightly detuned copies of fundamental (chorus shimmer)
  2. FIFTH: Open fifth (3:2 ratio) for hollow, airy quality
  3. BELL: FM bell at 7:1 ratio — sparkly, crystalline attack
  4. SUB: Gentle sub-octave for warmth
  5. TREMOLO: Slow amplitude modulation for organic movement
  6. ATTACK CLICK: Tiny broadband FM burst for percussive definition

Output: MP3 files for the learn-piano.html sampler.
"""

import numpy as np
import os
import subprocess

SAMPLE_RATE = 44100
DURATION = 5.0

# Borrow piano soundboard transfer functions for acoustic body/resonance
_TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
_TF_DATA = None
_IR_CACHE = {}

if os.path.exists(_TF_PATH):
    _TF_DATA = np.load(_TF_PATH)
    _TF_MIDI = _TF_DATA['midi_points']
    _TF_MAG = _TF_DATA['transfer_functions']
    _TF_IR_LEN = int(_TF_DATA['ir_length'])


def _minimum_phase_ir(magnitude_response, ir_length):
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


def _get_body_ir(midi):
    """Get per-note IR from piano soundboard TFs — adds acoustic resonance."""
    if _TF_DATA is None:
        return None
    if midi in _IR_CACHE:
        return _IR_CACHE[midi]

    log_tfs = np.log(_TF_MAG + 1e-10)
    interp_log = np.zeros(log_tfs.shape[1])
    for i_bin in range(log_tfs.shape[1]):
        interp_log[i_bin] = np.interp(midi, _TF_MIDI, log_tfs[:, i_bin])
    mag_response = np.exp(interp_log)

    # Use shorter IR for prism — just coloring, not full piano resonance
    ir_len = min(_TF_IR_LEN, 1024)
    ir = _minimum_phase_ir(mag_response, ir_len)
    ir_energy = np.sqrt(np.sum(ir ** 2))
    if ir_energy > 0:
        n_fft = (len(_TF_MAG[0]) - 1) * 2
        ir = ir / ir_energy * np.sqrt(ir_len) / np.sqrt(n_fft)

    _IR_CACHE[midi] = ir
    return ir

NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']

NOTES = [(m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}") for m in range(35, 87)]

VELOCITY_LAYERS = [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 1.00]


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def generate_prism_note(midi, duration=DURATION, velocity=0.75):
    """Generate a Prism Keys note with velocity-sensitive synthesis.

    Velocity model (Prism-specific):
      [1] FM bell mod index scales with velocity — crystalline sparkle at ff
      [2] Unison detune widens with velocity — thicker chorus at ff
      [3] Attack click intensity scales with velocity — percussive snap at ff
      [4] Fifth prominence grows with velocity — more open voicing at ff
      [5] Saturation drive increases with velocity — warmer edge at ff
      [6] Velocity amplitude — energy scaling
    """
    freq = midi_to_freq(midi)
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    key_scale = np.clip(1.0 - (midi - 40) / 60, 0.25, 1.0)

    vel_clamp = max(velocity, 0.05)
    effective_vel = 0.15 + 0.85 * vel_clamp ** 1.5

    # --- Envelope: soft attack, smooth decay ---
    attack_time = 0.008  # 8ms — slightly softer than Rhodes
    attack = np.minimum(t / attack_time, 1.0)
    decay_rate = 0.4 + (midi - 35) * 0.006
    env = attack * np.exp(-t * decay_rate)

    # ═══ LAYER 1: UNISON (3 detuned voices) ═══
    # [2] Detune widens with velocity: pp ±2 cents (tight), ff ±5 cents (wide chorus)
    detune_cents = 2.0 + 3.0 * effective_vel
    detune_ratio = 2 ** (detune_cents / 1200)
    phase_c = 2 * np.pi * freq * t
    phase_up = 2 * np.pi * freq * detune_ratio * t
    phase_dn = 2 * np.pi * freq / detune_ratio * t
    unison = (np.sin(phase_c) * 0.5 +
              np.sin(phase_up) * 0.25 +
              np.sin(phase_dn) * 0.25)

    # ═══ LAYER 2: OPEN FIFTH ═══
    # [4] Fifth prominence grows with velocity — spacious shimmer at ff
    fifth_freq = freq * 1.5
    fifth_env = np.exp(-t * (decay_rate + 0.5))
    fifth_amp = 0.10 + 0.14 * effective_vel  # pp: 0.12, fff: 0.24
    if fifth_freq < SAMPLE_RATE / 2 - 500:
        fifth = fifth_amp * fifth_env * np.sin(2 * np.pi * fifth_freq * t)
    else:
        fifth = np.zeros(n_samples)

    # ═══ LAYER 3: FM BELL (7:1 ratio) ═══
    # [1] Bell mod index scales with velocity — crystalline sparkle at ff
    bell_ratio = 7.0
    bell_mod_freq = freq * bell_ratio
    if bell_mod_freq < SAMPLE_RATE / 2 - 1000:
        bell_mod_idx = (0.8 + 0.4 * key_scale) * (0.3 + 0.9 * effective_vel)
        bell_mod_decay = (8.0 - 4.0 * key_scale) * (1.3 - 0.4 * effective_vel)
        bell_mod_env = bell_mod_idx * np.exp(-t * bell_mod_decay)
        bell_mod = bell_mod_env * np.sin(2 * np.pi * bell_mod_freq * t)
        bell_carrier_decay = (4.0 - 2.0 * key_scale) * (1.2 - 0.3 * effective_vel)
        bell = np.exp(-t * bell_carrier_decay) * np.sin(phase_c + bell_mod)
    else:
        bell = np.zeros(n_samples)

    # ═══ LAYER 4: SUB OCTAVE ═══
    sub = 0.10 * key_scale * np.sin(np.pi * freq * t)

    # ═══ LAYER 5: ATTACK CLICK ═══
    # [3] Click intensity scales with velocity — percussive snap at ff
    click_mod_idx = 1.0 + 3.5 * effective_vel  # pp: 1.15, fff: 4.5
    click_env = np.exp(-t * 80.0)  # gone in ~25ms
    click_mod = click_mod_idx * click_env * np.sin(2 * np.pi * freq * 5.5 * t)
    click_amp = 0.05 + 0.15 * effective_vel  # pp: 0.07, fff: 0.20
    click = click_amp * click_env * np.sin(phase_c + click_mod)

    # ═══ MIX ═══
    bell_mix = (0.30 + 0.15 * key_scale) * (0.5 + 0.7 * effective_vel)
    signal = unison * 0.55 + bell * bell_mix + fifth + sub + click

    # Apply envelope
    signal *= env

    # ═══ TREMOLO ═══
    # Slow amplitude modulation — organic, breathing quality
    trem_rate = 4.5  # Hz
    trem_depth = 0.08  # subtle
    tremolo = 1.0 - trem_depth * (0.5 + 0.5 * np.sin(2 * np.pi * trem_rate * t))
    signal *= tremolo

    # ═══ SOFT SATURATION ═══
    # [5] Saturation drive increases with velocity — pp: clean, ff: warm edge
    drive = 0.8 + 0.8 * effective_vel  # pp: 0.92, fff: 1.6
    signal = np.tanh(signal * drive) / np.tanh(drive)

    # ═══ ACOUSTIC BODY IR ═══
    # Borrow piano soundboard resonance for richness — subtle mix
    body_ir = _get_body_ir(midi)
    if body_ir is not None:
        wet = np.convolve(signal, body_ir, mode='full')[:n_samples]
        dry_rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
        wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-10
        wet *= dry_rms / wet_rms
        signal = 0.6 * signal + 0.4 * wet

    # [6] Velocity amplitude
    vel_amp = vel_clamp ** 1.0  # linear — electronic instrument, wide dynamics
    signal *= vel_amp

    # --- Fade out ---
    fade_samples = int(0.05 * SAMPLE_RATE)
    signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # Normalize — returns (signal, raw_peak) for relative normalization
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    return signal, peak


def write_wav(filename, signal):
    pcm = (signal * 32767).astype(np.int16)
    import wave
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def wav_to_mp3(wav_path, mp3_path):
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '128k',
        '-ar', '44100', mp3_path
    ], capture_output=True)


def main():
    import sys
    velocity_layers = '--velocity-layers' in sys.argv

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audio', 'prism')
    os.makedirs(out_dir, exist_ok=True)

    if velocity_layers:
        print(f"Generating {len(NOTES)} × {len(VELOCITY_LAYERS)} velocity layers = {len(NOTES) * len(VELOCITY_LAYERS)} Prism Keys samples...")
        print(f"  Velocities: {VELOCITY_LAYERS}")
    else:
        print(f"Generating {len(NOTES)} Prism Keys samples...")
    print()

    for midi, name in NOTES:
        freq = midi_to_freq(midi)

        if velocity_layers:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz", end='', flush=True)
            layer_signals = []
            layer_peaks = []
            for v_idx, vel in enumerate(VELOCITY_LAYERS):
                signal, raw_peak = generate_prism_note(midi, velocity=vel)
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
            signal, _ = generate_prism_note(midi)

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
