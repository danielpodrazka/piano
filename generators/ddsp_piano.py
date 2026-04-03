#!/usr/bin/env python3
"""
Differentiable Digital Signal Processing (DDSP) Piano Synthesizer.

Physics-informed differentiable synthesis following Simionato et al. (2023)
and DDSP-Piano (Renault et al. 2022):
  - Harmonic component: inharmonic additive synthesis with learned spectral envelope
  - Noise component: learned filtered noise for hammer/room/string noise
  - Neural network predicts per-note parameters for both components
  - Trained end-to-end with Adam + multi-scale STFT loss

Usage:
    python3 ddsp_piano.py                  # Train + generate samples
    python3 ddsp_piano.py --epochs 2000    # More training iterations
    python3 ddsp_piano.py --generate-only  # Generate from saved model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import subprocess
import wave

SAMPLE_RATE = 44100
DURATION = 6.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ddsp_piano_model.pt')

# Pre-allocate
t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)
PHASE_TABLE_GPU = torch.tensor(
    np.random.RandomState(6454).uniform(0, 2 * np.pi, (3, 64)),
    dtype=torch.float32, device=DEVICE,
)
# Deterministic noise seed for reproducible noise component
NOISE_TABLE = torch.tensor(
    np.random.RandomState(7777).randn(N_SAMPLES),
    dtype=torch.float32, device=DEVICE,
)

# Training/generation note lists
TRAIN_NOTES = [
    (36, 'C2'), (39, 'Ds2'), (42, 'Fs2'), (45, 'A2'),
    (48, 'C3'), (51, 'Ds3'), (54, 'Fs3'), (57, 'A3'),
    (60, 'C4'), (63, 'Ds4'), (66, 'Fs4'), (69, 'A4'),
    (72, 'C5'), (75, 'Ds5'), (78, 'Fs5'), (81, 'A5'),
    (84, 'C6'),
]

NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
ALL_NOTES = [(m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}") for m in range(35, 87)]

# ═══ PHYSICS CONSTANTS ═══
B_MIDI = torch.tensor([21, 33, 45, 57, 69, 84, 96], dtype=torch.float32, device=DEVICE)
B_VALS_LOG = torch.log(torch.tensor(
    [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2],
    dtype=torch.float32, device=DEVICE))

L_MIDI = torch.tensor([36, 60, 96], dtype=torch.float32, device=DEVICE)
L_VALS_LOG = torch.log(torch.tensor([1.92, 0.62, 0.09], dtype=torch.float32, device=DEVICE))

SB_MODE_CFS = torch.tensor([90.0, 170.0, 260.0], device=DEVICE)
SB_MODE_BWS = torch.tensor([30.0, 35.0, 45.0], device=DEVICE)

# ═══ SOUNDBOARD IR CONVOLUTION ═══
# Per-note transfer functions extracted from reference recordings (extract_soundboard_ir.py).
# Detrended to remove room/mic coloring — keeps only resonance detail across the bridge.
_TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
_SB_IR_CACHE = {}  # midi → torch tensor IR

if os.path.exists(_TF_PATH):
    _tf_data = np.load(_TF_PATH)
    _TF_MIDI = _tf_data['midi_points']
    _TF_MAG = _tf_data['transfer_functions']  # [n_notes, n_bins]
    _TF_IR_LEN = int(_tf_data['ir_length'])

    def _min_phase_ir_np(mag_resp, ir_len):
        mag = np.maximum(mag_resp, 1e-10)
        n_fft = (len(mag) - 1) * 2
        full = np.concatenate([mag, mag[-2:0:-1]])
        cep = np.fft.ifft(np.log(full)).real
        mc = np.zeros_like(cep)
        mc[0] = cep[0]
        mc[1:n_fft//2] = 2 * cep[1:n_fft//2]
        mc[n_fft//2] = cep[n_fft//2]
        ir = np.fft.ifft(np.exp(np.fft.fft(mc))).real[:ir_len]
        ir *= np.hanning(ir_len * 2)[ir_len:]
        e = np.sqrt(np.sum(ir**2))
        if e > 0:
            ir = ir / e * np.sqrt(ir_len) / np.sqrt(n_fft)
        return ir

    def _get_sb_ir_torch(midi):
        if midi not in _SB_IR_CACHE:
            log_tfs = np.log(_TF_MAG + 1e-10)
            interp_log = np.array([np.interp(midi, _TF_MIDI, log_tfs[:, i])
                                   for i in range(log_tfs.shape[1])])
            mag = np.exp(interp_log)
            ir = _min_phase_ir_np(mag, _TF_IR_LEN)
            _SB_IR_CACHE[midi] = torch.tensor(ir.astype(np.float32), device=DEVICE)
        return _SB_IR_CACHE[midi]
else:
    _get_sb_ir_torch = lambda midi: None


def apply_soundboard_ir(signal, midi):
    """Apply per-note soundboard IR via differentiable convolution.

    Uses shorter effective IR (1024 samples / ~23ms) than the physics model
    to avoid smearing the attack, and lower wet mix since DDSP already has
    learned spectral corrections.
    """
    sb_ir = _get_sb_ir_torch(midi)
    if sb_ir is None:
        return signal
    # Truncate IR to reduce attack smearing (full IR = 2048 = ~46ms)
    max_ir_len = 1024
    short_ir = sb_ir[:max_ir_len] * torch.hann_window(max_ir_len * 2, device=DEVICE)[max_ir_len:]
    n_samples = len(signal)
    ir = short_ir.unsqueeze(0).unsqueeze(0)
    sig = signal.unsqueeze(0).unsqueeze(0)
    pad_len = len(short_ir) - 1
    wet = F.conv1d(F.pad(sig, (pad_len, 0)), ir).squeeze()[:n_samples]
    dry_rms = (signal ** 2).mean().sqrt() + 1e-10
    wet_rms = (wet ** 2).mean().sqrt() + 1e-10
    wet = wet * dry_rms / wet_rms
    return 0.55 * signal + 0.45 * wet


# Spectral control: 8 points in log-partial space
N_SPECTRAL_CTRL = 8
SPECTRAL_CTRL_PARTIALS = torch.tensor([1, 2, 4, 8, 16, 24, 32, 48],
                                       dtype=torch.float32, device=DEVICE)

# Noise filter: 16 frequency bands (bark-ish spacing)
N_NOISE_BANDS = 16


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def log_interp(midi, calib_midi, log_vals):
    midi_f = float(midi)
    if midi_f <= calib_midi[0].item():
        return torch.exp(log_vals[0])
    if midi_f >= calib_midi[-1].item():
        return torch.exp(log_vals[-1])
    for i in range(len(calib_midi) - 1):
        if midi_f <= calib_midi[i + 1].item():
            frac = (midi_f - calib_midi[i].item()) / (calib_midi[i + 1].item() - calib_midi[i].item())
            return torch.exp(log_vals[i] + frac * (log_vals[i + 1] - log_vals[i]))
    return torch.exp(log_vals[-1])


def interp_spectral_curve(ctrl_values, n_partials):
    """Interpolate control points to n_partials using log-partial spacing."""
    ctrl_x = torch.log(SPECTRAL_CTRL_PARTIALS[:len(ctrl_values)])
    target_x = torch.log(torch.arange(1, n_partials + 1, dtype=torch.float32, device=DEVICE))
    target_x_clamped = torch.clamp(target_x, ctrl_x[0], ctrl_x[-1])

    result = torch.zeros(n_partials, device=DEVICE)
    for i in range(len(ctrl_values) - 1):
        mask = (target_x_clamped >= ctrl_x[i]) & (target_x_clamped <= ctrl_x[i + 1])
        if mask.any():
            frac = (target_x_clamped[mask] - ctrl_x[i]) / (ctrl_x[i + 1] - ctrl_x[i] + 1e-8)
            result[mask] = ctrl_values[i] + frac * (ctrl_values[i + 1] - ctrl_values[i])
    beyond = target_x > ctrl_x[-1]
    if beyond.any():
        result[beyond] = ctrl_values[-1]
    return result


# ═══ NEURAL NETWORK ═══

class PianoParamNet(nn.Module):
    """Predicts per-note synthesis parameters from MIDI number.

    Harmonic params: damping, soundboard, two-stage decay, spectral envelope, decay curve
    Noise params: level, decay rate, spectral tilt, bandwidth
    """

    # 2 damping + 6 soundboard + 3 two-stage + 1 air + 2 hammer
    # + 8 spectral + 8 decay + 4 noise = 34
    N_PARAMS = 34

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, self.N_PARAMS),
        )
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].weight.normal_(std=0.01)

    def forward(self, midi_normalized):
        raw = self.net(midi_normalized)
        params = {}
        i = 0

        # All params are SMALL offsets from hand-tuned defaults.
        # Network output starts near zero → synthesis starts identical to grand piano.

        # Damping: ±0.3 (0.74x - 1.35x of hand-tuned value)
        params['log_b1_offset'] = torch.tanh(raw[i]) * 0.3; i += 1
        params['log_b2_offset'] = torch.tanh(raw[i]) * 0.3; i += 1

        # Soundboard: small offsets from hand-tuned (cf=2500, bw=1200, gain=0.3)
        params['sb_bridge_cf'] = 2500.0 + torch.tanh(raw[i]) * 500.0; i += 1   # 2000-3000
        params['sb_bridge_bw'] = 1200.0 + torch.tanh(raw[i]) * 400.0; i += 1   # 800-1600
        params['sb_bridge_gain'] = 0.3 + torch.tanh(raw[i]) * 0.15; i += 1     # 0.15-0.45
        # Low modes: hand-tuned (0.15, 0.12, 0.10) ± small offset
        params['sb_mode1_gain'] = 0.15 + torch.tanh(raw[i]) * 0.10; i += 1     # 0.05-0.25
        params['sb_mode2_gain'] = 0.12 + torch.tanh(raw[i]) * 0.08; i += 1     # 0.04-0.20
        params['sb_mode3_gain'] = 0.10 + torch.tanh(raw[i]) * 0.06; i += 1     # 0.04-0.16

        # Two-stage decay: hand-tuned defaults with key_pos factored into synth
        # prompt=1.5+0.5*kp, after=0.25, A_after=0.15+0.05*kp → allow small offsets
        params['prompt_factor_offset'] = torch.tanh(raw[i]) * 0.5; i += 1       # ±0.5 from default
        params['after_factor'] = 0.25 + torch.tanh(raw[i]) * 0.15; i += 1       # 0.10-0.40
        params['A_after_offset'] = torch.tanh(raw[i]) * 0.08; i += 1            # ±0.08

        # Air fraction: hand-tuned = 0.2
        params['air_frac'] = 0.2 + torch.tanh(raw[i]) * 0.15; i += 1           # 0.05-0.35

        # Hammer/attack: scale factors centered on 1.0
        params['hammer_cutoff_scale'] = 1.0 + torch.tanh(raw[i]) * 0.4; i += 1  # 0.6-1.4
        params['attack_peak_scale'] = 1.0 + torch.tanh(raw[i]) * 0.3; i += 1    # 0.7-1.3

        # Spectral envelope: ±1.5 range (±13 dB) — subtle corrections only
        params['spectral_ctrl'] = torch.tanh(raw[i:i+N_SPECTRAL_CTRL]) * 1.5; i += N_SPECTRAL_CTRL

        # Decay correction: ±0.5 (0.6x-1.6x of physics rate)
        params['decay_ctrl'] = torch.tanh(raw[i:i+N_SPECTRAL_CTRL]) * 0.5; i += N_SPECTRAL_CTRL

        # Noise: subtle attack transient only
        params['noise_level'] = torch.sigmoid(raw[i]) * 0.04; i += 1             # 0-0.04
        params['noise_decay'] = 15.0 + torch.sigmoid(raw[i]) * 60.0; i += 1      # fast: 15-75
        params['noise_tilt'] = torch.tanh(raw[i]) * 2.0; i += 1
        params['noise_bandwidth'] = 1.0 + torch.sigmoid(raw[i]) * 3.0; i += 1

        return params


# ═══ DIFFERENTIABLE SYNTHESIS ═══

def synthesize_harmonic(midi, params, t):
    """Harmonic (additive) component."""
    freq = midi_to_freq(midi)
    key_pos = max(0.0, min(1.0, (midi - 21) / 87.0))

    B = log_interp(midi, B_MIDI, B_VALS_LOG)
    L = log_interp(midi, L_MIDI, L_VALS_LOG)
    piL2 = (np.pi / L) ** 2

    b1_phys = log_interp(midi, torch.tensor([36.0, 60.0, 96.0], device=DEVICE),
                         torch.log(torch.tensor([0.25, 1.1, 9.17], device=DEVICE)))
    b2_phys = log_interp(midi, torch.tensor([36.0, 60.0, 96.0], device=DEVICE),
                         torch.log(torch.tensor([7.5e-5, 2.7e-4, 2.1e-3], device=DEVICE)))
    b1 = b1_phys * torch.exp(params['log_b1_offset'])
    b2 = b2_phys * torch.exp(params['log_b2_offset'])

    max_partial = 1
    while max_partial * freq * np.sqrt(1 + B.item() * max_partial**2) < SAMPLE_RATE / 2 - 500:
        max_partial += 1
    max_partial = min(max_partial - 1, 64)
    if max_partial < 1:
        return torch.zeros(len(t), device=DEVICE)

    p_exp = 2.3 + 0.7 * key_pos
    T_c_base = np.interp(midi, [36, 60, 96], [0.004, 0.0025, 0.0008])
    T_c = T_c_base * (0.75 / 0.8) ** (1.0 / (p_exp + 1))
    hammer_cutoff = 2.5 / T_c * params['hammer_cutoff_scale']
    strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

    spectral_correction = interp_spectral_curve(params['spectral_ctrl'], max_partial)
    decay_correction = interp_spectral_curve(params['decay_ctrl'], max_partial)

    sb_mode_gains = torch.stack([params['sb_mode1_gain'], params['sb_mode2_gain'],
                                  params['sb_mode3_gain']])

    if midi < 36:
        string_detunes = [0.0]
    elif midi < 48:
        dc = 0.3 + 0.3 * key_pos
        string_detunes = [-dc, dc]
    else:
        dc = 0.5 + 1.0 * key_pos
        string_detunes = [-dc, 0.0, dc]
    n_strings = len(string_detunes)

    ns = torch.arange(1, max_partial + 1, dtype=torch.float32, device=DEVICE)
    signal = torch.zeros(len(t), device=DEVICE)

    for s_idx, d_cents in enumerate(string_detunes):
        detune_ratio = 2 ** (d_cents / 1200)
        string_amp = 1.0 / n_strings

        partial_freqs = ns * freq * detune_ratio * torch.sqrt(1 + B * ns ** 2)
        valid = partial_freqs < SAMPLE_RATE / 2

        # Base rolloff from hand-tuned physics
        rolloff_base = 0.7 + 0.6 * key_pos + 4.5 * key_pos ** 3
        amps = 1.0 / (ns ** rolloff_base)

        # Learned spectral correction
        amps = amps * torch.exp(spectral_correction)

        # Hammer (physics)
        amps = amps / (1.0 + (partial_freqs / hammer_cutoff) ** 1.5)
        fTc = partial_freqs * T_c
        denom = 1.0 - 4.0 * fTc * fTc
        safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
        cosine_mod = torch.clamp(torch.abs(torch.cos(np.pi * fTc) / safe_denom), max=1.0)
        cosine_mod = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(cosine_mod), cosine_mod)
        amps = amps * (0.7 + 0.3 * cosine_mod)

        # Strike position (physics)
        strike_factor = torch.abs(torch.sin(np.pi * ns * strike_pos))
        amps = amps * torch.clamp(strike_factor, min=0.03)

        # Soundboard (learned)
        sb_response = torch.ones_like(partial_freqs)
        for idx in range(3):
            sb_response = sb_response + sb_mode_gains[idx] * torch.exp(
                -0.5 * ((partial_freqs - SB_MODE_CFS[idx]) / SB_MODE_BWS[idx]) ** 2)
        sb_response = sb_response + params['sb_bridge_gain'] * torch.exp(
            -0.5 * ((partial_freqs - params['sb_bridge_cf']) / params['sb_bridge_bw']) ** 2)
        amps = amps * sb_response * string_amp

        # Decay (physics + learned correction)
        K_n = (ns ** 2) * piL2
        air_frac = params['air_frac']
        alpha_n = (b1 * (1.0 - air_frac)
                   + b1 * air_frac * torch.sqrt(torch.tensor(freq, device=DEVICE)
                                                  / torch.clamp(partial_freqs, min=20.0))
                   + b2 * K_n)
        alpha_n = alpha_n * torch.exp(decay_correction)

        # Two-stage envelope (hand-tuned defaults + learned offsets)
        prompt_factor = (1.5 + 0.5 * key_pos) + params['prompt_factor_offset']
        A_after_val = (0.15 + 0.05 * key_pos) + params['A_after_offset']
        prompt_rate = alpha_n * prompt_factor * sb_response
        after_rate = alpha_n * params['after_factor']
        A_after_n = torch.clamp(A_after_val / sb_response, max=0.95)
        A_prompt_n = 1.0 - A_after_n

        env = (A_prompt_n.unsqueeze(1) * torch.exp(-prompt_rate.unsqueeze(1) * t.unsqueeze(0))
               + A_after_n.unsqueeze(1) * torch.exp(-after_rate.unsqueeze(1) * t.unsqueeze(0)))

        phases = PHASE_TABLE_GPU[s_idx, :max_partial]
        sines = torch.sin(2 * np.pi * partial_freqs.unsqueeze(1) * t.unsqueeze(0)
                          + phases.unsqueeze(1))

        partials = amps.unsqueeze(1) * env * sines
        partials = partials * valid.unsqueeze(1).float()
        signal = signal + partials.sum(dim=0)

    return signal


def synthesize_noise(midi, params, t):
    """Noise component: filtered noise for hammer/room character.

    Uses frequency-domain filtering: multiply noise spectrum by learned
    spectral shape, then IFFT back.
    """
    freq = midi_to_freq(midi)
    n = len(t)

    noise_level = params['noise_level']
    noise_decay = params['noise_decay']
    noise_tilt = params['noise_tilt']
    noise_bw = params['noise_bandwidth']

    # Temporal envelope: exponential decay
    env = noise_level * torch.exp(-t * noise_decay)

    # Use deterministic noise
    raw_noise = NOISE_TABLE[:n]

    # Apply envelope
    shaped_noise = raw_noise * env

    # Frequency-domain filtering via STFT
    n_fft = 2048
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=DEVICE)

    # STFT of shaped noise
    S = torch.stft(shaped_noise, n_fft=n_fft, hop_length=hop,
                    window=window, return_complex=True)

    # Build spectral filter: peaked around note frequency with learned tilt
    freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
    center = freq * noise_bw
    # Broad filter centered on note's spectral region with tilt
    filter_shape = torch.exp(-0.5 * ((freqs - center) / (center * 0.8 + 100)) ** 2)
    # Add spectral tilt: negative tilt = darker (less HF)
    tilt_factor = torch.exp(noise_tilt * torch.log(freqs / (freq + 1e-8) + 1e-8))
    tilt_factor = tilt_factor / (tilt_factor.max() + 1e-8)
    filter_shape = filter_shape * tilt_factor
    # Ensure low frequencies always pass (body of noise)
    filter_shape = torch.clamp(filter_shape, min=0.01)

    # Apply filter
    S_filtered = S * filter_shape.unsqueeze(1)

    # ISTFT back to time domain
    noise_signal = torch.istft(S_filtered, n_fft=n_fft, hop_length=hop,
                                window=window, length=n)

    return noise_signal


def synthesize_note(midi, params, t=None):
    """Full synthesis: harmonic + noise, with attack shaping."""
    if t is None:
        t = t_gpu

    # Harmonic component (additive synthesis)
    harmonic = synthesize_harmonic(midi, params, t)

    # Noise component (filtered noise)
    noise = synthesize_noise(midi, params, t)

    # Mix
    signal = harmonic + noise

    # Attack shape
    attack_peak_base = np.interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
    attack_peak = attack_peak_base * params['attack_peak_scale']
    attack_env = torch.where(t < attack_peak,
                              0.5 - 0.5 * torch.cos(np.pi * t / torch.clamp(attack_peak, min=1e-4)),
                              torch.ones_like(t))
    signal = signal * attack_env

    # Soundboard IR convolution (per-note spectral coloring)
    signal = apply_soundboard_ir(signal, midi)

    # Fade out
    fade = int(0.1 * SAMPLE_RATE)
    signal = signal.clone()
    signal[-fade:] = signal[-fade:] * torch.linspace(1, 0, fade, device=DEVICE)

    # Normalize
    peak = torch.max(torch.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    return signal


# ═══ LOSS FUNCTIONS ═══

def multi_scale_stft_loss(target, generated):
    loss = torch.tensor(0.0, device=DEVICE)
    for n_fft in [512, 1024, 2048, 4096]:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=DEVICE)
        S_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                    window=window, return_complex=True))
        S_g = torch.abs(torch.stft(generated, n_fft=n_fft, hop_length=hop,
                                    window=window, return_complex=True))
        nf = min(S_t.shape[1], S_g.shape[1])
        S_t, S_g = S_t[:, :nf], S_g[:, :nf]
        sc = torch.norm(S_t - S_g) / (torch.norm(S_t) + 1e-8)
        lm = torch.mean(torch.abs(torch.log(S_t + 1e-7) - torch.log(S_g + 1e-7)))
        loss = loss + sc * 2.0 + lm * 0.5
    return loss


def envelope_loss(target, generated, frame_len=1024, hop=512):
    def rms_frames(x):
        n = (len(x) - frame_len) // hop + 1
        frames = x.unfold(0, frame_len, hop)[:n]
        return torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)
    rms_t = rms_frames(target)
    rms_g = rms_frames(generated)
    n = min(len(rms_t), len(rms_g))
    return torch.mean((rms_t[:n] - rms_g[:n]) ** 2) * 50.0


def centroid_loss(target, generated, n_fft=2048, hop=512):
    window = torch.hann_window(n_fft, device=DEVICE)
    def compute(x):
        S = torch.abs(torch.stft(x, n_fft=n_fft, hop_length=hop,
                                  window=window, return_complex=True))
        freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
        energy = torch.sum(S, dim=0)
        centroid = torch.sum(freqs.unsqueeze(1) * S, dim=0) / (energy + 1e-8)
        return centroid, energy
    c_t, e_t = compute(target)
    c_g, _ = compute(generated)
    n = min(len(c_t), len(c_g))
    weights = e_t[:n] / (e_t[:n].sum() + 1e-8)
    diff = (c_t[:n] - c_g[:n]) / (c_t[:n] + 100.0)
    return torch.sum(weights * diff ** 2) * 10.0


def anchor_regularizer(params):
    """L2 penalty pulling all params toward zero (= hand-tuned defaults).

    Every param is defined as an offset from the hand-tuned grand piano model.
    This regularizer ensures the network only deviates when the audio loss
    strongly justifies it, preserving the hand-tuned brightness and character.
    """
    reg = torch.tensor(0.0, device=DEVICE)
    # Damping offsets
    reg = reg + params['log_b1_offset'] ** 2 + params['log_b2_offset'] ** 2
    # Soundboard offsets from defaults
    reg = reg + ((params['sb_bridge_cf'] - 2500.0) / 500.0) ** 2
    reg = reg + ((params['sb_bridge_bw'] - 1200.0) / 400.0) ** 2
    reg = reg + ((params['sb_bridge_gain'] - 0.3) / 0.15) ** 2
    reg = reg + ((params['sb_mode1_gain'] - 0.15) / 0.10) ** 2
    reg = reg + ((params['sb_mode2_gain'] - 0.12) / 0.08) ** 2
    reg = reg + ((params['sb_mode3_gain'] - 0.10) / 0.06) ** 2
    # Decay offsets
    reg = reg + params['prompt_factor_offset'] ** 2
    reg = reg + ((params['after_factor'] - 0.25) / 0.15) ** 2
    reg = reg + params['A_after_offset'] ** 2
    reg = reg + ((params['air_frac'] - 0.2) / 0.15) ** 2
    # Hammer/attack scale offsets from 1.0
    reg = reg + (params['hammer_cutoff_scale'] - 1.0) ** 2
    reg = reg + (params['attack_peak_scale'] - 1.0) ** 2
    # Spectral and decay corrections (should stay small)
    reg = reg + torch.sum(params['spectral_ctrl'] ** 2)
    reg = reg + torch.sum(params['decay_ctrl'] ** 2)
    return reg * 0.03  # moderate anchor — allows meaningful per-note learning


def compute_loss(target, generated, params=None):
    # No anchor regularizer — tight param bounds keep us near hand-tuned defaults
    return (multi_scale_stft_loss(target, generated)
            + envelope_loss(target, generated)
            + centroid_loss(target, generated))


# ═══ TRAINING ═══

def load_audio(path):
    """Load audio file as mono float32 numpy array at SAMPLE_RATE."""
    import tempfile
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


def load_targets():
    targets = {}
    for midi, name in TRAIN_NOTES:
        path = os.path.join(BASE, 'audio', 'piano', f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        y = load_audio(path)
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        else:
            y = y[:N_SAMPLES]
        targets[(midi, name)] = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        print(f"  {name} (MIDI {midi}): loaded")
    return targets


def train(epochs=2000, lr=2e-3):
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading target samples...")
    targets = load_targets()
    if not targets:
        print("ERROR: No target samples found!")
        return None

    model = PianoParamNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining on {len(targets)} notes for {epochs} epochs")
    print(f"Network: 1→128→128→128→{PianoParamNet.N_PARAMS} ({n_params:,} weights)")
    print(f"Components: harmonic (additive) + noise (filtered)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    def norm_midi(midi):
        return torch.tensor([(midi - 60) / 30.0], dtype=torch.float32, device=DEVICE)

    best_loss = float('inf')
    best_state = None
    start_time = time.time()
    train_keys = list(targets.keys())

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(train_keys)

        for midi, name in train_keys:
            target = targets[(midi, name)]
            params = model(norm_midi(midi))
            generated = synthesize_note(midi, params)
            loss = compute_loss(target, generated, params)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        avg_loss = epoch_loss / len(targets)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - start_time
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:4d}/{epochs}: loss={avg_loss:.4f} "
                  f"(best={best_loss:.4f}) lr={lr_now:.5f} [{elapsed:.0f}s]")

        if (epoch + 1) % 250 == 0:
            model.eval()
            with torch.no_grad():
                note_losses = []
                for midi, name in train_keys:
                    params = model(norm_midi(midi))
                    gen = synthesize_note(midi, params)
                    nl = compute_loss(targets[(midi, name)], gen, params).item()
                    note_losses.append((name, nl))
                note_losses.sort(key=lambda x: -x[1])
                worst3 = ', '.join(f"{n}={l:.3f}" for n, l in note_losses[:3])
                best3 = ', '.join(f"{n}={l:.3f}" for n, l in note_losses[-3:])
                print(f"         Worst: {worst3}  |  Best: {best3}")
            model.train()

    if best_state:
        model.load_state_dict(best_state)

    torch.save({
        'model_state': model.state_dict(),
        'best_loss': best_loss,
        'epochs': epochs,
    }, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Best loss: {best_loss:.4f}")
    return model


# ═══ GENERATION ═══

def write_wav(filename, signal_np):
    pcm = (signal_np * 32767).astype(np.int16)
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


def generate_samples(model):
    out_dir = os.path.join(BASE, 'audio', 'ddsp-piano')
    os.makedirs(out_dir, exist_ok=True)

    def norm_midi(midi):
        return torch.tensor([(midi - 60) / 30.0], dtype=torch.float32, device=DEVICE)

    model.eval()
    print(f"\nGenerating {len(ALL_NOTES)} DDSP Piano samples...")

    with torch.no_grad():
        for midi, name in ALL_NOTES:
            freq = midi_to_freq(midi)
            params = model(norm_midi(midi))
            signal = synthesize_note(midi, params)
            signal_np = signal.cpu().numpy()

            wav_path = os.path.join(out_dir, f'{name}.wav')
            mp3_path = os.path.join(out_dir, f'{name}.mp3')
            write_wav(wav_path, signal_np)
            wav_to_mp3(wav_path, mp3_path)

            if os.path.exists(mp3_path):
                os.remove(wav_path)
                print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz")
            else:
                print(f"  {name} — WARNING: ffmpeg failed, keeping WAV")

    print(f"\nDone! Samples in {out_dir}/")


def compare_with_targets(model):
    print("\n" + "=" * 60)
    print("COMPARISON: DDSP Piano vs Recorded Samples")
    print("=" * 60)

    def norm_midi(midi):
        return torch.tensor([(midi - 60) / 30.0], dtype=torch.float32, device=DEVICE)

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for midi, name in TRAIN_NOTES:
            path = os.path.join(BASE, 'audio', 'piano', f'{name}.mp3')
            if not os.path.exists(path):
                continue
            y = load_audio(path)
            if len(y) < N_SAMPLES:
                y = np.pad(y, (0, N_SAMPLES - len(y)))
            else:
                y = y[:N_SAMPLES]
            target = torch.tensor(y, dtype=torch.float32, device=DEVICE)

            params = model(norm_midi(midi))
            gen = synthesize_note(midi, params)
            loss = compute_loss(target, gen).item()
            total_loss += loss

            window = torch.hann_window(2048, device=DEVICE)
            def centroid_of(x):
                S = torch.abs(torch.stft(x[:SAMPLE_RATE * 2], n_fft=2048, hop_length=512,
                                          window=window, return_complex=True))
                freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
                return (torch.sum(freqs.unsqueeze(1) * S) / (torch.sum(S) + 1e-8)).item()

            c_t = centroid_of(target)
            c_g = centroid_of(gen)
            ratio = c_g / c_t if c_t > 0 else 0
            print(f"  {name:>4s}: loss={loss:.3f}  centroid={c_t:.0f}→{c_g:.0f} Hz (ratio={ratio:.2f})")

    print(f"\n  Average loss: {total_loss / len(TRAIN_NOTES):.3f}")

    # Show noise params for sample notes
    print(f"\n{'─' * 60}")
    print("LEARNED PARAMETERS (sample notes)")
    print(f"{'─' * 60}")
    for midi, name in [(36, 'C2'), (60, 'C4'), (81, 'A5')]:
        params = model(norm_midi(midi))
        sc = params['spectral_ctrl'].detach().cpu().numpy()
        dc = params['decay_ctrl'].detach().cpu().numpy()
        print(f"\n  {name} (MIDI {midi}):")
        print(f"    b1 offset: {params['log_b1_offset'].item():+.3f} "
              f"(x{np.exp(params['log_b1_offset'].item()):.2f})")
        print(f"    b2 offset: {params['log_b2_offset'].item():+.3f}")
        print(f"    prompt_offset: {params['prompt_factor_offset'].item():+.3f}  "
              f"after: {params['after_factor'].item():.3f}  "
              f"A_after_offset: {params['A_after_offset'].item():+.3f}")
        print(f"    noise: level={params['noise_level'].item():.4f}  "
              f"decay={params['noise_decay'].item():.1f}  "
              f"tilt={params['noise_tilt'].item():+.2f}  "
              f"bw={params['noise_bandwidth'].item():.2f}")
        print(f"    spectral: {' '.join(f'{v:+.1f}' for v in sc)}")
        print(f"    decay:    {' '.join(f'{v:+.1f}' for v in dc)}")


if __name__ == '__main__':
    args = sys.argv[1:]
    epochs = 2000
    generate_only = False

    for i, arg in enumerate(args):
        if arg == '--epochs' and i + 1 < len(args):
            epochs = int(args[i + 1])
        elif arg == '--generate-only':
            generate_only = True

    if generate_only:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: No saved model at {MODEL_PATH}")
            sys.exit(1)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model = PianoParamNet().to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model (loss={checkpoint['best_loss']:.4f}, {checkpoint['epochs']} epochs)")
    else:
        model = train(epochs=epochs)
        if model is None:
            sys.exit(1)

    compare_with_targets(model)
    generate_samples(model)
