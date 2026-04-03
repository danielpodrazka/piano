#!/usr/bin/env python3
"""
GPU-accelerated optimization of FM Rhodes parameters to match sampled Rhodes.

Runs all synthesis and loss computation on CUDA (RTX 4080).
Uses differential evolution with batched GPU evaluation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
import time
from scipy.optimize import differential_evolution

SAMPLE_RATE = 44100
DURATION = 4.0
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_NOTES = [
    (50, 'D3'),
    (62, 'D4'),
    (71, 'B4'),
]

PARAM_DEFS = [
    ('body_mod_idx',       0.5,  4.0),
    ('mod_decay_base',     0.3,  5.0),
    ('mod_decay_kscale',   0.0,  3.0),
    ('tine_ratio',         3.0, 20.0),
    ('tine_mod_idx',       0.1,  3.0),
    ('tine_mod_decay',     2.0, 25.0),
    ('tine_carrier_decay', 1.0, 15.0),
    ('tine_level',         0.02, 0.40),
    ('sub_level',          0.0,  0.30),
    ('h2_level',           0.0,  0.25),
    ('h3_level',           0.0,  0.15),
    ('body_mix',           0.30, 0.85),
    ('drive_base',         0.1,  3.0),
    ('drive_kscale',       0.0,  2.0),
    ('asym',               0.2,  1.0),
    ('decay_rate_base',    0.2,  1.5),
    ('decay_rate_kscale',  0.0,  0.03),
]

BOUNDS = [(d[1], d[2]) for d in PARAM_DEFS]
PARAM_NAMES = [d[0] for d in PARAM_DEFS]

# Pre-compute time vector on GPU
N_SAMPLES = int(SAMPLE_RATE * DURATION)
t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def generate_note_gpu(midi, params_dict):
    """Generate FM Rhodes note entirely on GPU."""
    p = params_dict
    freq = midi_to_freq(midi)
    t = t_gpu
    n = N_SAMPLES

    key_scale = max(0.25, min(1.0, 1.0 - (midi - 40) / 60))

    # Amplitude envelope
    attack = torch.clamp(t / 0.001, max=1.0)
    decay_rate = p['decay_rate_base'] + (midi - 35) * p['decay_rate_kscale']
    env = attack * torch.exp(-t * decay_rate)

    # Body (1:1 FM)
    body_mod_idx = p['body_mod_idx'] * 0.75 * key_scale
    mod_decay_rate = p['mod_decay_base'] + (1.0 - key_scale) * p['mod_decay_kscale']
    body_mod_env = body_mod_idx * torch.exp(-t * mod_decay_rate)
    phase = 2 * np.pi * freq * t
    body_mod = body_mod_env * torch.sin(phase)
    body = torch.sin(phase + body_mod)

    # Tine
    tine_mod_freq = freq * p['tine_ratio']
    if tine_mod_freq < SAMPLE_RATE / 2 - 1000:
        tine_mod_idx = p['tine_mod_idx'] * 0.75
        tine_mod_env = tine_mod_idx * torch.exp(-t * p['tine_mod_decay'])
        tine_phase = 2 * np.pi * tine_mod_freq * t
        tine_mod = tine_mod_env * torch.sin(tine_phase)
        tine_carrier_env = torch.exp(-t * p['tine_carrier_decay'])
        tine = tine_carrier_env * torch.sin(phase + tine_mod)
    else:
        tine = torch.zeros(n, device=DEVICE)

    # Sub-harmonic
    sub = p['sub_level'] * key_scale * torch.sin(np.pi * freq * t)

    # Additive harmonics
    h2 = p['h2_level'] * torch.sin(4 * np.pi * freq * t)
    h3 = p['h3_level'] * torch.sin(6 * np.pi * freq * t)

    # Mix
    signal = body * p['body_mix'] + tine * p['tine_level'] + sub + h2 + h3
    signal = signal * env

    # Pickup distortion
    drive = p['drive_base'] + p['drive_kscale'] * key_scale ** 2
    asym = p['asym']
    drive_c = max(drive, 0.01)
    asym_c = max(drive * asym, 0.01)
    pos = torch.clamp(signal, min=0)
    neg = torch.clamp(signal, max=0)
    signal = torch.tanh(pos * drive_c) / np.tanh(drive_c) + torch.tanh(neg * asym_c) / np.tanh(asym_c)

    # Fade out
    fade = int(0.05 * SAMPLE_RATE)
    fade_env = torch.linspace(1, 0, fade, device=DEVICE)
    signal[-fade:] *= fade_env

    # Normalize
    peak = torch.max(torch.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    return signal


def multi_scale_stft_loss(target, generated):
    """Multi-scale STFT loss computed on GPU."""
    loss = torch.tensor(0.0, device=DEVICE)

    for n_fft in [512, 1024, 2048]:
        hop = n_fft // 4
        # Compute STFT using torch
        S_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                    window=torch.hann_window(n_fft, device=DEVICE),
                                    return_complex=True))
        S_g = torch.abs(torch.stft(generated, n_fft=n_fft, hop_length=hop,
                                    window=torch.hann_window(n_fft, device=DEVICE),
                                    return_complex=True))

        # Trim to same size
        nf = min(S_t.shape[1], S_g.shape[1])
        S_t = S_t[:, :nf]
        S_g = S_g[:, :nf]

        # Spectral convergence
        sc = torch.norm(S_t - S_g) / (torch.norm(S_t) + 1e-8)

        # Log-magnitude loss
        log_t = torch.log(S_t + 1e-7)
        log_g = torch.log(S_g + 1e-7)
        lm = torch.mean(torch.abs(log_t - log_g))

        loss += sc * 2.0 + lm * 0.5

    return loss


def envelope_loss(target, generated, frame_len=1024, hop=512):
    """RMS envelope trajectory loss on GPU."""
    def rms_frames(x):
        # Unfold into frames
        n = (len(x) - frame_len) // hop + 1
        frames = x.unfold(0, frame_len, hop)[:n]
        return torch.sqrt(torch.mean(frames ** 2, dim=1))

    rms_t = rms_frames(target)
    rms_g = rms_frames(generated)
    n = min(len(rms_t), len(rms_g))
    return torch.mean((rms_t[:n] - rms_g[:n]) ** 2) * 50.0


def centroid_loss(target, generated, n_fft=2048, hop=512):
    """Spectral centroid trajectory loss on GPU."""
    window = torch.hann_window(n_fft, device=DEVICE)

    def compute_centroid(x):
        S = torch.abs(torch.stft(x, n_fft=n_fft, hop_length=hop,
                                  window=window, return_complex=True))
        freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
        centroid = torch.sum(freqs.unsqueeze(1) * S, dim=0) / (torch.sum(S, dim=0) + 1e-8)
        return centroid

    c_t = compute_centroid(target)
    c_g = compute_centroid(generated)
    n = min(len(c_t), len(c_g))
    return torch.mean(((c_t[:n] - c_g[:n]) / (c_t[:n] + 1e-8)) ** 2) * 10.0


def compute_loss_gpu(target, generated):
    """Total perceptual loss on GPU."""
    loss = multi_scale_stft_loss(target, generated)
    loss += envelope_loss(target, generated)
    loss += centroid_loss(target, generated)
    return loss.item()


def objective(params):
    """Total loss across all target notes."""
    params_dict = dict(zip(PARAM_NAMES, params))
    total_loss = 0.0
    for midi, name in TARGET_NOTES:
        try:
            gen = generate_note_gpu(midi, params_dict)
            total_loss += compute_loss_gpu(targets_gpu[name], gen)
        except Exception:
            return 1e6
    return total_loss / len(TARGET_NOTES)


CURRENT_PARAMS = [
    2.0,   # body_mod_idx
    1.8,   # mod_decay_base
    0.5,   # mod_decay_kscale
    14.0,  # tine_ratio
    1.2,   # tine_mod_idx
    6.0,   # tine_mod_decay
    3.0,   # tine_carrier_decay
    0.18,  # tine_level
    0.18,  # sub_level
    0.10,  # h2_level
    0.05,  # h3_level
    0.58,  # body_mix
    0.3,   # drive_base
    0.2,   # drive_kscale
    0.5,   # asym
    0.5,   # decay_rate_base
    0.01,  # decay_rate_kscale
]


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading target Rhodes samples...")
    targets_gpu = {}
    for midi, name in TARGET_NOTES:
        path = os.path.join(BASE, 'audio', 'rhodes', f'{name}.mp3')
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        y = y[:N_SAMPLES]
        # Pad if shorter
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        targets_gpu[name] = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        print(f"  {name}: {len(y)} samples → GPU")

    # Evaluate current hand-tuned params
    current_loss = objective(CURRENT_PARAMS)
    print(f"\nCurrent hand-tuned loss: {current_loss:.4f}")

    # Benchmark one evaluation
    t0 = time.time()
    for _ in range(10):
        objective(CURRENT_PARAMS)
    eval_time = (time.time() - t0) / 10
    print(f"Eval time: {eval_time*1000:.0f}ms per evaluation")

    total_evals = 200 * 20  # maxiter * popsize (approx)
    est_time = total_evals * eval_time
    print(f"Estimated total time: {est_time/60:.0f} minutes ({total_evals} evals)")

    print(f"\nOptimizing {len(PARAM_DEFS)} parameters with differential evolution...")

    best_loss_so_far = [current_loss]
    iter_count = [0]
    start_time = time.time()

    def callback(xk, convergence):
        iter_count[0] += 1
        loss = objective(xk)
        elapsed = time.time() - start_time
        if loss < best_loss_so_far[0]:
            improvement = (1 - loss / current_loss) * 100
            print(f"\n  ★ Gen {iter_count[0]} ({elapsed:.0f}s): loss={loss:.4f} ({improvement:+.1f}% vs hand-tuned)")
            best_loss_so_far[0] = loss
            for n, v in zip(PARAM_NAMES, xk):
                cur = CURRENT_PARAMS[PARAM_NAMES.index(n)]
                if abs(v - cur) / (abs(cur) + 1e-8) > 0.05:
                    print(f"    {n:>22s}: {cur:.4f} → {v:.4f}")
        elif iter_count[0] % 10 == 0:
            print(f"  Gen {iter_count[0]} ({elapsed:.0f}s): best={best_loss_so_far[0]:.4f}, convergence={convergence:.4f}")

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        maxiter=200,
        popsize=15,
        tol=1e-4,
        seed=42,
        workers=1,  # GPU handles parallelism, keep CPU free
        callback=callback,
        disp=True,
        x0=CURRENT_PARAMS,
        init='sobol',
        mutation=(0.5, 1.5),
        recombination=0.8,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"Best loss: {result.fun:.4f} (was {current_loss:.4f})")
    print(f"Improvement: {(1 - result.fun/current_loss)*100:.1f}%")
    print(f"Generations: {result.nit}, Evaluations: {result.nfev}")
    print(f"\nOptimal parameters:")
    for name, val in zip(PARAM_NAMES, result.x):
        cur = CURRENT_PARAMS[PARAM_NAMES.index(name)]
        changed = abs(val - cur) / (abs(cur) + 1e-8) > 0.05
        mark = '  ◄' if changed else ''
        print(f"  {name:>22s} = {val:.6f}  (was {cur:.4f}){mark}")

    print(f"\n# Paste into generate_rhodes_fm.py:")
    print(f"# Loss: {result.fun:.4f} (hand-tuned was {current_loss:.4f})")
    for name, val in zip(PARAM_NAMES, result.x):
        print(f"# {name} = {val:.6f}")
