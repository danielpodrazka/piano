#!/usr/bin/env python3
"""
GPU-accelerated optimization of grand piano synthesis parameters.

Optimizes ~21 heuristic parameters against recorded piano samples using
differential evolution with multi-scale STFT + envelope + centroid loss.
Synthesis is vectorized on GPU (PyTorch) for ~75x speedup over NumPy.

Now includes per-note soundboard IR convolution in the synthesis loop,
so the optimizer focuses on actual physics parameters rather than trying
to compensate for missing spectral coloring.

Usage:
    python3 optimize_grand_piano.py              # 5 target notes
    python3 optimize_grand_piano.py --all-notes  # all 17 targets
"""

import torch
import numpy as np
import os
import sys
import time
import subprocess
import tempfile
import wave
from scipy.optimize import differential_evolution

SAMPLE_RATE = 44100
DURATION = 4.0  # shorter for faster optimization
N_SAMPLES = int(SAMPLE_RATE * DURATION)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)

# Load current phase table from generator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_grand_piano import _PHASE_TABLE
PHASE_TABLE_GPU = torch.tensor(_PHASE_TABLE, dtype=torch.float32, device=DEVICE)

# Load per-note soundboard transfer functions
_TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
_SB_IRS = {}
if os.path.exists(_TF_PATH):
    _tf_data = np.load(_TF_PATH)
    _TF_MIDI = _tf_data['midi_points']
    _TF_MAG = _tf_data['transfer_functions']
    _TF_IR_LEN = int(_tf_data['ir_length'])

    def _min_phase_ir_np(mag_resp, ir_len):
        mag = np.maximum(mag_resp, 1e-10)
        n_fft = (len(mag) - 1) * 2
        full = np.concatenate([mag, mag[-2:0:-1]])
        cep = np.fft.ifft(np.log(full)).real
        mc = np.zeros_like(cep)
        mc[0] = cep[0]
        mc[1:n_fft // 2] = 2 * cep[1:n_fft // 2]
        mc[n_fft // 2] = cep[n_fft // 2]
        ir = np.fft.ifft(np.exp(np.fft.fft(mc))).real[:ir_len]
        ir *= np.hanning(ir_len * 2)[ir_len:]
        e = np.sqrt(np.sum(ir ** 2))
        if e > 0:
            ir = ir / e * np.sqrt(ir_len) / np.sqrt(n_fft)
        return ir

    def get_sb_ir(midi):
        if midi not in _SB_IRS:
            log_tfs = np.log(_TF_MAG + 1e-10)
            interp_log = np.array([np.interp(midi, _TF_MIDI, log_tfs[:, i])
                                   for i in range(log_tfs.shape[1])])
            ir = _min_phase_ir_np(np.exp(interp_log), _TF_IR_LEN)
            _SB_IRS[midi] = torch.tensor(ir.astype(np.float32), device=DEVICE)
        return _SB_IRS[midi]
else:
    get_sb_ir = lambda midi: None

# ═══ TARGET NOTES ═══
TARGET_5 = [
    (36, 'C2'), (45, 'A2'), (60, 'C4'), (69, 'A4'), (81, 'A5'),
]
TARGET_ALL = [
    (36, 'C2'), (39, 'Ds2'), (42, 'Fs2'), (45, 'A2'),
    (48, 'C3'), (51, 'Ds3'), (54, 'Fs3'), (57, 'A3'),
    (60, 'C4'), (63, 'Ds4'), (66, 'Fs4'), (69, 'A4'),
    (72, 'C5'), (75, 'Ds5'), (78, 'Fs5'), (81, 'A5'),
    (84, 'C6'),
]

# ═══ PARAMETER DEFINITIONS ═══
# Current values match generate_grand_piano.py exactly
PARAM_DEFS = [
    # Damping b1 calibration (log-interpolated at MIDI 36/60/96)
    ('b1_low',              0.08,    0.8),    # C2, currently 0.25
    ('b1_mid',              0.35,    3.5),    # C4, currently 1.1
    ('b1_high',             3.0,    25.0),    # C7, currently 9.17

    # Damping b2 calibration
    ('b2_low',              2e-5,    3e-4),   # C2, currently 7.5e-5
    ('b2_mid',              8e-5,    8e-4),   # C4, currently 2.7e-4
    ('b2_high',             7e-4,    7e-3),   # C7, currently 2.1e-3

    # Soundboard bridge hill
    ('sb_bridge_cf',        800,     3500),   # center freq, currently 1800
    ('sb_bridge_bw',        300,     2000),   # bandwidth, currently 800
    ('sb_bridge_gain',      0.1,     0.8),    # gain, currently 0.40

    # Soundboard low modes (gains only)
    ('sb_mode1_gain',       0.03,    0.35),   # 90 Hz mode, currently 0.15
    ('sb_mode2_gain',       0.03,    0.25),   # 170 Hz mode, currently 0.12
    ('sb_mode3_gain',       0.03,    0.25),   # 260 Hz mode, currently 0.10

    # Two-stage decay (Weinreich)
    ('prompt_base',         0.8,     3.0),    # currently 1.5
    ('prompt_slope',        0.0,     2.0),    # currently 0.5
    ('after_factor',        0.1,     0.6),    # currently 0.25
    ('A_after_base',        0.05,    0.35),   # currently 0.15
    ('A_after_slope',       0.0,     0.2),    # currently 0.05

    # Three-term damping
    ('air_frac',            0.05,    0.5),    # currently 0.2

    # Spectral rolloff
    ('rolloff_base',        0.4,     1.5),    # currently 0.9
    ('rolloff_linear',      0.2,     1.5),    # currently 0.8
    ('rolloff_cubic',       1.0,    10.0),    # currently 4.0
]

BOUNDS = [(d[1], d[2]) for d in PARAM_DEFS]
PARAM_NAMES = [d[0] for d in PARAM_DEFS]

CURRENT_PARAMS = [
    0.25,    # b1_low
    1.1,     # b1_mid
    9.17,    # b1_high
    7.5e-5,  # b2_low
    2.7e-4,  # b2_mid
    2.1e-3,  # b2_high
    1800,    # sb_bridge_cf
    800,     # sb_bridge_bw
    0.40,    # sb_bridge_gain
    0.15,    # sb_mode1_gain
    0.12,    # sb_mode2_gain
    0.10,    # sb_mode3_gain
    1.5,     # prompt_base
    0.5,     # prompt_slope
    0.25,    # after_factor
    0.15,    # A_after_base
    0.05,    # A_after_slope
    0.2,     # air_frac
    0.9,     # rolloff_base
    0.8,     # rolloff_linear
    4.0,     # rolloff_cubic
]

# ═══ PHYSICS CONSTANTS (not optimized) ═══
B_MIDI = torch.tensor([21, 33, 45, 57, 69, 84, 96], dtype=torch.float32, device=DEVICE)
B_VALS_LOG = torch.log(torch.tensor([3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2],
                                     dtype=torch.float32, device=DEVICE))
L_MIDI = torch.tensor([36, 60, 96], dtype=torch.float32, device=DEVICE)
L_VALS_LOG = torch.log(torch.tensor([1.92, 0.62, 0.09], dtype=torch.float32, device=DEVICE))
SB_MODE_CFS = torch.tensor([90.0, 170.0, 260.0], device=DEVICE)
SB_MODE_BWS = torch.tensor([30.0, 35.0, 45.0], device=DEVICE)


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def log_interp_gpu(midi, calib_midi, log_vals):
    midi_t = torch.tensor(float(midi), device=DEVICE)
    if midi <= calib_midi[0].item():
        return torch.exp(log_vals[0])
    if midi >= calib_midi[-1].item():
        return torch.exp(log_vals[-1])
    for i in range(len(calib_midi) - 1):
        if midi <= calib_midi[i + 1].item():
            frac = (midi_t - calib_midi[i]) / (calib_midi[i + 1] - calib_midi[i])
            return torch.exp(log_vals[i] + frac * (log_vals[i + 1] - log_vals[i]))
    return torch.exp(log_vals[-1])


def generate_note_gpu(midi, params_dict):
    """Vectorized additive synthesis on GPU, matching generate_grand_piano.py."""
    p = params_dict
    freq = midi_to_freq(midi)
    key_pos = max(0.0, min(1.0, (midi - 21) / 87.0))
    t = t_gpu

    # Damping
    b1_log = torch.log(torch.tensor([p['b1_low'], p['b1_mid'], p['b1_high']],
                                     device=DEVICE))
    b2_log = torch.log(torch.tensor([p['b2_low'], p['b2_mid'], p['b2_high']],
                                     device=DEVICE))
    calib = torch.tensor([36.0, 60.0, 96.0], device=DEVICE)
    b1 = log_interp_gpu(midi, calib, b1_log)
    b2 = log_interp_gpu(midi, calib, b2_log)
    L = log_interp_gpu(midi, L_MIDI, L_VALS_LOG)
    piL2 = (np.pi / L) ** 2
    B = torch.exp(log_interp_gpu(midi, B_MIDI, B_VALS_LOG).log())

    # Max partials
    max_partial = 1
    while max_partial * freq * np.sqrt(1 + B.item() * max_partial ** 2) < SAMPLE_RATE / 2 - 500:
        max_partial += 1
    max_partial = min(max_partial - 1, 64)
    if max_partial < 1:
        return torch.zeros(N_SAMPLES, device=DEVICE)

    # Hammer (not optimized)
    p_exp = 2.3 + 0.7 * key_pos
    T_c_base = np.interp(midi, [36, 60, 96], [0.004, 0.0025, 0.0008])
    velocity = 0.8
    T_c = T_c_base * (0.75 / velocity) ** (1.0 / (p_exp + 1))
    hammer_cutoff = 2.5 / T_c
    strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

    # Two-stage decay
    prompt_factor = p['prompt_base'] + p['prompt_slope'] * key_pos
    after_factor = p['after_factor']
    A_after = p['A_after_base'] + p['A_after_slope'] * key_pos

    # Rolloff
    rolloff = p['rolloff_base'] + p['rolloff_linear'] * key_pos + p['rolloff_cubic'] * key_pos ** 3
    air_frac = p['air_frac']

    # Soundboard
    sb_mode_gains = torch.tensor([p['sb_mode1_gain'], p['sb_mode2_gain'], p['sb_mode3_gain']],
                                  device=DEVICE)
    sb_bridge_cf = p['sb_bridge_cf']
    sb_bridge_bw = p['sb_bridge_bw']
    sb_bridge_gain = p['sb_bridge_gain']

    # Detuning (matching generator exactly)
    if midi < 36:
        string_detunes = [0.0]
    elif midi < 48:
        dc = 0.3 + 0.3 * key_pos
        string_detunes = [-dc, dc]
    else:
        dc = 0.15 + 0.25 * key_pos
        string_detunes = [-dc, 0.0, dc]
    n_strings = len(string_detunes)

    ns = torch.arange(1, max_partial + 1, dtype=torch.float32, device=DEVICE)
    signal = torch.zeros(N_SAMPLES, device=DEVICE)

    for s_idx, d_cents in enumerate(string_detunes):
        detune_ratio = 2 ** (d_cents / 1200)
        string_amp = 1.0 / n_strings
        partial_freqs = ns * freq * detune_ratio * torch.sqrt(1 + B * ns ** 2)
        valid = partial_freqs < SAMPLE_RATE / 2
        if not valid.any():
            continue

        amps = 1.0 / (ns ** rolloff)
        amps = amps * 1.0 / (1.0 + (partial_freqs / hammer_cutoff) ** 1.5)
        fTc = partial_freqs * T_c
        denom = 1.0 - 4.0 * fTc * fTc
        safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
        cosine_mod = torch.clamp(torch.abs(torch.cos(np.pi * fTc) / safe_denom), max=1.0)
        cosine_mod = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(cosine_mod), cosine_mod)
        amps = amps * (0.7 + 0.3 * cosine_mod)

        strike_factor = torch.abs(torch.sin(np.pi * ns * strike_pos))
        amps = amps * torch.clamp(strike_factor, min=0.03)

        sb_response = torch.ones_like(partial_freqs)
        for i in range(3):
            sb_response = sb_response + sb_mode_gains[i] * torch.exp(
                -0.5 * ((partial_freqs - SB_MODE_CFS[i]) / SB_MODE_BWS[i]) ** 2)
        sb_response = sb_response + sb_bridge_gain * torch.exp(
            -0.5 * ((partial_freqs - sb_bridge_cf) / sb_bridge_bw) ** 2)
        amps = amps * sb_response * string_amp

        K_n = (ns ** 2) * piL2
        alpha_n = (b1 * (1.0 - air_frac)
                   + b1 * air_frac * torch.sqrt(torch.tensor(freq, device=DEVICE)
                                                 / torch.clamp(partial_freqs, min=20.0))
                   + b2 * K_n)

        prompt_rate = alpha_n * prompt_factor * sb_response
        after_rate = alpha_n * after_factor
        A_after_n = torch.clamp(A_after / sb_response, max=0.95)
        A_prompt_n = 1.0 - A_after_n

        env = (A_prompt_n.unsqueeze(1) * torch.exp(-prompt_rate.unsqueeze(1) * t.unsqueeze(0))
               + A_after_n.unsqueeze(1) * torch.exp(-after_rate.unsqueeze(1) * t.unsqueeze(0)))

        phases = PHASE_TABLE_GPU[s_idx % 3, :max_partial]
        sines = torch.sin(2 * np.pi * partial_freqs.unsqueeze(1) * t.unsqueeze(0)
                          + phases.unsqueeze(1))

        partials = amps.unsqueeze(1) * env * sines
        partials = partials * valid.unsqueeze(1).float()
        signal = signal + partials.sum(dim=0)

    # Attack shape
    attack_peak = np.interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
    attack_env = torch.where(t < attack_peak,
                              0.5 - 0.5 * torch.cos(np.pi * t / attack_peak),
                              torch.ones_like(t))
    signal = signal * attack_env

    # Per-note soundboard IR convolution
    sb_ir = get_sb_ir(midi)
    if sb_ir is not None:
        ir = sb_ir.unsqueeze(0).unsqueeze(0)
        sig = signal.unsqueeze(0).unsqueeze(0)
        pad_len = len(sb_ir) - 1
        wet = torch.nn.functional.conv1d(
            torch.nn.functional.pad(sig, (pad_len, 0)), ir
        ).squeeze()[:N_SAMPLES]
        dry_rms = (signal ** 2).mean().sqrt() + 1e-10
        wet_rms = (wet ** 2).mean().sqrt() + 1e-10
        wet = wet * dry_rms / wet_rms
        signal = 0.3 * signal + 0.7 * wet

    # Fade out
    fade = int(0.1 * SAMPLE_RATE)
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
        return torch.sqrt(torch.mean(frames ** 2, dim=1))
    rms_t = rms_frames(target)
    rms_g = rms_frames(generated)
    n = min(len(rms_t), len(rms_g))
    return torch.mean((rms_t[:n] - rms_g[:n]) ** 2) * 50.0


def centroid_loss(target, generated, n_fft=2048, hop=512):
    window = torch.hann_window(n_fft, device=DEVICE)
    def compute_centroid_and_energy(x):
        S = torch.abs(torch.stft(x, n_fft=n_fft, hop_length=hop,
                                  window=window, return_complex=True))
        freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
        energy = torch.sum(S, dim=0)
        centroid = torch.sum(freqs.unsqueeze(1) * S, dim=0) / (energy + 1e-8)
        return centroid, energy
    c_t, e_t = compute_centroid_and_energy(target)
    c_g, _ = compute_centroid_and_energy(generated)
    n = min(len(c_t), len(c_g))
    weights = e_t[:n] / (e_t[:n].sum() + 1e-8)
    diff = (c_t[:n] - c_g[:n]) / (c_t[:n] + 100.0)
    return torch.sum(weights * diff ** 2) * 10.0


def compute_loss_gpu(target, generated):
    return (multi_scale_stft_loss(target, generated)
            + envelope_loss(target, generated)
            + centroid_loss(target, generated)).item()


# ═══ REFERENCE LOADING ═══

def load_mp3(path):
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


# ═══ OBJECTIVE ═══

targets_gpu = {}


def objective(params):
    params_dict = dict(zip(PARAM_NAMES, params))
    total_loss = 0.0
    for midi, name in TARGET_NOTES:
        try:
            gen = generate_note_gpu(midi, params_dict)
            total_loss += compute_loss_gpu(targets_gpu[name], gen)
        except Exception:
            return 1e6
    return total_loss / len(TARGET_NOTES)


if __name__ == '__main__':
    use_all = '--all-notes' in sys.argv
    TARGET_NOTES = TARGET_ALL if use_all else TARGET_5

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading {len(TARGET_NOTES)} target piano samples...")
    for midi, name in TARGET_NOTES:
        path = os.path.join(BASE, 'audio', 'piano', f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        y = load_mp3(path)
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        else:
            y = y[:N_SAMPLES]
        targets_gpu[name] = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        print(f"  {name} (MIDI {midi}): loaded")

    TARGET_NOTES = [(m, n) for m, n in TARGET_NOTES if n in targets_gpu]
    print(f"\nOptimizing against {len(TARGET_NOTES)} notes: {', '.join(n for _, n in TARGET_NOTES)}")

    print("\nBenchmarking...")
    current_loss = objective(CURRENT_PARAMS)
    print(f"Current hand-tuned loss: {current_loss:.4f}")

    t0 = time.time()
    for _ in range(3):
        objective(CURRENT_PARAMS)
    eval_time = (time.time() - t0) / 3
    print(f"Eval time: {eval_time * 1000:.0f}ms per evaluation")

    total_evals = 300 * 20
    est_time = total_evals * eval_time
    print(f"Estimated optimization time: {est_time / 60:.1f} minutes ({total_evals} evals)")

    print(f"\nOptimizing {len(PARAM_DEFS)} parameters with differential evolution...")
    print(f"{'─' * 70}")

    best_loss_so_far = [current_loss]
    iter_count = [0]
    start_time = time.time()

    def callback(xk, convergence):
        iter_count[0] += 1
        loss = objective(xk)
        elapsed = time.time() - start_time
        if loss < best_loss_so_far[0]:
            improvement = (1 - loss / current_loss) * 100
            print(f"\n  ★ Gen {iter_count[0]} ({elapsed:.0f}s): loss={loss:.4f} "
                  f"({improvement:+.1f}% vs hand-tuned)")
            best_loss_so_far[0] = loss
            changed = []
            for n, v in zip(PARAM_NAMES, xk):
                cur = CURRENT_PARAMS[PARAM_NAMES.index(n)]
                if abs(v - cur) / (abs(cur) + 1e-8) > 0.05:
                    changed.append(f"    {n:>20s}: {cur:.6g} → {v:.6g}")
            if changed:
                print('\n'.join(changed[:8]))
                if len(changed) > 8:
                    print(f"    ... and {len(changed) - 8} more")
        elif iter_count[0] % 20 == 0:
            print(f"  Gen {iter_count[0]} ({elapsed:.0f}s): "
                  f"best={best_loss_so_far[0]:.4f}, convergence={convergence:.4f}")

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        maxiter=300,
        popsize=20,
        tol=1e-4,
        seed=42,
        workers=1,
        callback=callback,
        disp=True,
        x0=CURRENT_PARAMS,
        init='sobol',
        mutation=(0.5, 1.5),
        recombination=0.8,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"Best loss: {result.fun:.4f} (was {current_loss:.4f})")
    print(f"Improvement: {(1 - result.fun / current_loss) * 100:.1f}%")
    print(f"Generations: {result.nit}, Evaluations: {result.nfev}")

    print(f"\n{'─' * 70}")
    print("OPTIMIZED PARAMETERS")
    print(f"{'─' * 70}")
    print(f"{'Parameter':>22s}  {'Old':>12s}  {'New':>12s}  {'Change':>8s}")
    print(f"{'─' * 22}  {'─' * 12}  {'─' * 12}  {'─' * 8}")
    for name, val in zip(PARAM_NAMES, result.x):
        cur = CURRENT_PARAMS[PARAM_NAMES.index(name)]
        pct = (val - cur) / (abs(cur) + 1e-8) * 100
        mark = ' ◄' if abs(pct) > 5 else ''
        print(f"{name:>22s}  {cur:>12.6g}  {val:>12.6g}  {pct:>+7.1f}%{mark}")

    # Paste-ready output
    print(f"\n{'─' * 70}")
    print("PASTE INTO generate_grand_piano.py:")
    print(f"# ML-optimized (loss: {result.fun:.4f}, was {current_loss:.4f})")
    print(f"{'─' * 70}")

    b1_low, b1_mid, b1_high = result.x[0:3]
    b2_low, b2_mid, b2_high = result.x[3:6]
    print(f"CALIB_NOTES = {{")
    print(f"    36: {{'b1': {b1_low:.4f},  'b2': {b2_low:.4e},  'L': 1.92}},  # C2")
    print(f"    60: {{'b1': {b1_mid:.4f},  'b2': {b2_mid:.4e},  'L': 0.62}},  # C4")
    print(f"    96: {{'b1': {b1_high:.4f},  'b2': {b2_high:.4e},  'L': 0.09}},  # C7")
    print(f"}}")

    sb_cf, sb_bw, sb_gain = result.x[6:9]
    m1g, m2g, m3g = result.x[9:12]
    print(f"\n# Soundboard spectral envelope:")
    print(f"#   Low modes: (90, 30, {m1g:.3f}), (170, 35, {m2g:.3f}), (260, 45, {m3g:.3f})")
    print(f"#   Bridge hill: center={sb_cf:.0f} Hz, BW={sb_bw:.0f} Hz, gain={sb_gain:.3f}")

    pb, ps = result.x[12:14]
    af = result.x[14]
    aab, aas = result.x[15:17]
    print(f"\n# Two-stage decay:")
    print(f"#   prompt_factor = {pb:.4f} + {ps:.4f} * key_pos")
    print(f"#   after_factor = {af:.4f}")
    print(f"#   A_after = {aab:.4f} + {aas:.4f} * key_pos")

    air = result.x[17]
    print(f"\n# air_frac = {air:.4f}")

    rb, rl, rc = result.x[18:21]
    print(f"\n# rolloff = {rb:.4f} + {rl:.4f} * key_pos + {rc:.4f} * key_pos ** 3")
