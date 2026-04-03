#!/usr/bin/env python3
"""
Iterative per-parameter fine-tuning of grand piano synthesis.

Step 1: Rolloff (warmth) — sweeps rolloff_base, rolloff_linear, rolloff_cubic
to find values that better match recorded samples while keeping the character.

Uses GPU synthesis from optimize_grand_piano.py for fast evaluation.
"""

import torch
import numpy as np
import librosa
import os
import time
import itertools

SAMPLE_RATE = 44100
DURATION = 6.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)
PHASE_TABLE_GPU = torch.tensor(
    np.random.RandomState(6454).uniform(0, 2 * np.pi, (3, 64)),
    dtype=torch.float32, device=DEVICE,
)

# All 17 recorded references
TARGET_NOTES = [
    (36, 'C2'), (39, 'Ds2'), (42, 'Fs2'), (45, 'A2'),
    (48, 'C3'), (51, 'Ds3'), (54, 'Fs3'), (57, 'A3'),
    (60, 'C4'), (63, 'Ds4'), (66, 'Fs4'), (69, 'A4'),
    (72, 'C5'), (75, 'Ds5'), (78, 'Fs5'), (81, 'A5'),
    (84, 'C6'),
]

# Physics constants
B_MIDI = torch.tensor([21, 33, 45, 57, 69, 84, 96], dtype=torch.float32, device=DEVICE)
B_VALS_LOG = torch.log(torch.tensor(
    [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2],
    dtype=torch.float32, device=DEVICE))
L_MIDI = torch.tensor([36, 60, 96], dtype=torch.float32, device=DEVICE)
L_VALS_LOG = torch.log(torch.tensor([1.92, 0.62, 0.09], dtype=torch.float32, device=DEVICE))
SB_MODE_CFS = torch.tensor([90.0, 170.0, 260.0], device=DEVICE)
SB_MODE_BWS = torch.tensor([30.0, 35.0, 45.0], device=DEVICE)


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


def generate_note_gpu(midi, rolloff_base, rolloff_linear, rolloff_cubic,
                      sb_bridge_cf=2500, sb_bridge_bw=1200, sb_bridge_gain=0.3,
                      prompt_base=1.5, prompt_slope=0.5, after_factor_val=0.25,
                      a_after_base=0.15, a_after_slope=0.05):
    """GPU synthesis matching generate_grand_piano.py but with tunable params."""
    freq = midi_to_freq(midi)
    key_pos = max(0.0, min(1.0, (midi - 21) / 87.0))
    t = t_gpu

    # Physics params
    b1_log = torch.log(torch.tensor([0.25, 1.1, 9.17], device=DEVICE))
    b2_log = torch.log(torch.tensor([7.5e-5, 2.7e-4, 2.1e-3], device=DEVICE))
    calib = torch.tensor([36.0, 60.0, 96.0], device=DEVICE)
    b1 = log_interp(midi, calib, b1_log)
    b2 = log_interp(midi, calib, b2_log)
    L = log_interp(midi, L_MIDI, L_VALS_LOG)
    piL2 = (np.pi / L) ** 2
    B = log_interp(midi, B_MIDI, B_VALS_LOG)

    max_partial = 1
    while max_partial * freq * np.sqrt(1 + B.item() * max_partial**2) < SAMPLE_RATE / 2 - 500:
        max_partial += 1
    max_partial = min(max_partial - 1, 64)
    if max_partial < 1:
        return torch.zeros(N_SAMPLES, device=DEVICE)

    # Hammer
    p_exp = 2.3 + 0.7 * key_pos
    T_c_base = np.interp(midi, [36, 60, 96], [0.004, 0.0025, 0.0008])
    T_c = T_c_base * (0.75 / 0.8) ** (1.0 / (p_exp + 1))
    hammer_cutoff = 2.5 / T_c
    strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

    # Two-stage decay
    prompt_factor = prompt_base + prompt_slope * key_pos
    after_factor = after_factor_val
    A_after = a_after_base + a_after_slope * key_pos

    # Rolloff — THE PARAMETER BEING TUNED
    rolloff = rolloff_base + rolloff_linear * key_pos + rolloff_cubic * key_pos ** 3

    # Strings
    if midi < 36:
        string_detunes = [0.0]
    elif midi < 48:
        string_detunes = [-(0.3 + 0.3 * key_pos), (0.3 + 0.3 * key_pos)]
    else:
        dc = 0.5 + 1.0 * key_pos
        string_detunes = [-dc, 0.0, dc]
    n_strings = len(string_detunes)

    ns = torch.arange(1, max_partial + 1, dtype=torch.float32, device=DEVICE)
    signal = torch.zeros(N_SAMPLES, device=DEVICE)

    sb_mode_gains = torch.tensor([0.15, 0.12, 0.10], device=DEVICE)

    for s_idx, d_cents in enumerate(string_detunes):
        detune_ratio = 2 ** (d_cents / 1200)
        string_amp = 1.0 / n_strings

        partial_freqs = ns * freq * detune_ratio * torch.sqrt(1 + B * ns ** 2)
        valid = partial_freqs < SAMPLE_RATE / 2

        amps = 1.0 / (ns ** rolloff)

        # Hammer
        amps = amps / (1.0 + (partial_freqs / hammer_cutoff) ** 1.5)
        fTc = partial_freqs * T_c
        denom = 1.0 - 4.0 * fTc * fTc
        safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
        cosine_mod = torch.clamp(torch.abs(torch.cos(np.pi * fTc) / safe_denom), max=1.0)
        cosine_mod = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(cosine_mod), cosine_mod)
        amps = amps * (0.7 + 0.3 * cosine_mod)

        # Strike
        strike_factor = torch.abs(torch.sin(np.pi * ns * strike_pos))
        amps = amps * torch.clamp(strike_factor, min=0.03)

        # Soundboard
        sb_response = torch.ones_like(partial_freqs)
        for i in range(3):
            sb_response = sb_response + sb_mode_gains[i] * torch.exp(
                -0.5 * ((partial_freqs - SB_MODE_CFS[i]) / SB_MODE_BWS[i]) ** 2)
        sb_response = sb_response + sb_bridge_gain * torch.exp(
            -0.5 * ((partial_freqs - sb_bridge_cf) / sb_bridge_bw) ** 2)
        amps = amps * sb_response * string_amp

        # Decay
        K_n = (ns ** 2) * piL2
        air_frac = 0.2
        alpha_n = (b1 * (1.0 - air_frac)
                   + b1 * air_frac * torch.sqrt(torch.tensor(freq, device=DEVICE)
                                                  / torch.clamp(partial_freqs, min=20.0))
                   + b2 * K_n)

        prompt_rate = alpha_n * prompt_factor * sb_response
        after_rate = alpha_n * after_factor
        A_after_n = torch.clamp(torch.tensor(A_after, device=DEVICE) / sb_response, max=0.95)
        A_prompt_n = 1.0 - A_after_n

        env = (A_prompt_n.unsqueeze(1) * torch.exp(-prompt_rate.unsqueeze(1) * t.unsqueeze(0))
               + A_after_n.unsqueeze(1) * torch.exp(-after_rate.unsqueeze(1) * t.unsqueeze(0)))

        phases = PHASE_TABLE_GPU[s_idx, :max_partial]
        sines = torch.sin(2 * np.pi * partial_freqs.unsqueeze(1) * t.unsqueeze(0)
                          + phases.unsqueeze(1))

        partials = amps.unsqueeze(1) * env * sines
        partials = partials * valid.unsqueeze(1).float()
        signal = signal + partials.sum(dim=0)

    # Attack
    attack_peak = np.interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
    attack_env = torch.where(t < attack_peak,
                              0.5 - 0.5 * torch.cos(np.pi * t / attack_peak),
                              torch.ones_like(t))
    signal = signal * attack_env

    # Fade
    fade = int(0.1 * SAMPLE_RATE)
    signal[-fade:] = signal[-fade:] * torch.linspace(1, 0, fade, device=DEVICE)

    # Normalize
    peak = torch.max(torch.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    return signal


# Loss functions
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


def compute_loss(target, generated):
    return (multi_scale_stft_loss(target, generated)
            + envelope_loss(target, generated)
            + centroid_loss(target, generated)).item()


def centroid_of(x, window):
    S = torch.abs(torch.stft(x[:SAMPLE_RATE * 2], n_fft=2048, hop_length=512,
                              window=window, return_complex=True))
    freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
    return (torch.sum(freqs.unsqueeze(1) * S) / (torch.sum(S) + 1e-8)).item()


if __name__ == '__main__':
    import sys

    # What to tune: rolloff, bridge, or damping
    param_group = sys.argv[1] if len(sys.argv) > 1 else 'rolloff'

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading target samples...")
    targets = {}
    for midi, name in TARGET_NOTES:
        path = os.path.join(BASE, 'audio', 'piano', f'{name}.mp3')
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        else:
            y = y[:N_SAMPLES]
        targets[(midi, name)] = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    print(f"  Loaded {len(targets)} samples")

    # Current hand-tuned values
    CURRENT = {'rolloff_base': 0.7, 'rolloff_linear': 0.6, 'rolloff_cubic': 4.5,
               'sb_bridge_cf': 2500, 'sb_bridge_bw': 1200, 'sb_bridge_gain': 0.3}

    window = torch.hann_window(2048, device=DEVICE)

    if param_group == 'rolloff':
        print("\n" + "=" * 70)
        print("STEP 1: ROLLOFF (warmth)")
        print("Current: rolloff = 0.7 + 0.6*kp + 4.5*kp^3")
        print("=" * 70)

        # Sweep: increase base (warmer), adjust linear and cubic
        candidates = []
        bases = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        linears = [0.4, 0.6, 0.8]
        cubics = [3.0, 4.0, 4.5, 5.0]

        print(f"\nSweeping {len(bases)*len(linears)*len(cubics)} combinations...")
        t0 = time.time()

        for rb, rl, rc in itertools.product(bases, linears, cubics):
            total_loss = 0
            for midi, name in TARGET_NOTES:
                gen = generate_note_gpu(midi, rb, rl, rc)
                total_loss += compute_loss(targets[(midi, name)], gen)
            avg = total_loss / len(TARGET_NOTES)
            candidates.append((avg, rb, rl, rc))

        elapsed = time.time() - t0
        candidates.sort()
        print(f"Done in {elapsed:.0f}s")

        # Evaluate current
        current_loss = 0
        for midi, name in TARGET_NOTES:
            gen = generate_note_gpu(midi, 0.7, 0.6, 4.5)
            current_loss += compute_loss(targets[(midi, name)], gen)
        current_loss /= len(TARGET_NOTES)

        print(f"\n{'Rank':>4s}  {'Base':>5s}  {'Lin':>5s}  {'Cub':>5s}  {'Loss':>8s}  {'vs current':>10s}")
        print("-" * 50)
        print(f" cur   0.70   0.60   4.50  {current_loss:8.3f}  {'baseline':>10s}")
        for i, (loss, rb, rl, rc) in enumerate(candidates[:10]):
            delta = (loss - current_loss) / current_loss * 100
            mark = " <-- best" if i == 0 else ""
            print(f"  {i+1:2d}   {rb:.2f}   {rl:.2f}   {rc:.2f}  {loss:8.3f}  {delta:+8.1f}%{mark}")

        # Show per-note centroid comparison for best vs current
        best_loss, best_rb, best_rl, best_rc = candidates[0]
        print(f"\n{'─' * 70}")
        print(f"CENTROID COMPARISON: current vs best ({best_rb}, {best_rl}, {best_rc})")
        print(f"{'─' * 70}")
        print(f"{'Note':>5s}  {'Target':>7s}  {'Current':>8s}  {'Ratio':>6s}  {'Best':>8s}  {'Ratio':>6s}")
        for midi, name in TARGET_NOTES:
            target = targets[(midi, name)]
            c_t = centroid_of(target, window)
            gen_cur = generate_note_gpu(midi, 0.7, 0.6, 4.5)
            c_cur = centroid_of(gen_cur, window)
            gen_best = generate_note_gpu(midi, best_rb, best_rl, best_rc)
            c_best = centroid_of(gen_best, window)
            r_cur = c_cur / c_t if c_t > 0 else 0
            r_best = c_best / c_t if c_t > 0 else 0
            better = " *" if abs(r_best - 1.0) < abs(r_cur - 1.0) else ""
            print(f"  {name:>4s}  {c_t:7.0f}  {c_cur:8.0f}  {r_cur:5.2f}x  {c_best:8.0f}  {r_best:5.2f}x{better}")

        print(f"\nTo apply best rolloff, update generate_grand_piano.py line 167:")
        print(f"  rolloff = {best_rb} + {best_rl} * key_pos + {best_rc} * key_pos ** 3")

    elif param_group == 'bridge':
        print("\n" + "=" * 70)
        print("STEP 2: BRIDGE HILL (brightness character)")
        print("Current: center=2500 Hz, BW=1200 Hz, gain=0.3")
        print("=" * 70)

        # Read current rolloff from argv or use defaults
        rb = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
        rl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
        rc = float(sys.argv[4]) if len(sys.argv) > 4 else 4.5

        candidates = []
        cfs = [1800, 2000, 2200, 2500, 2800]
        bws = [800, 1000, 1200, 1500]
        gains = [0.15, 0.20, 0.25, 0.30, 0.35]

        print(f"Using rolloff=({rb}, {rl}, {rc})")
        print(f"Sweeping {len(cfs)*len(bws)*len(gains)} bridge combinations...")
        t0 = time.time()

        for cf, bw, g in itertools.product(cfs, bws, gains):
            total_loss = 0
            for midi, name in TARGET_NOTES:
                gen = generate_note_gpu(midi, rb, rl, rc,
                                        sb_bridge_cf=cf, sb_bridge_bw=bw, sb_bridge_gain=g)
                total_loss += compute_loss(targets[(midi, name)], gen)
            avg = total_loss / len(TARGET_NOTES)
            candidates.append((avg, cf, bw, g))

        elapsed = time.time() - t0
        candidates.sort()
        print(f"Done in {elapsed:.0f}s")

        current_loss = 0
        for midi, name in TARGET_NOTES:
            gen = generate_note_gpu(midi, rb, rl, rc)
            current_loss += compute_loss(targets[(midi, name)], gen)
        current_loss /= len(TARGET_NOTES)

        print(f"\n{'Rank':>4s}  {'CF':>5s}  {'BW':>5s}  {'Gain':>5s}  {'Loss':>8s}  {'vs current':>10s}")
        print("-" * 55)
        print(f" cur   2500   1200   0.30  {current_loss:8.3f}  {'baseline':>10s}")
        for i, (loss, cf, bw, g) in enumerate(candidates[:10]):
            delta = (loss - current_loss) / current_loss * 100
            print(f"  {i+1:2d}   {cf:4.0f}   {bw:4.0f}   {g:.2f}  {loss:8.3f}  {delta:+8.1f}%")

        best_loss, best_cf, best_bw, best_g = candidates[0]
        print(f"\nBest bridge: center={best_cf} Hz, BW={best_bw} Hz, gain={best_g}")
