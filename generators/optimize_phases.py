#!/usr/bin/env python3
"""
Optimize the 3x64 phase table for grand piano synthesis using gradient descent.

Only optimizes phases (192 values). All other params are fixed to match
generate_grand_piano.py. Uses mel-scale STFT loss for perceptually weighted
optimization.

Re-run after changing any synthesis parameters (rolloff, bridge hill, etc.).
"""

import numpy as np
import torch
import torch.nn.functional as F
import subprocess
import tempfile
import os
import time

SAMPLE_RATE = 44100
DURATION = 4.0

CALIB_MIDI = [36, 60, 96]
CALIB_B1 = [0.25, 1.1, 9.17]
CALIB_B2 = [7.5e-5, 2.7e-4, 2.1e-3]
CALIB_L = [1.92, 0.62, 0.09]
B_MIDI = [21, 33, 45, 57, 69, 84, 96]
B_VALS = [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Per-note soundboard IRs (must match generate_grand_piano.py)
# Per-note soundboard IRs — mirrors generate_grand_piano.py format detection
_TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
_SB_IRS  = {}  # midi → torch tensor IR
_tf_data = None
_TF_MODE_OPT = None
if os.path.exists(_TF_PATH):
    _tf_data = np.load(_TF_PATH)
    if 'irs' in _tf_data:
        _TF_MIDI_OPT   = _tf_data['midi_points']
        _TF_IRS_OPT    = _tf_data['irs']
        _TF_MODE_OPT   = 'irs'
    elif 'transfer_functions' in _tf_data:
        _TF_MIDI_OPT   = _tf_data['midi_points']
        _TF_MAG_OPT    = _tf_data['transfer_functions']
        _TF_IR_LEN_OPT = int(_tf_data['ir_length'])
        _TF_MODE_OPT   = 'mag'
    else:
        _tf_data = None

if _tf_data is not None:
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
        if midi not in _SB_IRS:
            if _TF_MODE_OPT == 'irs':
                idx = np.searchsorted(_TF_MIDI_OPT, midi)
                if idx == 0:
                    ir = _TF_IRS_OPT[0].copy()
                elif idx >= len(_TF_MIDI_OPT):
                    ir = _TF_IRS_OPT[-1].copy()
                else:
                    lo, hi = idx - 1, idx
                    t = (midi - _TF_MIDI_OPT[lo]) / (_TF_MIDI_OPT[hi] - _TF_MIDI_OPT[lo])
                    ir = (1.0 - t) * _TF_IRS_OPT[lo] + t * _TF_IRS_OPT[hi]
            else:
                log_tfs = np.log(_TF_MAG_OPT + 1e-10)
                interp_log = np.array([np.interp(midi, _TF_MIDI_OPT, log_tfs[:, i])
                                       for i in range(log_tfs.shape[1])])
                ir = _min_phase_ir_np(np.exp(interp_log), _TF_IR_LEN_OPT)
            _SB_IRS[midi] = torch.tensor(ir.astype(np.float32), device=DEVICE)
        return _SB_IRS[midi]
else:
    _get_sb_ir_torch = lambda midi: None


def log_interp(midi, midi_pts, vals):
    return float(np.exp(np.interp(midi, midi_pts, np.log(vals))))


def lin_interp(midi, midi_pts, vals):
    return float(np.interp(midi, midi_pts, vals))


def load_reference(path, duration=DURATION):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        'ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE),
        '-ac', '1', '-f', 'wav', tmp_path
    ], capture_output=True)
    import wave
    with wave.open(tmp_path, 'r') as wf:
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    os.unlink(tmp_path)
    n_samples = int(SAMPLE_RATE * duration)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    return torch.tensor(audio, device=DEVICE)


def create_mel_filterbank(n_fft, n_mels=80, f_min=20, f_max=8000):
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.clip(np.round(hz_points * n_fft / SAMPLE_RATE).astype(int), 0, n_freqs - 1)
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        if center > left:
            filterbank[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if right > center:
            filterbank[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    return torch.tensor(filterbank, dtype=torch.float32, device=DEVICE)


MEL_BANKS = {nfft: create_mel_filterbank(nfft) for nfft in [512, 1024, 2048, 4096]}


def mel_stft_loss(predicted, target, n_ffts=[512, 1024, 2048, 4096]):
    loss = torch.tensor(0.0, device=predicted.device)
    for n_fft in n_ffts:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=predicted.device)
        pred_stft = torch.stft(predicted, n_fft, hop, window=window, return_complex=True)
        targ_stft = torch.stft(target, n_fft, hop, window=window, return_complex=True)
        pred_mel = torch.matmul(MEL_BANKS[n_fft], pred_stft.abs() + 1e-8)
        targ_mel = torch.matmul(MEL_BANKS[n_fft], targ_stft.abs() + 1e-8)
        loss += (targ_mel - pred_mel).norm() / targ_mel.norm()
        loss += F.l1_loss(torch.log(pred_mel + 1e-8), torch.log(targ_mel + 1e-8))
    return loss / len(n_ffts)


def synthesize_note_gpu(midi, phase_table, duration=DURATION):
    freq = 440.0 * 2 ** ((midi - 69) / 12.0)
    n_samples = int(SAMPLE_RATE * duration)
    t = torch.linspace(0, duration, n_samples, device=DEVICE)
    key_pos = max(0, min(1, (midi - 21) / 87))
    velocity = 0.8

    b1 = log_interp(midi, CALIB_MIDI, CALIB_B1)
    b2 = log_interp(midi, CALIB_MIDI, CALIB_B2)
    L = log_interp(midi, CALIB_MIDI, CALIB_L)
    piL2 = (np.pi / L) ** 2
    B = log_interp(midi, B_MIDI, B_VALS)

    max_partial = 1
    while max_partial * freq * np.sqrt(1 + B * max_partial ** 2) < SAMPLE_RATE / 2 - 500:
        max_partial += 1
    max_partial = min(max_partial - 1, 64)

    # Must match generate_grand_piano.py exactly
    vel_clamp = max(velocity, 0.05)
    effective_hardness = 0.15 + 0.85 * vel_clamp ** 1.5
    T_c_base = lin_interp(midi, [36, 60, 96], [0.003, 0.0018, 0.0006])
    T_c = T_c_base * (2.5 - 1.9 * effective_hardness)
    hammer_cutoff = 2.5 / T_c
    hammer_rolloff_exp = 2.0 - 0.8 * effective_hardness
    strike_pos = lin_interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

    prompt_factor = (1.2 + 0.3 * key_pos) * (0.7 + 0.5 * effective_hardness)
    after_factor = 0.45
    A_after = (0.18 + 0.07 * key_pos) * (1.3 - 0.4 * effective_hardness)

    rolloff = 1.0 + 0.3 * key_pos + 1.2 * key_pos ** 2

    if midi < 36:
        string_detunes = [0.0]
    elif midi < 48:
        dc = 0.3 + 0.3 * key_pos
        string_detunes = [-dc, dc]
    else:
        dc = 0.15 + 0.25 * key_pos
        string_detunes = [-dc, 0.0, dc]

    n_strings = len(string_detunes)
    low_modes = [(90, 30, 0.15), (170, 35, 0.12), (260, 45, 0.10)]
    signal = torch.zeros(n_samples, device=DEVICE)

    for s_idx, d_cents in enumerate(string_detunes):
        detune_ratio = 2 ** (d_cents / 1200)
        string_amp = 1.0 / n_strings

        for n in range(1, max_partial + 1):
            partial_freq = n * freq * detune_ratio * np.sqrt(1 + B * n ** 2)
            if partial_freq >= SAMPLE_RATE / 2:
                break

            amp = 1.0 / (n ** rolloff)
            amp *= 1.0 / (1.0 + (partial_freq / hammer_cutoff) ** hammer_rolloff_exp)
            fTc = partial_freq * T_c
            denom = 1.0 - 4.0 * fTc * fTc
            if abs(denom) < 1e-6:
                cosine_mod = 1.0
            else:
                cosine_mod = min(abs(np.cos(np.pi * fTc) / denom), 1.0)
            amp *= 0.7 + 0.3 * cosine_mod
            strike_factor = abs(np.sin(np.pi * n * strike_pos))
            amp *= max(strike_factor, 0.03)

            sb_response = 1.0
            for cf, bw, gain in low_modes:
                sb_response += gain * np.exp(-0.5 * ((partial_freq - cf) / bw) ** 2)
            sb_response += 0.40 * np.exp(-0.5 * ((partial_freq - 1800) / 800) ** 2)
            amp *= sb_response * string_amp

            K_n = (n ** 2) * piL2
            air_frac = 0.2
            alpha_n = (b1 * (1.0 - air_frac)
                       + b1 * air_frac * np.sqrt(freq / max(partial_freq, 20.0))
                       + b2 * K_n)
            sb_coupling = np.sqrt(sb_response)
            prompt_rate = alpha_n * prompt_factor * sb_coupling
            after_rate = alpha_n * after_factor
            A_after_n = A_after / sb_coupling
            A_prompt_n = 1.0 - A_after_n
            env = A_prompt_n * torch.exp(-t * prompt_rate) + A_after_n * torch.exp(-t * after_rate)

            phase = phase_table[s_idx % 3, (n - 1) % 64]
            signal = signal + amp * env * torch.sin(2 * np.pi * partial_freq * t + phase)

    attack_peak = lin_interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
    attack_env = torch.where(t < attack_peak,
                             0.5 - 0.5 * torch.cos(np.pi * t / attack_peak),
                             torch.ones_like(t))
    signal = signal * attack_env

    # Per-note soundboard IR convolution (matching generate_grand_piano.py)
    sb_ir = _get_sb_ir_torch(midi)
    if sb_ir is not None:
        ir = sb_ir.unsqueeze(0).unsqueeze(0)
        sig = signal.unsqueeze(0).unsqueeze(0)
        pad_len = len(sb_ir) - 1
        wet = torch.nn.functional.conv1d(
            torch.nn.functional.pad(sig, (pad_len, 0)),
            ir
        ).squeeze()[:n_samples]
        dry_rms = (signal ** 2).mean().sqrt() + 1e-10
        wet_rms = (wet ** 2).mean().sqrt() + 1e-10
        wet = wet * dry_rms / wet_rms
        signal = 0.15 * signal + 0.85 * wet

    fade_samples = int(0.1 * SAMPLE_RATE)
    fade = torch.linspace(1, 0, fade_samples, device=DEVICE)
    signal = signal.clone()
    signal[-fade_samples:] = signal[-fade_samples:] * fade

    peak = signal.abs().max()
    if peak > 0:
        signal = signal / peak * 0.85
    return signal


def main():
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'audio', 'piano')
    target_notes = {
        'C2': 36, 'A2': 45, 'C3': 48, 'C4': 60, 'A4': 69, 'C5': 72, 'A5': 81
    }

    print(f"Device: {DEVICE}")
    print(f"Loading {len(target_notes)} reference samples...")
    refs = {}
    for name, midi in target_notes.items():
        path = os.path.join(ref_dir, f'{name}.mp3')
        if os.path.exists(path):
            refs[midi] = load_reference(path)
            print(f"  {name}: loaded")

    current_phases = np.array(np.random.RandomState(6454).uniform(0, 2 * np.pi, (3, 64)),
                              dtype=np.float32)
    phase_table = torch.tensor(current_phases, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([phase_table], lr=0.05)
    n_iters = 400
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    print(f"\nOptimizing 192 phase values (mel-scale STFT loss)...")
    best_loss = float('inf')
    best_phases = current_phases.copy()
    t_start = time.time()

    for iteration in range(n_iters):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=DEVICE)
        for midi, ref in refs.items():
            synth = synthesize_note_gpu(midi, phase_table)
            min_len = min(len(synth), len(ref))
            total_loss = total_loss + mel_stft_loss(synth[:min_len], ref[:min_len])
        total_loss = total_loss / len(refs)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        pct = (iteration + 1) / n_iters
        elapsed = time.time() - t_start
        eta = elapsed / max(pct, 0.001) * (1 - pct)
        bar_len = 30
        filled = int(bar_len * pct)
        bar = '=' * filled + '>' * (1 if filled < bar_len else 0) + '.' * (bar_len - filled - 1)
        print(f"\r  [{bar}] {pct*100:5.1f}%  loss={loss_val:.3f}  best={best_loss:.3f}  ETA={eta:.0f}s", end='', flush=True)

        if loss_val < best_loss:
            best_loss = loss_val
            best_phases = phase_table.detach().cpu().numpy().copy()

    best_phases = best_phases % (2 * np.pi)
    print(f"\n\nBest loss: {best_loss:.4f} (baseline: ~6.95)")
    print(f"Time: {time.time() - t_start:.0f}s")

    print("\n_PHASE_TABLE = np.array([")
    for s in range(3):
        vals = ', '.join(f'{v:.4f}' for v in best_phases[s])
        comma = ',' if s < 2 else ''
        print(f'    [{vals}]{comma}')
    print("])")

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimized_phases.npy'),
            best_phases)


if __name__ == '__main__':
    main()
