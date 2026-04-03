#!/usr/bin/env python3
"""
Generate Grand Piano samples using physics-based modal synthesis.

Calibrated from measured parameters in the literature:
  - Bensa, Bilbao, Kronland-Martinet & Smith (JASA, 2003):
    String stiffness (epsilon), damping coefficients (b1, b2), string lengths
  - Chaigne & Askenfelt (JASA, 1994):
    Hammer force law F=K·x^p, hammer masses, strike positions
  - Weinreich, "Coupled Piano Strings" (JASA, 1977):
    Two-stage decay from coupled string modes (prompt/aftersound)
  - Bank & Sujbert (JASA, 2005):
    Phantom partials from longitudinal string vibrations
  - Steinway B inharmonicity measurements (U. Alabama Huntsville):
    A0: B≈0.00031, A3: B≈0.00021, A4: B≈0.00075

Physical model per note:
  1. INHARMONIC PARTIALS: f_n = n·f₀·√(1 + B·n²)
  2. CALIBRATED DAMPING: α_n = b₁ + b₂·(nπ/L)²  (Bensa et al.)
  3. TWO-STAGE DECAY: prompt (soundboard-coupled) + aftersound (decoupled)
  4. NONLINEAR HAMMER: velocity→contact duration→spectral tilt (F=Kx^p)
  5. MULTIPLE STRINGS: 1-3 per note, slight detuning → beating + chorus
  6. HAMMER ATTACK TRANSIENT: soundboard impulse approximation
  7. PHANTOM PARTIALS: sum-frequency longitudinal modes in bass register

Output: MP3 files ready for the learn-piano.html sampler.
"""

import numpy as np
import os
import subprocess

SAMPLE_RATE = 44100
DURATION = 6.0  # enough for full two-stage decay

# Per-note soundboard IRs. Two formats supported:
#   'irs'                — time-domain IRs (synthesize_soundboard_ir.py)
#   'transfer_functions' — magnitude responses (extract_soundboard_ir.py, legacy)
_TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
_SOUNDBOARD_IR = None  # kept for compatibility check in optimizer
_TF_MODE = None
if os.path.exists(_TF_PATH):
    _tf_data = np.load(_TF_PATH)
    if 'irs' in _tf_data:
        _TF_MIDI   = _tf_data['midi_points']
        _TF_IRS    = _tf_data['irs']           # [n_notes, ir_length]
        _TF_IR_LEN = int(_tf_data['ir_length'])
        _IR_CACHE  = {}
        _TF_MODE   = 'irs'
    elif 'transfer_functions' in _tf_data:
        _TF_MIDI   = _tf_data['midi_points']
        _TF_FREQS  = _tf_data['freqs']
        _TF_MAG    = _tf_data['transfer_functions']  # [n_notes, n_bins]
        _TF_IR_LEN = int(_tf_data['ir_length'])
        _IR_CACHE  = {}
        _TF_MODE   = 'mag'
    else:
        _tf_data = None
else:
    _tf_data = None

NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']


def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


# Every chromatic note from B1 (35) to D6 (86)
NOTES = [(m, midi_to_name(m)) for m in range(35, 87)]

# ═══ MEASURED PARAMETERS FROM BENSA ET AL. (2003) ═══
# Reference notes with calibrated values
# Format: midi → (b1, b2, string_length_m)
#   b1: frequency-independent damping (s⁻¹)
#   b2: frequency-dependent damping (s)
#   L:  vibrating string length (m)
CALIB_NOTES = {
    36: {'b1': 0.25,  'b2': 7.5e-5,  'L': 1.92},  # C2
    60: {'b1': 1.1,   'b2': 2.7e-4,  'L': 0.62},  # C4
    96: {'b1': 9.17,  'b2': 2.1e-3,  'L': 0.09},  # C7
}
CALIB_MIDI = sorted(CALIB_NOTES.keys())

# Per-partial per-string phase table, optimized via gradient descent
# (optimize_phases.py) against recorded reference samples using mel-scale
# STFT loss. Re-run optimize_phases.py after changing any synthesis parameters
# (rolloff, bridge hill, decay, etc.) since phases are coupled to them.
_PHASE_TABLE = np.array([
    [4.0132, 2.4928, 1.6716, 0.8096, 0.9535, 5.3968, 3.8574, 2.4742, 4.9981, 1.8611, 3.7908, 5.9255, 0.6455, 0.3276, 5.8913, 5.4893, 4.6771, 1.2357, 3.2581, 2.7561, 0.8428, 4.0558, 5.2799, 0.9260, 0.6861, 5.1939, 1.2753, 1.4505, 0.1271, 3.7157, 0.7629, 1.4660, 0.2760, 2.8631, 4.7170, 4.0562, 1.4049, 2.9267, 2.4257, 0.3066, 1.6580, 1.7747, 5.1121, 0.7141, 0.1317, 3.3997, 4.9164, 4.2310, 0.9365, 1.5663, 6.2455, 1.8572, 2.0187, 5.6055, 1.2101, 0.2159, 2.9470, 3.6037, 1.2411, 3.0013, 2.1811, 4.8246, 5.0224, 3.2362],
    [1.3617, 4.1407, 3.1679, 2.0926, 2.8498, 5.8210, 4.4870, 2.3379, 5.2825, 2.0538, 4.1686, 6.1295, 0.6556, 0.3439, 5.9858, 5.2892, 4.1228, 0.8159, 2.6249, 2.3138, 0.0429, 3.2546, 4.8054, 1.0011, 5.9662, 4.5308, 0.1382, 0.4243, 5.1007, 2.6385, 5.3386, 6.0777, 4.8098, 2.4797, 3.5437, 2.2902, 5.5527, 0.8090, 0.7646, 4.7900, 5.6842, 5.9511, 2.8493, 4.6506, 4.0068, 1.0178, 2.3836, 1.5110, 4.2781, 5.6876, 3.9259, 5.7782, 5.7022, 2.6450, 4.3279, 3.0211, 5.3764, 6.0428, 3.6381, 5.4519, 4.6440, 0.9308, 1.0450, 5.1668],
    [5.3995, 5.6364, 4.5929, 3.5345, 4.3810, 0.1389, 5.3429, 2.5298, 5.5007, 2.4105, 4.6763, 0.2695, 0.9782, 0.8100, 0.1505, 5.2394, 3.1700, 0.0055, 1.8871, 2.4310, 2.5955, 1.1359, 4.7438, 1.2807, 0.7266, 3.9978, 5.2003, 5.8520, 3.4716, 1.8749, 3.4160, 4.3617, 2.2013, 2.4004, 2.6537, 0.4599, 3.2560, 4.7558, 5.4229, 3.1209, 2.9916, 3.3205, 0.2833, 2.1776, 1.5710, 4.9177, 6.1946, 5.1710, 0.3391, 0.8279, 0.8513, 4.7880, 0.2794, 6.2582, 3.6578, 5.5488, 4.6602, 1.7406, 2.8100, 1.5382, 3.6269, 5.0149, 3.2789, 5.5434],
])


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


def _get_soundboard_ir(midi):
    """Get per-note soundboard IR, interpolated from reference notes."""
    if _tf_data is None:
        return None
    if midi in _IR_CACHE:
        return _IR_CACHE[midi]

    if _TF_MODE == 'irs':
        # Time-domain IRs (from synthesize_soundboard_ir.py): linear interp
        idx = np.searchsorted(_TF_MIDI, midi)
        if idx == 0:
            ir = _TF_IRS[0].copy()
        elif idx >= len(_TF_MIDI):
            ir = _TF_IRS[-1].copy()
        else:
            lo, hi = idx - 1, idx
            t = (midi - _TF_MIDI[lo]) / (_TF_MIDI[hi] - _TF_MIDI[lo])
            ir = (1.0 - t) * _TF_IRS[lo] + t * _TF_IRS[hi]
    else:
        # Magnitude responses (legacy extract_soundboard_ir.py): log interp → min-phase FIR
        log_tfs = np.log(_TF_MAG + 1e-10)
        interp_log = np.zeros(log_tfs.shape[1])
        for i_bin in range(log_tfs.shape[1]):
            interp_log[i_bin] = np.interp(midi, _TF_MIDI, log_tfs[:, i_bin])
        mag_response = np.exp(interp_log)
        ir = _minimum_phase_ir(mag_response, _TF_IR_LEN)
        ir_energy = np.sqrt(np.sum(ir ** 2))
        if ir_energy > 0:
            n_fft = (len(_TF_FREQS) - 1) * 2
            ir = ir / ir_energy * np.sqrt(_TF_IR_LEN) / np.sqrt(n_fft)

    _IR_CACHE[midi] = ir
    return ir


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def interp_param(midi, param):
    """Log-linear interpolation of a calibrated parameter across the keyboard."""
    vals = [CALIB_NOTES[m][param] for m in CALIB_MIDI]
    return np.exp(np.interp(midi, CALIB_MIDI, np.log(vals)))


def generate_grand_piano_note(midi, duration=DURATION, velocity=0.8):
    """Generate a grand piano note using physics-based modal synthesis."""
    freq = midi_to_freq(midi)
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Reproducible randomness per note
    rng = np.random.RandomState(midi * 1000 + 42)

    # Key position 0-1 across keyboard (A0=21 to C8=108)
    key_pos = np.clip((midi - 21) / 87, 0, 1)

    # ═══ CALIBRATED STRING PARAMETERS ═══
    b1 = interp_param(midi, 'b1')     # frequency-independent damping
    b2 = interp_param(midi, 'b2')     # frequency-dependent damping
    L = interp_param(midi, 'L')       # string length
    piL2 = (np.pi / L) ** 2           # spatial frequency factor

    # ═══ INHARMONICITY COEFFICIENT B ═══
    # Calibrated from Steinway B grand measurements (U. Alabama Huntsville):
    #   A0: 0.00031, A3: 0.00021, A4: 0.00075
    # Wound strings (A0-A3): B dips in mid-bass (optimal string design)
    # Plain strings (A4+): B rises steeply as strings get shorter/stiffer
    # Log-interpolation between measured/estimated calibration points.
    _B_midi = [21,   33,    45,    57,    69,    84,    96]
    _B_vals = [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2]
    B = np.exp(np.interp(midi, _B_midi, np.log(_B_vals)))

    # ═══ MAX PARTIALS (up to Nyquist) ═══
    max_partial = 1
    while max_partial * freq * np.sqrt(1 + B * max_partial**2) < SAMPLE_RATE / 2 - 500:
        max_partial += 1
    max_partial = min(max_partial - 1, 64)

    # ═══ HAMMER MODEL (Chaigne & Askenfelt, Russell & Rossing 1998) ═══
    # Nonlinear felt: F = K·x^p with three-layer hardness gradient.
    # Real hammer felt has soft outer surface → medium → hard inner core.
    # pp only compresses soft surface (dark tone); ff reaches hard core (bright).
    # p exponent: bass ~2.0, treble ~3.5 (Russell & Rossing measurements).
    vel_clamp = max(velocity, 0.05)

    # [1] VELOCITY-DEPENDENT HARDNESS (nonlinear felt F=Kx^p)
    # Effective felt stiffness increases nonlinearly with compression depth.
    # Concave curve models the three-layer gradient: slow change pp→mp,
    # accelerating change mf→ff as harder inner layers engage.
    # Floor of 0.15 ensures pp retains some harmonic character (not pure sine).
    # pp(0.15)→0.20, mp(0.6)→0.54, f(0.8)→0.76, ff(0.9)→0.88, fff(1.0)→1.0
    effective_hardness = 0.15 + 0.85 * vel_clamp ** 1.5

    # [2] THREE-LAYER CONTACT TIME (Chaigne & Askenfelt 1994, Fig. 5)
    # T_c varies ~3x from pp to fff. Soft surface gives longer contact (darker),
    # hard core gives shorter contact (brighter).
    # Shortened vs previous (was 4ms/2.5ms/0.8ms) — analysis showed partials
    # h5+ were 10-30 dB too weak at A4, meaning the hammer filter cutoff was
    # too low. Shorter contact = higher cutoff = more surviving upper partials.
    T_c_base = np.interp(midi, [36, 60, 96], [0.003, 0.0018, 0.0006])
    T_c = T_c_base * (2.5 - 1.9 * effective_hardness)
    hammer_cutoff = 2.5 / T_c

    # [3] VELOCITY-DEPENDENT ROLLOFF STEEPNESS
    # Soft felt (pp) acts as a low-pass with steeper rolloff — fewer harmonics.
    # Hard felt (ff) has gentle rolloff — rich harmonic content.
    # This is the primary timbral mechanism: pp sounds "round", ff sounds "brilliant".
    # Reduced steepness (was 2.4-1.2) so upper partials survive better.
    hammer_rolloff_exp = 2.0 - 0.8 * effective_hardness
    # pp(0.15): 1.84 (steep), mp(0.6): 1.57, f(0.8): 1.39, fff(1.0): 1.20 (gentle)

    # ═══ STRIKE POSITION (Chaigne & Askenfelt) ═══
    # Bass: 0.12 of string length, treble: 0.0625
    strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

    # ═══ TWO-STAGE DECAY (Weinreich) ═══
    # [4] VELOCITY-DEPENDENT PROMPT/AFTERSOUND RATIO
    # Harder strikes couple more energy into the soundboard (stronger prompt),
    # but the prompt also decays faster because energy radiates away quickly.
    # Soft strikes: less initial energy transfer, more stays as aftersound.
    # Reduced prompt_factor vs previous: analysis showed prompt/aftersound ratio
    # was 3-8x too extreme in treble (Ds4: 36x vs Salamander's 6x).
    prompt_factor = (1.2 + 0.3 * key_pos) * (0.7 + 0.5 * effective_hardness)
    # Increased after_factor: aftersound was decaying too slowly (t_40dB 1.65x
    # longer than Salamander). Higher rate = faster tail decay.
    after_factor = 0.45
    # Aftersound fraction: slightly higher at pp (more energy stays in strings)
    A_after = (0.18 + 0.07 * key_pos) * (1.3 - 0.4 * effective_hardness)
    # pp: A_after ~0.24 (more sustain), fff: A_after ~0.17 (more prompt)

    # ═══ NUMBER OF STRINGS & DETUNING ═══
    if midi < 36:      # monochord (1 string)
        string_detunes = [0.0]
    elif midi < 48:    # bichord (2 strings)
        dc = 0.3 + 0.3 * key_pos
        string_detunes = [-dc, dc]
    else:              # trichord (3 strings)
        dc = 0.15 + 0.25 * key_pos
        string_detunes = [-dc, 0.0, dc]

    n_strings = len(string_detunes)

    # ═══ MODAL SYNTHESIS: partials per string ═══
    signal = np.zeros(n_samples)

    for s_idx, d_cents in enumerate(string_detunes):
        detune_ratio = 2 ** (d_cents / 1200)
        string_amp = 1.0 / n_strings

        for n in range(1, max_partial + 1):
            # Inharmonic partial frequency
            partial_freq = n * freq * detune_ratio * np.sqrt(1 + B * n**2)
            if partial_freq >= SAMPLE_RATE / 2:
                break

            # --- Initial amplitude ---
            # Bass: moderate rolloff preserves harmonics without overwhelming fundamental.
            # Mid/treble: gentle rolloff — upper partials need to survive the hammer
            # filter and still be audible. Salamander A4 h8 is -22dB (not -54dB).
            # The hammer filter (below) provides most of the HF shaping, so
            # the base rolloff should be gentle, letting the hammer physics
            # determine brightness rather than an aggressive power law.
            rolloff = 1.2 + 0.3 * key_pos + 1.2 * key_pos ** 2
            # B1(0.16): 1.28, C4(0.45): 1.58, A4(0.55): 1.73, A5(0.69): 1.98, D6(0.75): 2.10
            # +0.2 offset vs v2: leaves fundamental unchanged, -1 to -5 dB on upper partials
            amp = 1.0 / (n ** rolloff)

            # Hammer spectral shaping (Chaigne & Askenfelt)
            # Velocity-dependent rolloff: steep at pp (soft felt), gentle at ff (hard felt).
            # Modulated by shallow half-cosine dips at 1.5/Tc, 2.5/Tc.
            # Real felt nonlinearity (F=Kx^p) fills theoretical nulls — dips ~3 dB.
            amp *= 1.0 / (1.0 + (partial_freq / hammer_cutoff) ** hammer_rolloff_exp)
            fTc = partial_freq * T_c
            denom = 1.0 - 4.0 * fTc * fTc
            if abs(denom) < 1e-6:
                cosine_mod = 1.0
            else:
                cosine_mod = min(abs(np.cos(np.pi * fTc) / denom), 1.0)
            amp *= 0.7 + 0.3 * cosine_mod

            # Strike position node suppression
            strike_factor = abs(np.sin(np.pi * n * strike_pos))
            amp *= max(strike_factor, 0.03)

            # Soundboard spectral envelope (Smith & Van Duyne 1995,
            # Boutillon & Ege 2013). The soundboard's resonance structure
            # colors each partial — the piano's "voice."
            # Below ~300 Hz: distinct global modes add warmth.
            # 1-4 kHz: "bridge hill" — broad resonance that gives brightness
            # (Giordano 1998, Conklin 1996).
            f_hz = partial_freq
            sb_response = 1.0
            for cf, bw, gain in [(90, 30, 0.15), (170, 35, 0.12),
                                  (260, 45, 0.10)]:
                sb_response += gain * np.exp(-0.5 * ((f_hz - cf) / bw) ** 2)
            sb_response += 0.40 * np.exp(-0.5 * ((f_hz - 1800) / 800) ** 2)
            amp *= sb_response

            amp *= string_amp

            # --- Per-partial decay rate ---
            # Three-term model (Desvages & Bilbao 2016, Issanchou et al. 2017)
            # Splits Bensa's b₁ into support coupling + air viscosity.
            # Air drag ∝ 1/√f creates a mid-frequency dip where harmonics
            # h2-h5 sustain longer than the fundamental — key piano trait.
            # At the fundamental, total = b₁ (unchanged from Bensa).
            K_n = (n ** 2) * piL2
            air_frac = 0.2
            alpha_n = (b1 * (1.0 - air_frac)
                       + b1 * air_frac * np.sqrt(freq / max(partial_freq, 20.0))
                       + b2 * K_n)

            # Two-stage envelope: prompt + aftersound
            # Partial-dependent coupling (Bank et al. 2010, Miranda Valiente 2024):
            # partials near soundboard resonances couple more strongly →
            # faster prompt decay, less energy remains as aftersound.
            # Creates natural timbre evolution: brightness fades before fundamental.
            # Use sqrt(sb_response) to moderate coupling — full sb_response made
            # treble prompt/aftersound ratio 3-8x too extreme vs Salamander.
            sb_coupling = np.sqrt(sb_response)
            prompt_rate = alpha_n * prompt_factor * sb_coupling
            after_rate = alpha_n * after_factor  # decoupled mode — unaffected
            A_after_n = A_after / sb_coupling
            A_prompt_n = 1.0 - A_after_n
            env = A_prompt_n * np.exp(-t * prompt_rate) + A_after_n * np.exp(-t * after_rate)

            # Fixed phase from lookup table — consistent timbre across keyboard
            phase = _PHASE_TABLE[s_idx][n - 1]

            signal += amp * env * np.sin(2 * np.pi * partial_freq * t + phase)

    # ═══ PHANTOM PARTIALS (Bank & Sujbert 2005, Conklin 1999) ═══
    # Longitudinal string vibrations from geometric nonlinearity produce
    # "phantom partials" at sum frequencies of transverse partial pairs,
    # adding metallic shimmer to the attack. Extended to C5 (midi 72)
    # since they're audible higher than commonly assumed (Bank 2010).
    if midi < 72 and max_partial >= 4:
        # Precompute parent partial frequencies and decay rates
        parent_freqs = {}
        parent_alphas = {}
        for n in range(1, min(12, max_partial + 1)):
            f_n = n * freq * np.sqrt(1 + B * n**2)
            K_n = (n ** 2) * piL2
            air_frac = 0.2
            a_n = (b1 * (1.0 - air_frac)
                   + b1 * air_frac * np.sqrt(freq / max(f_n, 20.0))
                   + b2 * K_n)
            parent_freqs[n] = f_n
            parent_alphas[n] = a_n

        for n in range(2, min(8, max_partial)):
            for m in range(1, n):
                # Correct sum frequency from actual inharmonic partials
                phantom_freq = parent_freqs[n] + parent_freqs[m]
                if phantom_freq >= SAMPLE_RATE / 2:
                    continue
                # Real phantoms are 30-40 dB below fundamental (Bank 2005)
                # Fade out above C3 — analysis showed 5-10dB too much inter-harmonic
                # energy in bass vs Salamander. Tighter fade and lower amplitude.
                phantom_scale = np.clip((52 - midi) / 16.0, 0.0, 1.0)  # full at B1, zero at E3+
                phantom_amp = 0.002 * phantom_scale / (n * m) ** 0.5
                # Decay = sum of parent rates (Bank 2010)
                phantom_decay = parent_alphas[n] + parent_alphas[m]
                # Concentrated in first ~15ms of attack (was 30ms, too persistent)
                phantom_env = np.exp(-t * phantom_decay) * np.exp(-t * 50.0)
                ph = rng.uniform(0, 2 * np.pi)
                signal += phantom_amp * phantom_env * np.sin(2 * np.pi * phantom_freq * t + ph)

        # Free longitudinal modes: steel string v_L ≈ 5100 m/s
        # These ring at multiples of v_L/(2L), independent of transverse modes.
        # Produce a brief metallic "ping" at note onset (sound precursor).
        v_L = 5100.0  # m/s, longitudinal wave speed in steel
        f_long_1 = v_L / (2.0 * L)  # fundamental longitudinal frequency
        for k in range(1, 4):  # first 3 longitudinal modes
            f_long = k * f_long_1
            if f_long >= SAMPLE_RATE / 2 or f_long < 20:
                continue
            # Very weak, very fast decay — barely audible ping in first ~20ms
            long_amp = 0.002 / k
            long_decay = b1 * 3 + k * 2.0 + 30.0  # ~20ms time constant
            long_env = np.exp(-t * long_decay)
            ph = rng.uniform(0, 2 * np.pi)
            signal += long_amp * long_env * np.sin(2 * np.pi * f_long * t + ph)

    # ═══ HAMMER ATTACK TRANSIENT ═══
    # [5] VELOCITY-DEPENDENT HAMMER NOISE
    # The soundboard is briefly excited by the hammer impact,
    # producing a short broadband "thump" that gives piano its
    # percussive attack character. Without this, piano → organ.
    # Harder strikes: louder thump, broader bandwidth, shorter duration.
    noise = rng.randn(n_samples)
    hammer_env = np.exp(-t / max(T_c * 1.5, 0.001))
    hammer_env *= (t < 0.03).astype(float)

    # Band-limit: harder strikes allow more HF through (smaller kernel)
    # Analysis showed Salamander has 5-80x more energy above 4kHz in attack,
    # especially C5+. Wider bandwidth and added HF emphasis to match.
    # pp: freq*6 bandwidth, fff: freq*16 bandwidth (was 4/10)
    noise_bw = freq * (6.0 + 10.0 * effective_hardness)
    # Ensure minimum 8kHz bandwidth for treble notes
    noise_bw = max(noise_bw, 8000.0)
    lp_size = max(int(SAMPLE_RATE / max(noise_bw, 500)), 3)
    if lp_size % 2 == 0:
        lp_size += 1
    kernel = np.ones(lp_size) / lp_size
    hammer_noise = np.convolve(noise * hammer_env, kernel, mode='same')

    # Add HF emphasis: real hammer impacts have a sharp crack component
    # that the low-pass alone can't produce. Mix in unfiltered noise
    # for the first 3ms, scaled by key position (more prominent in treble).
    # Salamander shows 5-80x more HF energy in attack, especially C5+.
    hf_dur = int(0.003 * SAMPLE_RATE)
    hf_env = np.zeros(n_samples)
    hf_env[:hf_dur] = np.exp(-np.linspace(0, 6, hf_dur))
    hf_crack = noise * hf_env * 0.6 * (0.2 + 0.8 * key_pos)
    hammer_noise += hf_crack

    # Quadratic velocity scaling: ff hammer thump is much more prominent than pp.
    # vel^2: pp(0.15)→0.023, mp(0.6)→0.36, ff(0.9)→0.81, fff(1.0)→1.0
    # Increased base level from 0.06 to 0.10 to better match Salamander's
    # attack centroid (was 176 Hz too low on average).
    hammer_level = 0.10 * vel_clamp ** 2 * (0.5 + 0.5 * key_pos) * (1.0 - 0.3 * key_pos)
    # Bass hammer thump leads the tone too much with the shorter attack ramp,
    # standing out as a click. Taper level down below G4 (MIDI 67).
    hammer_level *= np.interp(midi, [36, 67], [0.45, 1.0])

    signal += hammer_noise * hammer_level

    # ═══ ATTACK SHAPE ═══
    # Tightened targets vs v1 (was [0.050, 0.030, 0.012]):
    # v1 avg attack: 43ms; ref avg: 27ms. New values → 30ms avg (+3ms off ref).
    attack_peak = np.interp(midi, [36, 60, 96], [0.030, 0.018, 0.008])
    attack_env = np.where(t < attack_peak,
                          0.5 - 0.5 * np.cos(np.pi * t / attack_peak),
                          1.0)
    signal *= attack_env

    # ═══ VELOCITY AMPLITUDE ═══
    # Real piano: sound intensity roughly proportional to velocity^2 (kinetic energy).
    # Use a compressed power law so pp is audible but fff is significantly louder.
    vel_amp = velocity ** 1.5  # pp(0.15)→0.058, mp(0.60)→0.465, ff(0.90)→0.854, fff(1.0)→1.0
    signal *= vel_amp

    # ═══ SOUNDBOARD IR CONVOLUTION ═══
    # Per-note filter interpolated from reference recordings' transfer functions.
    # Models how soundboard resonance varies across the bridge — each note gets
    # its own spectral coloring (detrended: no room/mic tilt, just resonance detail).
    sb_ir = _get_soundboard_ir(midi)
    if sb_ir is not None:
        wet = np.convolve(signal, sb_ir, mode='full')[:n_samples]
        dry_rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
        wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-10
        wet *= dry_rms / wet_rms
        sb_mix = 0.85
        signal = (1.0 - sb_mix) * signal + sb_mix * wet

    # ═══ FADE OUT ═══
    fade_samples = int(0.1 * SAMPLE_RATE)
    signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # ═══ NORMALIZE ═══
    # When generating velocity layers, normalization is done externally
    # (relative to loudest layer) to preserve dynamics. For single-velocity
    # generation, normalize to 0.85 peak.
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


VELOCITY_LAYERS = [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 1.00]
# Map MIDI velocity (1-127) to layer index:
# Layer boundaries at midpoints: 0-24, 25-40, 41-56, 57-72, 73-88, 89-104, 105-116, 117-127


def main():
    global _tf_data
    import sys
    velocity_layers = '--velocity-layers' in sys.argv
    no_ir = '--no-ir' in sys.argv

    if no_ir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audio', 'grand-piano-dry')
        _tf_data = None
        _IR_CACHE.clear()
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audio', 'grand-piano')
    os.makedirs(out_dir, exist_ok=True)

    if velocity_layers:
        print(f"Generating {len(NOTES)} × {len(VELOCITY_LAYERS)} velocity layers = {len(NOTES) * len(VELOCITY_LAYERS)} Grand Piano samples...")
        print(f"  Velocities: {VELOCITY_LAYERS}")
    else:
        print(f"Generating {len(NOTES)} Grand Piano samples...")
    print(f"  Model: Physics-based modal synthesis (Bensa/Chaigne/Weinreich)")
    print()

    for midi, name in NOTES:
        freq = midi_to_freq(midi)
        _B_midi = [21, 33, 45, 57, 69, 84, 96]
        _B_vals = [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2]
        B = np.exp(np.interp(midi, _B_midi, np.log(_B_vals)))
        b1 = interp_param(midi, 'b1')
        L = interp_param(midi, 'L')

        if velocity_layers:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz", end='', flush=True)
            # Generate all layers, then normalize relative to loudest (v8)
            layer_signals = []
            layer_peaks = []
            for v_idx, vel in enumerate(VELOCITY_LAYERS):
                signal, raw_peak = generate_grand_piano_note(midi, velocity=vel)
                # Undo the internal normalization to get raw amplitude
                signal = signal / 0.85 * raw_peak
                layer_signals.append(signal)
                layer_peaks.append(np.max(np.abs(signal)))

            # Normalize all layers relative to the loudest (preserve dynamics)
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
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz, B={B:.6f}, b1={b1:.3f}, L={L:.3f}m")
            signal, _ = generate_grand_piano_note(midi)

            wav_path = os.path.join(out_dir, f'{name}.wav')
            mp3_path = os.path.join(out_dir, f'{name}.mp3')
            write_wav(wav_path, signal)
            wav_to_mp3(wav_path, mp3_path)
            if os.path.exists(mp3_path):
                os.remove(wav_path)
            else:
                print(f"    Warning: ffmpeg conversion failed, keeping WAV")

    print(f"\nDone! Samples written to {out_dir}/")


if __name__ == '__main__':
    main()
