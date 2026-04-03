# Physics-Based Piano Synthesis

A browser-based synthesizer that generates piano and keyboard sounds from physical models and signal processing — no neural networks, no sample manipulation, just physics equations and math.

**[Open piano.html in your browser to play]**

---

## Instruments

### Grand Piano (physics-based modal synthesis)

The main instrument. Each of the 52 notes (B1–D6) is synthesized independently from first principles:

1. **Inharmonic partials** — `f_n = n·f₀·√(1 + B·n²)` using measured stiffness coefficients (Bensa et al., JASA 2003)
2. **Calibrated damping** — `α_n = b₁ + b₂·(nπ/L)²` matched to Steinway B measurements
3. **Two-stage decay** — prompt (soundboard-coupled) + aftersound (decoupled), from Weinreich (JASA 1977)
4. **Nonlinear hammer** — velocity → contact duration → spectral tilt via `F = K·x^p` (Chaigne & Askenfelt, JASA 1994)
5. **Multiple string coupling** — 1–3 strings per note with slight detuning → beating and chorus
6. **Phantom partials** — sum-frequency longitudinal modes in the bass register (Bank & Sujbert, JASA 2005)
7. **Soundboard IR convolution** — per-note transfer functions extracted from 8 Salamander reference notes via spectral division; magnitude-based IR with Wiener deconvolution, 85% wet mix
8. **8 velocity layers** — timbre changes with strike velocity (pp → fff)

Phases were optimized with gradient descent against spectral loss. See `generators/optimize_phases.py`.

**Comparison vs. Salamander (Yamaha C5, recorded):**
- Spectral centroid: 670 Hz vs 555 Hz reference (+21%)
- Brightness (>2kHz energy): 0.028 vs 0.037 reference
- Mean MFCC-L2 across C2–C5: ~2.0 (C6 diverges due to high-note thinness)

### Rhodes FM (physics-based FM synthesis)

Electric piano model using FM synthesis with parameters derived from the electromechanical tine vibration:
- Tine + tonebar as coupled oscillators (carrier + modulator)
- Per-note tuning, decay, and FM index from reference recordings
- Soundboard impulse response convolution
- 8 velocity layers

The original jRhodes3d recordings (CC BY-NC 4.0) were used as reference for tuning.

### Prism

Experimental instrument combining additive synthesis with metallic inharmonicity and custom spectral shaping. 8 velocity layers.

### DDSP Piano (comparison)

A data-driven approach using Differentiable Digital Signal Processing (DDSP): a small neural network controls additive synthesis parameters trained against reference recordings. Included for comparison — the physics model sounds more convincing at this scale.

### Reference instruments

- **Salamander (ref)** — Salamander Grand Piano V3 by Alexander Holm (Yamaha C5, CC BY 3.0). Used as the spectral reference for soundboard IR extraction.
- **Rhodes (ref)** — jRhodes3d by Jeff Learman (1977 Rhodes Mark I Stage 73, CC BY-NC 4.0). Used to tune the FM Rhodes model.

---

## Interface (piano.html)

A single-file web app — open directly in a modern browser (Chrome/Firefox/Safari). No server needed.

Features:
- **Playable keyboard** — click or use computer keyboard
- **Staff notation** — treble and bass clef display
- **Ear training modules** — intervals, chords, scales, progressions, rhythm, dictation
- **MIDI player** — load `.mid` files or use the included warmup exercises
- **MIDI input** — connect a MIDI keyboard and play live

---

## Generators

All synthesis scripts are in `generators/`. Python 3.10+, dependencies: `numpy`, `scipy`, `soundfile`, `ffmpeg` in PATH.

| Script | Purpose |
|--------|---------|
| `generate_grand_piano.py` | Generate all 52 grand piano notes (8 velocity layers) |
| `extract_soundboard_ir.py` | Extract per-note soundboard transfer functions from Salamander reference |
| `optimize_phases.py` | Gradient descent phase optimization against spectral loss |
| `optimize_grand_piano.py` | Differential Evolution optimizer for physical parameters |
| `generate_rhodes_fm.py` | Generate FM Rhodes samples (8 velocity layers) |
| `extract_rhodes_tf.py` | Extract Rhodes transfer functions from reference samples |
| `optimize_rhodes.py` | Optimize FM Rhodes parameters |
| `generate_prism.py` | Generate Prism samples |
| `ddsp_piano.py` | DDSP piano synthesis (runs inference from `ddsp_piano_model.pt`) |
| `compare_piano.py` | Spectral comparison: synth vs reference |
| `compare_rhodes.py` | Spectral comparison: Rhodes FM vs reference |
| `analyze_comparison.py` | Detailed per-note acoustic analysis |
| `deep_compare.py` | MFCC and partial structure analysis |
| `tune_warmth.py` | Warmth/brightness parameter sweep |

Pre-computed data files:
- `soundboard_tf.npz` — soundboard transfer functions (8 Salamander anchor notes)
- `rhodes_tf.npz` — Rhodes FM transfer functions
- `ddsp_piano_model.pt` — trained DDSP model weights

### Regenerating samples

```bash
cd generators

# Grand piano (outputs to ../audio/grand-piano/)
python generate_grand_piano.py

# Rhodes FM (outputs to ../audio/rhodes-fm/)
python generate_rhodes_fm.py

# Prism (outputs to ../audio/prism/)
python generate_prism.py

# DDSP piano (outputs to ../audio/ddsp-piano/)
python ddsp_piano.py
```

---

## Repository Layout

```
piano/
├── piano.html                  # Main interface (self-contained)
├── piano-module-*.js           # Ear training modules
├── audio/
│   ├── ATTRIBUTION.md          # Sample licenses and provenance
│   ├── piano/                  # Salamander reference (17 notes, used for IR)
│   ├── salamander/             # Salamander full set (8 velocity layers)
│   ├── grand-piano/            # Physics synthesis (52 notes × 8 velocity layers)
│   ├── rhodes/                 # jRhodes3d reference (12 samples)
│   ├── rhodes-fm/              # FM synthesis (52 notes × 8 velocity layers)
│   ├── prism/                  # Prism synthesis (52 notes × 8 velocity layers)
│   └── ddsp-piano/             # DDSP synthesis (52 notes)
├── midi/songs/                 # Warmup exercises (public domain)
└── generators/                 # All synthesis scripts
```

---

## Licenses

**Code** (Python generators, JavaScript, HTML): MIT — see `LICENSE`

**Audio samples:**

| Source | License | Use |
|--------|---------|-----|
| Salamander Grand Piano V3 (Alexander Holm) | CC BY 3.0 | Reference + in-repo samples |
| jRhodes3d (Jeff Learman) | CC BY-NC 4.0 | Reference samples only (no commercial distribution) |
| Generated samples (grand-piano/, rhodes-fm/, prism/, ddsp-piano/) | MIT | Freely reusable |

The CC BY-NC 4.0 license on the jRhodes3d reference samples (`audio/rhodes/`) restricts commercial use of those files specifically. All other audio in this repo is freely usable.

---

## Physical Model References

- Bensa, Bilbao, Kronland-Martinet & Smith (2003). *The simulation of piano string vibration: From physical models to finite difference schemes and digital waveguides*. JASA 114(2).
- Chaigne & Askenfelt (1994). *Numerical simulations of struck strings*. JASA 95(2).
- Weinreich (1977). *Coupled piano strings*. JASA 62(6).
- Bank & Sujbert (2005). *Generation of longitudinal vibrations in piano strings*. JASA 117(4).
- Stulov (1995). *Hysteretic model of the grand piano hammer felt*. JASA 97(4).
