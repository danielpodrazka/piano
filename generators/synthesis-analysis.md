# Grand Piano Synthesis vs Salamander — Comparative Analysis

Run: March 2026, 17 notes C2–C6, pyAudioAnalysis + scipy.
Reference: Salamander Grand Piano (Yamaha C5, v4/mf layer).
Synthesis: Physics-based modal synthesis with soundboard IR (piano/ 8-point reference).

---

## Summary Table

| Feature | Synth | Ref | Delta | % diff |
|---|---|---|---|---|
| Attack time | 43.5 ms | 27.1 ms | +16.5 ms | +61% |
| Decay rate | −8.09 dB/s | −8.85 dB/s | +0.77 dB/s | +9% |
| Sustain level | −19.7 dB | −22.8 dB | +3.1 dB | +14% |
| Spectral centroid | 641 Hz | 555 Hz | +86 Hz | +15% |
| Spectral spread | 495 Hz | 366 Hz | +129 Hz | +35% |
| Spectral rolloff | 951 Hz | 721 Hz | +230 Hz | +32% |
| Spectral flatness | 2.07e-6 | 2.31e-5 | −91% | −91% |
| Brightness (>2kHz) | 0.032 | 0.037 | −0.005 | −14% |
| Inharmonicity B | ~0 | ~6e-6 | — | matched* |
| HNR | 17.6 dB | 13.7 dB | +3.9 dB | +28% |
| Partial amp slope | −2.24 dB/n | −2.56 dB/n | +0.32 dB/n | +12% |

*Inharmonicity deviations in cents are nearly identical up to partial ~10; peak-finder
loses track above that. Not a real problem.

---

## Issues Ranked by Priority

### 1. Attack too slow (+61%, 43ms vs 27ms) — HIGH
Real piano: hard percussive hammer transient spike → fast onset → resonant tail.
Synthesis: smooth onset ramp, no initial transient burst.
Fix: faster attack envelope + short noise burst at onset.

### 2. Too clean — no noise floor (spectral flatness −91%, HNR +3.9 dB) — HIGH
Synth is ~10× more tonal than reference.
Real piano has:
- Hammer felt noise at onset (broadband, decays in ~30ms)
- Sympathetic string buzz during sustain
- Low-level broadband noise floor under tone

Fix: shaped noise injection — loud burst at onset (decays quickly), residual in sustain.

### 3. High notes thin above C5 — HIGH
MFCC-L2 distance: C2 = 1.5, C4 = 1.8, C5 = 2.9, A5 = 11.6, C6 = 17.8.
Dominant gap: MFCC-1 (log spectral energy), increasingly negative for synth above C5.
Real treble register gets richness from:
- Sympathetic resonance from all undamped strings (pedal effect)
- Soundboard "bloom" — plate modes sustaining longer at high freq
- More complex IR shape the 8-point interpolation can't capture

Fix: More IR anchor points in C5–C6 range, or a per-note sustain-level correction.

### 4. Sub-bass artifact in treble notes (Ds4 and up) — MEDIUM
Synth sub-bass (0–250 Hz) energy: Ds4=−70, Fs4=−72, A4=−74, C5=−75 dB
Ref sub-bass energy:            Ds4=−29, Fs4=−32, A4=−25, C5=−25 dB
~45 dB excess sub-bass in synthesis above Ds4. Likely DC or very-low-freq artifact.

Fix: High-pass filter on synthesis output above ~E4 (MIDI 64).

### 5. Upper-mid 2–5 kHz inconsistency in bass register (C2–A3) — MEDIUM
The presence band (8th–16th harmonics of bass notes) is 6–9 dB off, with
inconsistent sign note-to-note. Root cause: only 8 IR anchor points → rough
interpolation in this register.

Example:
- C2: synth −27.5 dB, ref −19.0 dB (−8.5 dB deficit)
- A2: synth −15.9 dB, ref −23.3 dB (+7.4 dB excess)

Fix: Add more IR anchor points in C2–A3 range (currently C2, A2, C3, A3).
Could add Ds2, Fs2, Ds3, Fs3 to double density.

### 6. Partial rolloff too shallow (+12%) — LOW
Synth: −2.24 dB/partial, Ref: −2.56 dB/partial.
Upper harmonics slightly too loud — contributes to elevated spectral centroid.
Fix: Increase harmonic amplitude rolloff exponent in generator.

---

## Per-Note MFCC L2 Distance

| Note | L2 | Cosine | Worst MFCC |
|---|---|---|---|
| C2 | 1.45 | 0.9985 | MFCC-2 |
| A2 | 1.68 | 0.9985 | MFCC-2 |
| C3 | 1.68 | 0.9992 | MFCC-1 |
| A3 | 2.39 | 0.9974 | MFCC-3 |
| C4 | 1.81 | 0.9985 | MFCC-3 |
| A4 | 2.69 | 0.9973 | MFCC-2 |
| C5 | 2.87 | 0.9990 | MFCC-1 |
| A5 | 11.57 | 0.9982 | MFCC-1 |
| C6 | 17.84 | 0.9980 | MFCC-1 |

---

## C4 Partial Structure (first 10 partials well-matched)

| n | Ideal Hz | Synth Hz | Ref Hz | S dB | R dB | S dev | R dev |
|---|---|---|---|---|---|---|---|
| 1 | 261.6 | 261.1 | 261.1 | 56.2 | 60.8 | −3.5¢ | −3.5¢ |
| 2 | 523.3 | 523.5 | 523.5 | 59.0 | 65.3 | +0.9¢ | +0.9¢ |
| 3 | 784.9 | 786.0 | 786.0 | 52.3 | 42.6 | +2.4¢ | +2.4¢ |
| 4 | 1046.5 | 1048.4 | 1048.4 | 50.8 | 44.8 | +3.1¢ | +3.1¢ |
| 5 | 1308.1 | 1312.2 | 1312.2 | 41.9 | 46.7 | +5.4¢ | +5.4¢ |
| 6 | 1569.8 | 1578.7 | 1577.3 | 46.7 | 43.0 | +9.8¢ | +8.3¢ |
| 7 | 1831.4 | 1845.1 | 1843.8 | 47.1 | 46.5 | +12.9¢ | +11.7¢ |
| 8 | 2093.0 | 2112.9 | 2111.6 | 34.6 | 35.3 | +16.4¢ | +15.3¢ |
| 9 | 2354.6 | 2383.5 | 2382.1 | 33.3 | 32.9 | +21.1¢ | +20.1¢ |
| 10 | 2616.3 | 2655.3 | 2652.6 | 38.7 | 38.4 | +25.7¢ | +23.9¢ |

Inharmonicity cents deviation almost identical — synthesis is well-calibrated here.

---

---

## Version Stats (vs Salamander reference)

Columns: attack_ms | decay_rate dB/s | centroid Hz | brightness | HNR dB | MFCC-L2 (mean)

### v1 — IR switched from Salamander to audio/piano/ (8-point), phases re-optimized
IR anchor notes: C2 A2 C3 A3 C4 A4 C5 A5. Phase loss: 7.47 → 4.62.

| Feature | v1 | Ref | Delta |
|---|---|---|---|
| Attack time | 43.5 ms | 27.1 ms | +16.5 ms (+61%) |
| Decay rate | −8.09 dB/s | −8.85 dB/s | +0.77 dB/s (+9%) |
| Sustain level | −19.7 dB | −22.8 dB | +3.1 dB (+14%) |
| Spectral centroid | 641 Hz | 555 Hz | +86 Hz (+15%) |
| Spectral spread | 495 Hz | 366 Hz | +129 Hz (+35%) |
| Spectral flatness | 2.07e-6 | 2.31e-5 | −91% |
| Brightness (>2kHz) | 0.032 | 0.037 | −14% |
| HNR | 17.6 dB | 13.7 dB | +3.9 dB (+28%) |
| Partial amp slope | −2.24 dB/n | −2.56 dB/n | +0.32 dB/n (+12%) |
| Mean MFCC L2 | 3.65 | — | — |

Per-note MFCC L2: C2=1.45, A2=1.68, C3=1.68, A3=2.39, C4=1.81, A4=2.69,
C5=2.87, A5=11.57, C6=17.84

---

## Planned Improvement Sequence

Each change: generate 52 notes → optimize phases → save as versioned directory
for A/B comparison before continuing.

- [x] v1: IR switched from Salamander to audio/piano/ (8-point), phases re-optimized
- [ ] v2: Faster attack + onset noise burst (fix #1 + #2)
- [ ] v3: Sub-bass HPF for treble notes (fix #4)
- [ ] v4: Partial rolloff steepening (fix #6)
- [ ] v5: More IR anchor points / denser bass reference (fix #5)
- [ ] v6: High-note sustain correction (fix #3)
