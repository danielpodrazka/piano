# Audio Sample Attribution

## Salamander Grand Piano (`audio/piano/` and `audio/salamander/`)

**Source:** Salamander Grand Piano V3 by Alexander Holm
- Instrument: Yamaha C5 Grand Piano, recorded with two AKG C414 microphones in AB position
- Original project: https://sfzinstruments.github.io/pianos/salamander/
- Archive: https://archive.org/details/SalamanderGrandPianoV3
- GitHub: https://github.com/sfzinstruments/SalamanderGrandPiano

**License:** CC BY 3.0 (Creative Commons Attribution 3.0)
- You may share and adapt the samples for any purpose, including commercially
- You must give appropriate credit to Alexander Holm
- https://creativecommons.org/licenses/by/3.0/

**`audio/piano/`** — 17 samples (A1, C2–C6 at roughly every 3 semitones), MP3, sourced from the
Tone.js CDN. These are used exclusively as the **soundboard impulse response reference** in
`generators/extract_soundboard_ir.py`: per-note transfer functions are extracted from these
recordings via spectral division, then convolved into the physics synthesis to give the grand
piano its body resonance.

**`audio/salamander/`** — Full set of 19 samples with 8 velocity layers (pp–fff), used as the
in-browser **reference instrument** for A/B comparison against the physics-based synthesis.

---

## Rhodes Samples (`audio/rhodes/`)

**Source:** jRhodes3d by Jeff Learman (jlearman)
- Instrument: 1977 Rhodes Mark I Stage 73 Electric Piano
- Original project: https://github.com/sfzinstruments/jlearman.jRhodes3d
- SFZ Instruments page: https://sfzinstruments.github.io/pianos/jrhodes3d/

**License:** Mixed licensing:
- **Samples:** CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
  - Attribution required, noncommercial use only for distributing the samples themselves
  - For commercial use of the samples, contact jjlearman@gmail.com
  - https://creativecommons.org/licenses/by-nc/4.0/
- **Music made with the samples:** CC0 (no restrictions on your creative works)
- **Control files & example clips:** CC0

**`audio/rhodes/`** — 12 samples (velocity layer 2), converted from FLAC to MP3 with ffmpeg
(trimmed to 4 seconds with fade-out). These serve two roles:

1. **IR bed** — used in `generators/extract_rhodes_tf.py` to extract per-note transfer
   functions via spectral division (same approach as the grand piano soundboard IR). The
   resulting `generators/rhodes_tf.npz` captures the pickup, amp, and cabinet character
   across the keyboard, which `generators/generate_rhodes_fm.py` then convolves into the
   FM synthesis.
2. **In-browser reference** — included as the "Rhodes (ref)" instrument for A/B comparison
   against the FM synthesis.

---

## Open-Source Compatibility Note

- **Piano (Salamander):** Fully compatible with open-source distribution under CC BY 3.0. Just keep this attribution file.
- **Rhodes (jRhodes3d):** The CC BY-NC 4.0 license on the samples means they can be freely distributed in noncommercial/open-source projects. If the project is used commercially, the Rhodes samples would need to be replaced or a commercial license obtained from the author.
