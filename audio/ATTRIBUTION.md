# Audio Sample Attribution

## Piano Samples (`audio/piano/`)

**Source:** Salamander Grand Piano V3 by Alexander Holm
- Instrument: Yamaha C5 Grand Piano, recorded with two AKG C414 microphones in AB position
- Distributed via Tone.js CDN: https://tonejs.github.io/audio/salamander/
- Original project: https://sfzinstruments.github.io/pianos/salamander/
- Archive: https://archive.org/details/SalamanderGrandPianoV3
- GitHub: https://github.com/sfzinstruments/SalamanderGrandPiano

**License:** CC BY 3.0 (Creative Commons Attribution 3.0)
- You may share and adapt the samples for any purpose, including commercially
- You must give appropriate credit to Alexander Holm
- https://creativecommons.org/licenses/by/3.0/

**Processing:** 17 samples selected (every ~3 semitones from C2 to C6), served as MP3 from the Tone.js CDN. Intermediate notes are pitch-shifted at runtime via the Web Audio API.

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

**Processing:** 12 FLAC samples (velocity layer 2) downloaded, converted to MP3 with ffmpeg (trimmed to 4 seconds with fade-out). Intermediate notes are pitch-shifted at runtime via the Web Audio API.

---

## Open-Source Compatibility Note

- **Piano (Salamander):** Fully compatible with open-source distribution under CC BY 3.0. Just keep this attribution file.
- **Rhodes (jRhodes3d):** The CC BY-NC 4.0 license on the samples means they can be freely distributed in noncommercial/open-source projects. If the project is used commercially, the Rhodes samples would need to be replaced or a commercial license obtained from the author.
