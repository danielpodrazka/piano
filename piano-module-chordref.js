(function () {
  const CHORD_NAMES = {
    'major': 'Major', 'minor': 'Minor', 'dim': 'Dim', 'aug': 'Aug',
    'sus2': 'Sus2', 'sus4': 'Sus4',
    'dom7': 'Dom 7', 'maj7': 'Maj 7', 'min7': 'Min 7',
    'dim7': 'Dim 7', 'hdim7': 'Half-dim 7', 'minmaj7': 'Min/Maj 7',
  };

  const ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  let selectedRoot = 'C';
  let selectedOctave = 3;
  let activeBtn = null;
  // 'off' | 'ascending' | 'descending'
  let arpMode = 'off';
  let lastQuality = null;
  let _keyHandler = null;

  function rootToMidi(rootName, octave) {
    const idx = ROOTS.indexOf(rootName);
    return (octave + 1) * 12 + idx;
  }

  function playAndShow(quality) {
    lastQuality = quality;
    const root = rootToMidi(selectedRoot, selectedOctave);
    const notes = buildChord(root, quality);
    if (arpMode !== 'off') {
      const ordered = arpMode === 'descending' ? notes.slice().reverse() : notes;
      playSequence(ordered, 200, 1.5);
    } else {
      playChord(notes, 2.0);
    }
    drawStaff();
    notes.forEach(midi => {
      let info = midiToStaffNote(midi, 'treble');
      let clef = 'treble';
      if (!info) {
        info = midiToStaffNote(midi, 'bass');
        clef = 'bass';
      }
      if (!info) return;
      drawNoteOnStaff(info.note, '#00ffff', 320, clef, info.accidental);
    });
  }

  function shiftRoot(delta) {
    const idx = ROOTS.indexOf(selectedRoot);
    let newIdx = idx + delta;
    if (newIdx < 0) {
      newIdx = ROOTS.length - 1;
      if (selectedOctave > 2) selectedOctave--;
    } else if (newIdx >= ROOTS.length) {
      newIdx = 0;
      if (selectedOctave < 5) selectedOctave++;
    }
    selectedRoot = ROOTS[newIdx];
    renderUI();
    if (lastQuality) playAndShow(lastQuality);
  }

  function installKeyHandler() {
    if (_keyHandler) return;
    _keyHandler = function (e) {
      if (state.currentMode !== 'chordref') return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        shiftRoot(-1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        shiftRoot(1);
      }
    };
    document.addEventListener('keydown', _keyHandler);
  }

  function renderUI() {
    installKeyHandler();
    const area = document.getElementById('module-area');
    area.innerHTML = '';

    // Root selector row
    const rootRow = document.createElement('div');
    rootRow.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;justify-content:center;margin-bottom:4px;';
    ROOTS.forEach(r => {
      const btn = document.createElement('button');
      btn.className = 'module-btn';
      btn.textContent = r;
      btn.style.cssText = 'min-width:38px;padding:6px 8px;font-size:0.85rem;';
      if (r === selectedRoot) {
        btn.style.borderColor = '#00ffff';
        btn.style.color = '#00ffff';
        btn.style.background = '#00ffff15';
      }
      btn.addEventListener('click', () => {
        selectedRoot = r;
        renderUI();
        if (lastQuality) playAndShow(lastQuality);
      });
      rootRow.appendChild(btn);
    });
    area.appendChild(rootRow);

    // Octave selector
    const octRow = document.createElement('div');
    octRow.style.cssText = 'display:flex;gap:4px;align-items:center;margin-bottom:8px;';
    const octLabel = document.createElement('span');
    octLabel.textContent = 'Octave:';
    octLabel.style.cssText = 'color:#666;font-size:0.8rem;margin-right:4px;';
    octRow.appendChild(octLabel);
    for (let o = 2; o <= 5; o++) {
      const btn = document.createElement('button');
      btn.className = 'module-btn';
      btn.textContent = o;
      btn.style.cssText = 'min-width:32px;padding:4px 8px;font-size:0.85rem;';
      if (o === selectedOctave) {
        btn.style.borderColor = '#00ffff';
        btn.style.color = '#00ffff';
        btn.style.background = '#00ffff15';
      }
      btn.addEventListener('click', () => {
        selectedOctave = o;
        renderUI();
        if (lastQuality) playAndShow(lastQuality);
      });
      octRow.appendChild(btn);
    }
    area.appendChild(octRow);

    // Arpeggiate toggle
    const arpRow = document.createElement('div');
    arpRow.style.cssText = 'display:flex;gap:8px;align-items:center;margin-bottom:4px;';
    const arpBtn = document.createElement('button');
    arpBtn.className = 'module-btn';
    arpBtn.textContent = arpMode === 'off' ? 'Arpeggiate: OFF' : arpMode === 'ascending' ? 'Arpeggiate: UP' : 'Arpeggiate: DOWN';
    arpBtn.style.cssText = 'padding:6px 14px;font-size:0.85rem;';
    if (arpMode !== 'off') {
      arpBtn.style.borderColor = '#00ffff';
      arpBtn.style.color = '#00ffff';
      arpBtn.style.background = '#00ffff15';
    }
    arpBtn.addEventListener('click', () => {
      arpMode = arpMode === 'off' ? 'ascending' : arpMode === 'ascending' ? 'descending' : 'off';
      renderUI();
    });
    arpRow.appendChild(arpBtn);
    area.appendChild(arpRow);

    // Chord buttons grid
    const grid = document.createElement('div');
    grid.style.cssText = 'display:grid;grid-template-columns:repeat(4,1fr);max-width:480px;width:100%;';

    Object.entries(CHORD_NAMES).forEach(([quality, label]) => {
      const btn = document.createElement('button');
      btn.className = 'module-btn';
      btn.textContent = selectedRoot + ' ' + label;
      btn.style.cssText = 'aspect-ratio:1;border-radius:0;margin:0;min-width:0;width:100%;font-size:0.85rem;padding:4px;';
      if (quality === lastQuality) {
        btn.style.borderColor = '#00ffff';
        btn.style.color = '#00ffff';
        btn.style.background = '#00ffff15';
        activeBtn = btn;
      }
      btn.addEventListener('click', () => {
        if (activeBtn) {
          activeBtn.style.borderColor = '#333';
          activeBtn.style.color = '#e0e0e0';
          activeBtn.style.background = '#111';
        }
        btn.style.borderColor = '#00ffff';
        btn.style.color = '#00ffff';
        btn.style.background = '#00ffff15';
        activeBtn = btn;
        playAndShow(quality);
      });
      grid.appendChild(btn);
    });
    area.appendChild(grid);
  }

  registerModule('chordref', {
    label: 'Chord Ref',
    getItems: () => [],
    getLevels: () => [],
    startTrial: () => { renderUI(); },
    renderUI: renderUI,
    handleAnswer: () => {},
    showFeedback: () => {},
  });
})();
