(function () {
  let currentItem = null;
  let trialStartTime = null;
  let rootMidi = null;
  let awaitingAnswer = false;
  // 'off' | 'ascending' | 'descending'
  let arpMode = 'off';

  // Jazz chord intervals not in the global CHORD_INTERVALS
  const JAZZ_INTERVALS = {
    'dom7sharp5': [0, 4, 8, 10],  // root, M3, aug5, m7
    'dom7b5':     [0, 4, 6, 10],  // root, M3, dim5, m7
  };

  // Apply inversion: rotate bottom N notes up an octave
  function applyInversion(midiNotes, inversion) {
    const notes = midiNotes.slice();
    for (let i = 0; i < inversion; i++) {
      notes.push(notes.shift() + 12);
    }
    return notes;
  }

  // Parse an item id into its base quality and inversion number
  function parseItemId(id) {
    const match = id.match(/^(.+)-(1st|2nd|3rd)$/);
    if (!match) return { quality: id, inversion: 0 };
    const invMap = { '1st': 1, '2nd': 2, '3rd': 3 };
    return { quality: match[1], inversion: invMap[match[2]] };
  }

  // Build chord notes for any item id (handles inversions and jazz chords)
  function buildChordNotes(root, itemId) {
    const { quality, inversion } = parseItemId(itemId);
    let notes;
    if (JAZZ_INTERVALS[quality]) {
      notes = JAZZ_INTERVALS[quality].map(i => root + i);
    } else {
      notes = buildChord(root, quality);
    }
    return inversion > 0 ? applyInversion(notes, inversion) : notes;
  }

  const TRIAD_TYPES = ['major', 'minor', 'dim', 'aug', 'sus2', 'sus4'];
  const SEVENTH_TYPES = ['dom7', 'maj7', 'min7', 'dim7', 'hdim7', 'minmaj7'];

  // Label lookup for generating inversion labels
  const BASE_LABELS = {
    'major': 'Major', 'minor': 'Minor', 'dim': 'Dim', 'aug': 'Aug',
    'sus2': 'Sus2', 'sus4': 'Sus4',
    'dom7': 'Dom 7', 'maj7': 'Maj 7', 'min7': 'Min 7',
    'dim7': 'Dim 7', 'hdim7': 'Half-dim 7', 'minmaj7': 'Min/Maj 7',
  };

  const ALL_ITEMS = [
    // Root position triads
    { id: 'major',   label: 'Major'       },
    { id: 'minor',   label: 'Minor'       },
    { id: 'dim',     label: 'Diminished'  },
    { id: 'aug',     label: 'Augmented'   },
    { id: 'sus2',    label: 'Sus2'        },
    { id: 'sus4',    label: 'Sus4'        },
    // Root position 7ths
    { id: 'dom7',    label: 'Dom 7'       },
    { id: 'maj7',    label: 'Maj 7'       },
    { id: 'min7',    label: 'Min 7'       },
    { id: 'dim7',    label: 'Dim 7'       },
    { id: 'hdim7',   label: 'Half-dim 7'  },
    { id: 'minmaj7', label: 'Min/Maj 7'   },
    // Triad 1st inversions
    ...TRIAD_TYPES.map(t => ({ id: `${t}-1st`, label: `${BASE_LABELS[t]} 1st inv` })),
    // Triad 2nd inversions
    ...TRIAD_TYPES.map(t => ({ id: `${t}-2nd`, label: `${BASE_LABELS[t]} 2nd inv` })),
    // 7th chord 1st inversions
    ...SEVENTH_TYPES.map(t => ({ id: `${t}-1st`, label: `${BASE_LABELS[t]} 1st inv` })),
    // 7th chord 2nd inversions
    ...SEVENTH_TYPES.map(t => ({ id: `${t}-2nd`, label: `${BASE_LABELS[t]} 2nd inv` })),
    // 7th chord 3rd inversions
    ...SEVENTH_TYPES.map(t => ({ id: `${t}-3rd`, label: `${BASE_LABELS[t]} 3rd inv` })),
    // Jazz chords
    { id: 'dom7sharp5', label: 'Dom 7#5' },
    { id: 'dom7b5',     label: 'Dom 7b5' },
  ];

  const LEVELS = [
    ['major', 'minor'],
    ['dim', 'aug'],
    ['sus2', 'sus4'],
    ['dom7', 'maj7', 'min7'],
    ['dim7', 'hdim7', 'minmaj7'],
    // L6: triad inversions
    [...TRIAD_TYPES.map(t => `${t}-1st`), ...TRIAD_TYPES.map(t => `${t}-2nd`)],
    // L7: 7th chord inversions
    [...SEVENTH_TYPES.map(t => `${t}-1st`), ...SEVENTH_TYPES.map(t => `${t}-2nd`), ...SEVENTH_TYPES.map(t => `${t}-3rd`)],
    // L8: jazz chords
    ['dom7sharp5', 'dom7b5'],
  ];

  function getItems() {
    return ALL_ITEMS;
  }

  function getLevels() {
    return LEVELS;
  }

  function playCurrentChord() {
    if (rootMidi === null || currentItem === null) return;
    const notes = buildChordNotes(rootMidi, currentItem.id);
    const ordered = arpMode === 'descending' ? notes.slice().reverse() : notes;
    if (awaitingAnswer) {
      // Don't show keys on piano while answering — audio only
      if (arpMode !== 'off') {
        ordered.forEach((midi, i) => setTimeout(() => playMidiNote(midi, 1.5), i * 200));
      } else {
        notes.forEach(midi => playMidiNote(midi, 2.0));
      }
    } else {
      // After answering: show keys on piano during replay
      if (arpMode !== 'off') {
        playSequence(ordered, 200, 1.5);
      } else {
        playChord(notes, 2.0);
      }
    }
  }

  function startTrial(prevItemId) {
    awaitingAnswer = true;
    const unlocked = getUnlockedItems('chords', LEVELS, ALL_ITEMS);
    const selected = selectItem('chords', unlocked, prevItemId);
    currentItem = selected;
    rootMidi = 48 + Math.floor(Math.random() * 20); // C3 (48) to G4 (67)
    trialStartTime = Date.now();
    renderUI();
    playCurrentChord();
  }

  function renderUI() {
    const area = document.getElementById('module-area');
    if (!area) return;

    const unlocked = getUnlockedItems('chords', LEVELS, ALL_ITEMS);

    area.innerHTML = `
      <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap; justify-content:center;">
        <button id="module-replay-btn">&#x1F50A; Replay</button>
        <button id="chord-arp-toggle" class="module-btn" style="padding:6px 14px;font-size:0.85rem;">
          ${arpMode === 'off' ? 'Arpeggiate: OFF' : arpMode === 'ascending' ? 'Arpeggiate: UP' : 'Arpeggiate: DOWN'}
        </button>
      </div>
      <div class="module-answers">
        ${unlocked.map(item => `
          <button class="module-btn" data-id="${item.id}">${item.label}</button>
        `).join('')}
      </div>
    `;

    document.getElementById('module-replay-btn').addEventListener('click', () => {
      playCurrentChord();
    });

    document.getElementById('chord-arp-toggle').addEventListener('click', function () {
      arpMode = arpMode === 'off' ? 'ascending' : arpMode === 'ascending' ? 'descending' : 'off';
      this.textContent = arpMode === 'off' ? 'Arpeggiate: OFF' : arpMode === 'ascending' ? 'Arpeggiate: UP' : 'Arpeggiate: DOWN';
    });

    area.querySelectorAll('.module-answers .module-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!awaitingAnswer) return;
        handleAnswer(btn.dataset.id);
      });
    });
  }

  function handleAnswer(answerId) {
    if (!awaitingAnswer) return;
    awaitingAnswer = false;

    const timeTaken = Date.now() - trialStartTime;
    const correct = answerId === currentItem.id;
    const confusionId = correct ? null : answerId;

    recordModuleAnswer('chords', currentItem.id, correct, timeTaken, confusionId);
    checkModuleLevelUnlock('chords', LEVELS);
    updateModuleStatsDisplay('chords');
    renderModuleCharts('chords');
    renderModuleMasteryGrid('chords');
    saveState();

    showFeedback({ correct, answerId });
  }

  function showFeedback({ correct, answerId }) {
    const area = document.getElementById('module-area');
    if (!area) return;

    // Highlight buttons
    area.querySelectorAll('.module-answers .module-btn').forEach(btn => {
      if (btn.dataset.id === currentItem.id) {
        btn.classList.add('correct');
      } else if (!correct && btn.dataset.id === answerId) {
        btn.classList.add('incorrect');
      }
      btn.disabled = true;
    });

    // Show result text
    showResult(correct ? 'Correct!' : `Wrong — ${currentItem.label}`, correct ? 'correct' : 'incorrect');

    // Draw chord notes on staff
    drawStaff();
    const notes = buildChordNotes(rootMidi, currentItem.id);
    const noteColor = correct ? '#00ff00' : '#ff6666';
    notes.forEach(midi => {
      let info = midiToStaffNote(midi, 'treble');
      let clef = 'treble';
      if (!info) {
        info = midiToStaffNote(midi, 'bass');
        clef = 'bass';
      }
      if (!info) return;
      drawNoteOnStaff(info.note, noteColor, 320, clef, info.accidental);
    });

    // Wait for user to continue
    moduleNextTrial(() => startTrial(currentItem.id));
  }

  registerModule('chords', {
    label: 'Chords',
    getItems,
    getLevels,
    startTrial,
    renderUI,
    handleAnswer,
    showFeedback,
  });
})();
