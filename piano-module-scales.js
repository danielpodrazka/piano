(function () {
  const SCALE_PATTERNS = {
    major:         [0, 2, 4, 5, 7, 9, 11, 12],
    natural_minor: [0, 2, 3, 5, 7, 8, 10, 12],
    harmonic_minor:[0, 2, 3, 5, 7, 8, 11, 12],
    melodic_minor: [0, 2, 3, 5, 7, 9, 11, 12],
    dorian:        [0, 2, 3, 5, 7, 9, 10, 12],
    phrygian:      [0, 1, 3, 5, 7, 8, 10, 12],
    lydian:        [0, 2, 4, 6, 7, 9, 11, 12],
    mixolydian:    [0, 2, 4, 5, 7, 9, 10, 12],
    blues:         [0, 3, 5, 6, 7, 10, 12],
    major_pent:    [0, 2, 4, 7, 9, 12],
    minor_pent:    [0, 3, 5, 7, 10, 12],
    whole_tone:    [0, 2, 4, 6, 8, 10, 12],
    bebop_dom:     [0, 2, 4, 5, 7, 9, 10, 11, 12],
    bebop_major:   [0, 2, 4, 5, 7, 8, 9, 11, 12],
    altered:       [0, 1, 3, 4, 6, 8, 10, 12],
    dim_wh:        [0, 2, 3, 5, 6, 8, 9, 11, 12],
    dim_hw:        [0, 1, 3, 4, 6, 7, 9, 10, 12],
  };

  const SCALE_LABELS = {
    major:         'Major',
    natural_minor: 'Natural Minor',
    harmonic_minor:'Harmonic Minor',
    melodic_minor: 'Melodic Minor',
    dorian:        'Dorian',
    phrygian:      'Phrygian',
    lydian:        'Lydian',
    mixolydian:    'Mixolydian',
    blues:         'Blues',
    major_pent:    'Major Pent.',
    minor_pent:    'Minor Pent.',
    whole_tone:    'Whole Tone',
    bebop_dom:     'Bebop Dom.',
    bebop_major:   'Bebop Maj.',
    altered:       'Altered',
    dim_wh:        'Dim. W-H',
    dim_hw:        'Dim. H-W',
  };

  let currentItem = null;
  let rootMidi = null;
  let scaleNotes = null;
  let trialStartTime = null;
  let isAwaitingAnswer = false;

  function buildScaleNotes(root, scaleId) {
    return SCALE_PATTERNS[scaleId].map(interval => root + interval);
  }

  function playScale() {
    if (rootMidi === null || currentItem === null || scaleNotes === null) return;
    playSequence(scaleNotes, 300, 1.0);
  }

  function getItems() {
    return Object.entries(SCALE_LABELS).map(([id, label]) => ({ id, label }));
  }

  function getLevels() {
    return [
      ['major', 'natural_minor'],
      ['harmonic_minor', 'melodic_minor'],
      ['dorian', 'mixolydian'],
      ['phrygian', 'lydian'],
      ['blues', 'major_pent', 'minor_pent'],
      ['whole_tone'],
      ['bebop_dom', 'bebop_major'],
      ['altered', 'dim_wh', 'dim_hw'],
    ];
  }

  function startTrial(prevItemId) {
    isAwaitingAnswer = true;
    const allItems = getItems();
    const levels = getLevels();
    const unlocked = getUnlockedItems('scales', levels, allItems);
    const selected = selectItem('scales', unlocked, prevItemId);
    currentItem = selected;
    rootMidi = 48 + Math.floor(Math.random() * 13); // C3 (48) to C4 (60)
    scaleNotes = buildScaleNotes(rootMidi, currentItem.id);
    trialStartTime = Date.now();
    renderUI();
    playScale();
  }

  function renderUI() {
    const area = document.getElementById('module-area');
    if (!area) return;

    const allItems = getItems();
    const levels = getLevels();
    const unlocked = getUnlockedItems('scales', levels, allItems);

    area.innerHTML = `
      <button id="module-replay-btn">&#128266; Replay</button>
      <div class="module-answers">
        ${unlocked.map(item => `
          <button class="module-btn wide" data-id="${item.id}">${item.label}</button>
        `).join('')}
      </div>
    `;

    document.getElementById('module-replay-btn').addEventListener('click', () => {
      playScale();
    });

    area.querySelectorAll('.module-answers .module-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!isAwaitingAnswer) return;
        handleAnswer(btn.dataset.id);
      });
    });
  }

  function handleAnswer(answerId) {
    if (!isAwaitingAnswer) return;
    isAwaitingAnswer = false;

    const timeTaken = Date.now() - trialStartTime;
    const correct = answerId === currentItem.id;
    const confusionId = correct ? null : answerId;

    recordModuleAnswer('scales', currentItem.id, correct, timeTaken, confusionId);
    checkModuleLevelUnlock('scales', getLevels());
    updateModuleStatsDisplay('scales');
    renderModuleCharts('scales');
    renderModuleMasteryGrid('scales');
    saveState();

    showFeedback({ correct, answerId });
  }

  function showFeedback({ correct, answerId }) {
    const area = document.getElementById('module-area');
    if (!area) return;

    // Highlight answer buttons
    area.querySelectorAll('.module-answers .module-btn').forEach(btn => {
      if (btn.dataset.id === currentItem.id) {
        btn.classList.add('correct');
      } else if (!correct && btn.dataset.id === answerId) {
        btn.classList.add('incorrect');
      }
      btn.disabled = true;
    });

    // Show result text
    showResult(
      correct ? 'Correct!' : `Wrong — ${currentItem.label}`,
      correct ? 'correct' : 'incorrect'
    );

    // Draw scale notes on staff
    drawStaff();
    const clef = rootMidi >= 60 ? 'treble' : 'bass';
    const xStart = 100;
    const xSpacing = 50;

    scaleNotes.forEach((midi, i) => {
      const info = midiToStaffNote(midi, clef);
      if (!info) return;
      const x = xStart + i * xSpacing;
      const color = correct ? '#00cc44' : '#888888';
      drawNoteOnStaff(info.note, color, x, clef, info.accidental);
    });

    // Wait for user to continue
    moduleNextTrial(() => startTrial(currentItem.id));
  }

  registerModule('scales', {
    label: 'Scales',
    getItems,
    getLevels,
    startTrial,
    renderUI,
    handleAnswer,
    showFeedback,
  });
})();
