(function () {
  let currentItem = null;
  let trialStartTime = null;
  let rootMidi = null;
  let awaitingAnswer = false;

  // Parse item ID to get base interval and direction
  // e.g. "P5" -> { base: "P5", direction: "asc" }
  // e.g. "P5-desc" -> { base: "P5", direction: "desc" }
  // e.g. "P5-harm" -> { base: "P5", direction: "harm" }
  function parseItemId(id) {
    if (id.endsWith('-desc')) return { base: id.slice(0, -5), direction: 'desc' };
    if (id.endsWith('-harm')) return { base: id.slice(0, -5), direction: 'harm' };
    return { base: id, direction: 'asc' };
  }

  function playInterval() {
    if (rootMidi === null || currentItem === null) return;
    const { base, direction } = parseItemId(currentItem.id);
    const semitones = INTERVAL_SEMITONES[base];
    if (direction === 'harm') {
      playChord([rootMidi, rootMidi + semitones], 1.5);
    } else if (direction === 'desc') {
      playMidiNote(rootMidi + semitones, 1.5);
      setTimeout(() => playMidiNote(rootMidi, 1.5), 500);
    } else {
      playMidiNote(rootMidi, 1.5);
      setTimeout(() => playMidiNote(rootMidi + semitones, 1.5), 500);
    }
  }

  const DIRECTION_LABELS = { asc: '\u2191', desc: '\u2193', harm: '\u2016' };

  function getItems() {
    const items = [];
    for (const [id, label] of Object.entries(INTERVAL_LABELS)) {
      items.push({ id, label: label + ' ' + DIRECTION_LABELS.asc });
      items.push({ id: id + '-desc', label: label + ' ' + DIRECTION_LABELS.desc });
      items.push({ id: id + '-harm', label: label + ' ' + DIRECTION_LABELS.harm });
    }
    return items;
  }

  function getLevels() {
    return [
      // Ascending (L1-L5)
      ['P5', 'P8', 'P4'],
      ['M3', 'm3'],
      ['M2', 'm2'],
      ['M6', 'm6'],
      ['m7', 'M7', 'TT'],
      // Descending (L6-L10)
      ['P5-desc', 'P8-desc', 'P4-desc'],
      ['M3-desc', 'm3-desc'],
      ['M2-desc', 'm2-desc'],
      ['M6-desc', 'm6-desc'],
      ['m7-desc', 'M7-desc', 'TT-desc'],
      // Harmonic (L11-L15)
      ['P5-harm', 'P8-harm', 'P4-harm'],
      ['M3-harm', 'm3-harm'],
      ['M2-harm', 'm2-harm'],
      ['M6-harm', 'm6-harm'],
      ['m7-harm', 'M7-harm', 'TT-harm'],
    ];
  }

  function startTrial(prevItemId) {
    awaitingAnswer = true;
    const allItems = getItems();
    const levels = getLevels();
    const unlocked = getUnlockedItems('intervals', levels, allItems);
    const selected = selectItem('intervals', unlocked, prevItemId);
    currentItem = selected;
    rootMidi = 48 + Math.floor(Math.random() * 25); // C3 (48) to C5 (72)
    trialStartTime = Date.now();
    renderUI();
    playInterval();
  }

  function renderUI() {
    const area = document.getElementById('module-area');
    if (!area) return;

    const unlocked = getUnlockedItems('intervals', getLevels(), getItems());

    area.innerHTML = `
      <button id="module-replay-btn" class="module-btn replay-btn">Replay</button>
      <div class="module-answers">
        ${unlocked.map(item => `
          <button class="module-btn" data-id="${item.id}">${item.label}</button>
        `).join('')}
      </div>
    `;

    document.getElementById('module-replay-btn').addEventListener('click', () => {
      playInterval();
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

    recordModuleAnswer('intervals', currentItem.id, correct, timeTaken, confusionId);
    checkModuleLevelUnlock('intervals', getLevels());
    updateModuleStatsDisplay('intervals');
    renderModuleCharts('intervals');
    renderModuleMasteryGrid('intervals');
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

    // Draw staff with both notes
    drawStaff();
    const { base, direction } = parseItemId(currentItem.id);
    const semitones = INTERVAL_SEMITONES[base];
    const topMidi = rootMidi + semitones;

    function drawNoteAt(midi, x) {
      let info = midiToStaffNote(midi, 'treble');
      let clef = 'treble';
      if (!info) {
        info = midiToStaffNote(midi, 'bass');
        clef = 'bass';
      }
      if (!info) return;
      drawNoteOnStaff(info.note, '#555', x, clef, info.accidental);
    }

    if (direction === 'harm') {
      // Draw stacked (same x position)
      drawNoteAt(rootMidi, 320);
      drawNoteAt(topMidi, 320);
    } else if (direction === 'desc') {
      drawNoteAt(topMidi, 280);
      drawNoteAt(rootMidi, 360);
    } else {
      drawNoteAt(rootMidi, 280);
      drawNoteAt(topMidi, 360);
    }

    // Wait for user to continue
    moduleNextTrial(() => startTrial(currentItem.id));
  }

  registerModule('intervals', {
    label: 'Intervals',
    getItems,
    getLevels,
    startTrial,
    renderUI,
    handleAnswer,
    showFeedback,
  });
})();
