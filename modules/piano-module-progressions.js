(function () {
  // --- Progression definitions ---
  // Each progression maps to an array of {degree, quality} chord steps.
  // Degree is semitones above the tonic. Progressions with 3 unique chords are
  // padded to 4 by repeating the final chord so timing stays consistent.
  const PROGRESSIONS = {
    'I-IV-V-I': [
      { degree: 0,  quality: 'major' },
      { degree: 5,  quality: 'major' },
      { degree: 7,  quality: 'major' },
      { degree: 0,  quality: 'major' },
    ],
    'I-V-vi-IV': [
      { degree: 0,  quality: 'major' },
      { degree: 7,  quality: 'major' },
      { degree: 9,  quality: 'minor' },
      { degree: 5,  quality: 'major' },
    ],
    'ii-V-I': [
      { degree: 2,  quality: 'minor' },
      { degree: 7,  quality: 'major' },
      { degree: 0,  quality: 'major' },
      { degree: 0,  quality: 'major' }, // padded repeat
    ],
    'I-vi-IV-V': [
      { degree: 0,  quality: 'major' },
      { degree: 9,  quality: 'minor' },
      { degree: 5,  quality: 'major' },
      { degree: 7,  quality: 'major' },
    ],
    'vi-IV-I-V': [
      { degree: 9,  quality: 'minor' },
      { degree: 5,  quality: 'major' },
      { degree: 0,  quality: 'major' },
      { degree: 7,  quality: 'major' },
    ],
    'I-IV-vi-V': [
      { degree: 0,  quality: 'major' },
      { degree: 5,  quality: 'major' },
      { degree: 9,  quality: 'minor' },
      { degree: 7,  quality: 'major' },
    ],
    'I-bVII-IV-I': [
      { degree: 0,  quality: 'major' },
      { degree: 10, quality: 'major' },
      { degree: 5,  quality: 'major' },
      { degree: 0,  quality: 'major' },
    ],
    'i-bVI-bIII-bVII': [
      { degree: 0,  quality: 'minor' },
      { degree: 8,  quality: 'major' },
      { degree: 3,  quality: 'major' },
      { degree: 10, quality: 'major' },
    ],
    'iii-vi-ii-V': [
      { degree: 4,  quality: 'minor' },
      { degree: 9,  quality: 'minor' },
      { degree: 2,  quality: 'minor' },
      { degree: 7,  quality: 'major' },
    ],
    'I-vi-ii-V': [
      { degree: 0,  quality: 'major' },
      { degree: 9,  quality: 'minor' },
      { degree: 2,  quality: 'minor' },
      { degree: 7,  quality: 'major' },
    ],
    'i-bVII-bVI-V': [
      { degree: 0,  quality: 'minor' },
      { degree: 10, quality: 'major' },
      { degree: 8,  quality: 'major' },
      { degree: 7,  quality: 'major' },
    ],
    'I7-IV7-I7-V7': [
      { degree: 0,  quality: 'major' },
      { degree: 5,  quality: 'major' },
      { degree: 0,  quality: 'major' },
      { degree: 7,  quality: 'major' },
    ],
    'i-iv-v-i': [
      { degree: 0,  quality: 'minor' },
      { degree: 5,  quality: 'minor' },
      { degree: 7,  quality: 'minor' },
      { degree: 0,  quality: 'minor' },
    ],
  };

  const PROGRESSION_LABELS = {
    'I-IV-V-I':       'I - IV - V - I',
    'I-V-vi-IV':      'I - V - vi - IV',
    'ii-V-I':         'ii - V - I',
    'I-vi-IV-V':      'I - vi - IV - V',
    'vi-IV-I-V':      'vi - IV - I - V',
    'I-IV-vi-V':      'I - IV - vi - V',
    'I-bVII-IV-I':    'I - bVII - IV - I',
    'i-bVI-bIII-bVII':'i - bVI - bIII - bVII',
    'iii-vi-ii-V':    'iii - vi - ii - V',
    'I-vi-ii-V':      'I - vi - ii - V',
    'i-bVII-bVI-V':   'i - bVII - bVI - V',
    'I7-IV7-I7-V7':   'I7 - IV7 - I7 - V7',
    'i-iv-v-i':       'i - iv - v - i',
  };

  const MODULE_ID = 'progressions';

  // --- Local state ---
  let currentItem    = null;
  let currentTonic   = null;
  let currentChords  = null; // array of MIDI arrays, one per chord step
  let trialStartTime = null;
  let awaitingAnswer = false;

  // --- Helpers ---
  function getItems() {
    return Object.keys(PROGRESSIONS).map(id => ({
      id,
      label: PROGRESSION_LABELS[id],
    }));
  }

  function getLevels() {
    return [
      ['I-IV-V-I',    'I-V-vi-IV'],
      ['ii-V-I',      'I-vi-IV-V'],
      ['vi-IV-I-V',   'I-IV-vi-V'],
      ['I-bVII-IV-I', 'i-bVI-bIII-bVII'],
      ['iii-vi-ii-V', 'I-vi-ii-V'],
      ['i-bVII-bVI-V', 'I7-IV7-I7-V7', 'i-iv-v-i'],
    ];
  }

  // Build the 4-chord MIDI sequence for a given progression id and tonic.
  function buildChordSequence(progressionId, tonic) {
    const steps = PROGRESSIONS[progressionId];
    return steps.map(step => buildChord(tonic + step.degree, step.quality));
  }

  // Play the current chord sequence (also used by replay).
  function playProgression() {
    if (!currentChords) return;
    currentChords.forEach((chord, i) => {
      setTimeout(() => playChord(chord, 0.8), i * 800);
    });
  }

  // --- Trial lifecycle ---
  function startTrial(prevItemId) {
    awaitingAnswer = true;

    const allItems  = getItems();
    const levels    = getLevels();
    const unlocked  = getUnlockedItems(MODULE_ID, levels, allItems);
    const selected  = selectItem(MODULE_ID, unlocked, prevItemId);

    currentItem   = selected;
    currentTonic  = 48 + Math.floor(Math.random() * 6); // MIDI 48–53 (C3–F3)
    currentChords = buildChordSequence(currentItem.id, currentTonic);
    trialStartTime = Date.now();

    renderUI();
    playProgression();
  }

  // --- UI rendering ---
  function renderUI() {
    const area = document.getElementById('module-area');
    if (!area) return;

    const allItems = getItems();
    const levels   = getLevels();
    const unlocked = getUnlockedItems(MODULE_ID, levels, allItems);

    const buttonRows = unlocked.map(item => `
      <button class="module-btn wide" data-id="${item.id}">${item.label}</button>
    `).join('');

    area.innerHTML = `
      <button id="module-replay-btn" class="module-btn replay-btn">Replay</button>
      <div class="module-answers">
        ${buttonRows}
      </div>
    `;

    document.getElementById('module-replay-btn').addEventListener('click', () => {
      playProgression();
    });

    area.querySelectorAll('.module-answers .module-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!awaitingAnswer) return;
        handleAnswer(btn.dataset.id);
      });
    });
  }

  // --- Answer handling ---
  function handleAnswer(answerId) {
    if (!awaitingAnswer) return;
    awaitingAnswer = false;

    const timeTaken  = Date.now() - trialStartTime;
    const correct    = answerId === currentItem.id;
    const confusionId = correct ? null : answerId;

    recordModuleAnswer(MODULE_ID, currentItem.id, correct, timeTaken, confusionId);
    checkModuleLevelUnlock(MODULE_ID, getLevels());
    updateModuleStatsDisplay(MODULE_ID);
    renderModuleCharts(MODULE_ID);
    renderModuleMasteryGrid(MODULE_ID);
    saveState();

    showFeedback({ correct, answerId });
  }

  // --- Feedback display ---
  function showFeedback({ correct, answerId }) {
    const area = document.getElementById('module-area');
    if (!area) return;

    area.querySelectorAll('.module-answers .module-btn').forEach(btn => {
      if (btn.dataset.id === currentItem.id) {
        btn.classList.add('correct');
      } else if (!correct && btn.dataset.id === answerId) {
        btn.classList.add('incorrect');
      }
      btn.disabled = true;
    });

    showResult(
      correct ? 'Correct!' : `Wrong — ${currentItem.label}`,
      correct ? 'correct' : 'incorrect'
    );

    // Wait for user to continue
    moduleNextTrial(() => startTrial(currentItem.id));
  }

  // --- Register ---
  registerModule(MODULE_ID, {
    label: 'Progressions',
    getItems,
    getLevels,
    startTrial,
    renderUI,
    handleAnswer,
    showFeedback,
  });
})();
