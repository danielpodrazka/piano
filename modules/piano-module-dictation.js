/* ═══════════════════════════════════════════════════════
   DICTATION MODULE
   Hear a short melody, transcribe it by clicking the staff.
   ═══════════════════════════════════════════════════════ */

(function () {

  /* ─── Items ─── */
  const ALL_ITEMS = [
    { id: 'stepwise-3',   label: 'Steps (3)' },
    { id: 'stepwise-4',   label: 'Steps (4)' },
    { id: 'small_leaps-3',label: 'Sm. Leaps (3)' },
    { id: 'small_leaps-4',label: 'Sm. Leaps (4)' },
    { id: 'stepwise-5',   label: 'Steps (5)' },
    { id: 'small_leaps-5',label: 'Sm. Leaps (5)' },
    { id: 'leaps-3',      label: 'Leaps (3)' },
    { id: 'leaps-4',      label: 'Leaps (4)' },
    { id: 'leaps-5',      label: 'Leaps (5)' },
    { id: 'large_leaps-5',label: 'Lg. Leaps (5)' },
  ];

  const LEVELS = [
    ['stepwise-3', 'stepwise-4'],
    ['small_leaps-3', 'small_leaps-4'],
    ['stepwise-5', 'small_leaps-5'],
    ['leaps-3', 'leaps-4'],
    ['leaps-5', 'large_leaps-5'],
  ];

  /* ─── Melody generation ─── */
  const MOTION_RANGES = {
    stepwise:    [1, 2],
    small_leaps: [1, 5],
    leaps:       [1, 7],
    large_leaps: [1, 12],
  };

  const MIDI_MIN = 55;  // G3
  const MIDI_MAX = 79;  // G5

  function parseItemId(id) {
    // e.g. "small_leaps-4" → { motion: 'small_leaps', count: 4 }
    const lastDash = id.lastIndexOf('-');
    return {
      motion: id.slice(0, lastDash),
      count:  parseInt(id.slice(lastDash + 1), 10),
    };
  }

  function generateMelody(itemId) {
    const { motion, count } = parseItemId(itemId);
    const [minInterval, maxInterval] = MOTION_RANGES[motion];

    // Random starting note in C4–C5 range (MIDI 60–72)
    let root = 60 + Math.floor(Math.random() * 13);
    // Clamp root to valid range
    root = Math.max(MIDI_MIN, Math.min(MIDI_MAX, root));

    const melody = [root];

    for (let i = 1; i < count; i++) {
      let attempts = 0;
      let next;
      do {
        const interval = minInterval + Math.floor(Math.random() * (maxInterval - minInterval + 1));
        const direction = Math.random() < 0.5 ? 1 : -1;
        next = melody[melody.length - 1] + direction * interval;
        attempts++;
      } while (
        attempts < 30 &&
        (next < MIDI_MIN || next > MIDI_MAX || next === melody[melody.length - 1])
      );

      // If we failed to find a valid note, try forcing a valid step
      if (next < MIDI_MIN || next > MIDI_MAX || next === melody[melody.length - 1]) {
        const prev = melody[melody.length - 1];
        if (prev + 1 <= MIDI_MAX) {
          next = prev + 1;
        } else {
          next = prev - 1;
        }
      }

      melody.push(next);
    }

    return melody;
  }

  /* ─── Module-local state ─── */
  let currentItem    = null;
  let melodyMidi     = [];
  let userAnswer     = [];
  let trialStartTime = 0;
  let isAwaitingInput = false;
  let staffClickHandler = null;

  /* ─── Helpers ─── */
  const MODULE_ID = 'dictation';
  const FIRST_NOTE_X = 140;
  const NOTE_STEP_X  = 60;

  function noteXForIndex(i) {
    return FIRST_NOTE_X + i * NOTE_STEP_X;
  }

  function noteCountStatus() {
    return `Note ${userAnswer.length}/${melodyMidi.length}`;
  }

  function getStatusEl()  { return document.getElementById('dict-status'); }
  function getSubmitBtn() { return document.getElementById('dict-submit'); }

  /* ─── Redraw staff with current user answer ─── */
  function redrawWithUserNotes() {
    drawStaff();
    for (let i = 0; i < userAnswer.length; i++) {
      const midi = userAnswer[i];
      const info = midiToStaffNote(midi, 'treble');
      if (info) {
        drawNoteOnStaff(info.note, '#00ffff', noteXForIndex(i), 'treble', info.accidental);
      }
    }
    const statusEl = getStatusEl();
    if (statusEl) statusEl.textContent = noteCountStatus();
  }

  /* ─── Staff click handler ─── */
  function makeStaffClickHandler() {
    return function (e) {
      if (!isAwaitingInput) return;
      if (userAnswer.length >= melodyMidi.length) return;

      const rect = staffCanvas.getBoundingClientRect();
      const y = e.clientY - rect.top;
      const { clef, pos } = yToClefAndPos(y);

      // For dictation, we use the treble clef only
      const notes = TREBLE_NOTES;
      const clicked = notes.find(n => n.pos === pos);
      if (!clicked) return;

      userAnswer.push(clicked.midi);
      playMidiNote(clicked.midi, 0.5);
      redrawWithUserNotes();

      const statusEl = getStatusEl();
      if (statusEl) statusEl.textContent = noteCountStatus();

      // Enable submit when all notes entered; also auto-focus submit
      if (userAnswer.length === melodyMidi.length) {
        const submitBtn = getSubmitBtn();
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.focus();
        }
      }
    };
  }

  /* ─── Attach / detach staff click handler ─── */
  function attachStaffClickHandler() {
    if (staffClickHandler) {
      staffCanvas.removeEventListener('click', staffClickHandler);
    }
    staffClickHandler = makeStaffClickHandler();
    staffCanvas.addEventListener('click', staffClickHandler);
    staffCanvas.classList.add('clickable');
  }

  function detachStaffClickHandler() {
    if (staffClickHandler) {
      staffCanvas.removeEventListener('click', staffClickHandler);
      staffClickHandler = null;
    }
    staffCanvas.classList.remove('clickable');
  }

  /* ─── Render module UI ─── */
  function renderUI() {
    const area = document.getElementById('module-area');
    area.innerHTML = `
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; justify-content:center;">
        <button id="dict-replay" class="module-btn" style="min-width:90px;">&#x1F509; Replay</button>
        <span id="dict-status" style="color:#888; font-size:0.95rem; min-width:90px; text-align:center;">Note 0/${melodyMidi.length}</span>
        <button id="dict-submit" class="module-btn" style="min-width:90px;" disabled>Submit</button>
        <button id="dict-undo" class="module-btn" style="min-width:90px;">Undo</button>
      </div>
      <div style="color:#555; font-size:0.85rem; font-style:italic;">Click the staff to place notes</div>
    `;

    document.getElementById('dict-replay').addEventListener('click', () => {
      playSequence(melodyMidi, 500, 1.0);
    });

    document.getElementById('dict-submit').addEventListener('click', () => {
      if (userAnswer.length < melodyMidi.length) return;
      submitAnswer();
    });

    document.getElementById('dict-undo').addEventListener('click', () => {
      if (userAnswer.length === 0) return;
      userAnswer.pop();
      redrawWithUserNotes();
      const submitBtn = getSubmitBtn();
      if (submitBtn) submitBtn.disabled = true;
    });
  }

  /* ─── Start a new trial ─── */
  function startTrial(prevItemId) {
    // Clean up any previous click handler
    detachStaffClickHandler();
    isAwaitingInput = false;
    userAnswer = [];

    if (feedbackTimeout) {
      clearTimeout(feedbackTimeout);
      feedbackTimeout = null;
    }

    const unlocked = getUnlockedItems(MODULE_ID, LEVELS, ALL_ITEMS);
    if (unlocked.length === 0) return;

    currentItem = selectItem(MODULE_ID, unlocked, prevItemId);
    melodyMidi  = generateMelody(currentItem.id);

    // Draw empty staff
    drawStaff();

    // Render UI controls
    renderUI();

    // Show result area blank
    showResult('', '');

    trialStartTime  = Date.now();
    isAwaitingInput = true;

    // Play melody then attach click handler
    playSequence(melodyMidi, 500, 1.0).then(() => {
      attachStaffClickHandler();
    });
  }

  /* ─── Submit and score ─── */
  function submitAnswer() {
    if (!currentItem || !isAwaitingInput) return;
    isAwaitingInput = false;
    detachStaffClickHandler();

    const timeTaken = Date.now() - trialStartTime;
    const n = melodyMidi.length;
    let correctCount = 0;

    for (let i = 0; i < n; i++) {
      if (userAnswer[i] === melodyMidi[i]) correctCount++;
    }

    const score   = correctCount / n;
    const correct = score >= 0.7;

    // Show feedback on staff
    drawStaff();
    for (let i = 0; i < n; i++) {
      const expectedInfo = midiToStaffNote(melodyMidi[i], 'treble');
      const userMidi = userAnswer[i];
      const x = noteXForIndex(i);

      if (userMidi === melodyMidi[i]) {
        // Correct — draw green
        if (expectedInfo) {
          drawNoteOnStaff(expectedInfo.note, '#00ff00', x, 'treble', expectedInfo.accidental);
        }
      } else {
        // Wrong — draw user's answer in red, correct in green offset slightly
        if (userMidi !== undefined) {
          const userInfo = midiToStaffNote(userMidi, 'treble');
          if (userInfo) {
            drawNoteOnStaff(userInfo.note, '#ff4444', x, 'treble', userInfo.accidental);
          }
        }
        // Draw correct answer in green with a small x-offset so they don't overlap
        if (expectedInfo) {
          const correctX = userMidi !== undefined ? x + 16 : x;
          drawNoteOnStaff(expectedInfo.note, '#00ff00', correctX, 'treble', expectedInfo.accidental);
        }
      }
    }

    // Record result
    recordModuleAnswer(MODULE_ID, currentItem.id, correct, timeTaken, null);
    checkModuleLevelUnlock(MODULE_ID, LEVELS);
    updateModuleStatsDisplay(MODULE_ID);
    renderModuleCharts(MODULE_ID);
    renderModuleMasteryGrid(MODULE_ID);
    saveState();

    // Show result text
    const resultText = `${correctCount}/${n} notes correct`;
    showResult(resultText, correct ? 'correct' : 'incorrect');

    // Wait for user to continue
    const doneItemId = currentItem.id;
    moduleNextTrial(() => startTrial(doneItemId));
  }

  /* ─── Register module ─── */
  registerModule(MODULE_ID, {
    label: 'Dictation',

    getItems()  { return ALL_ITEMS; },
    getLevels() { return LEVELS;    },

    startTrial(prevItemId) {
      startTrial(prevItemId);
    },

    // Answers are handled internally via the submit button
    handleAnswer() {},

    cleanup() {
      detachStaffClickHandler();
      isAwaitingInput = false;
    },
  });

})();
