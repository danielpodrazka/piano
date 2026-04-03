(function () {

  /* ── Constants ─────────────────────────────────────────── */

  const MODULE_ID = 'rhythm';
  const BPM = 90;
  const BEAT_DUR = 60 / BPM;                   // seconds per quarter note (~0.667s)
  const TOLERANCE = BEAT_DUR * 0.25;           // ~167ms — forgiving but still rhythmic
  const PASS_THRESHOLD = 0.70;
  const AUTO_SUBMIT_EXTRA_BEATS = 2.5;         // wait this many beats after last expected beat

  const RHYTHM_PATTERNS = [
    {
      id: 'q-q-q-q',
      label: '4 Quarters',
      beats: [0, 1, 2, 3],
      timeSignature: 4,
    },
    {
      id: 'h-h',
      label: '2 Halves',
      beats: [0, 2],
      timeSignature: 4,
    },
    {
      id: 'q-q-h',
      label: '\u2669 \u2669 \uD834\uDD57',
      beats: [0, 1, 2],
      timeSignature: 4,
    },
    {
      id: 'h-q-q',
      label: '\uD834\uDD57 \u2669 \u2669',
      beats: [0, 2, 3],
      timeSignature: 4,
    },
    {
      id: 'e-e-q-q-q',
      label: '\u266A\u266A \u2669 \u2669 \u2669',
      beats: [0, 0.5, 1, 2, 3],
      timeSignature: 4,
    },
    {
      id: 'q-e-e-q-q',
      label: '\u2669 \u266A\u266A \u2669 \u2669',
      beats: [0, 1, 1.5, 2, 3],
      timeSignature: 4,
    },
    {
      id: 'q-q-e-e-q',
      label: '\u2669 \u2669 \u266A\u266A \u2669',
      beats: [0, 1, 2, 2.5, 3],
      timeSignature: 4,
    },
    {
      id: 'dq-e-q-q',
      label: '\u2669. \u266A \u2669 \u2669',
      beats: [0, 1.5, 2, 3],
      timeSignature: 4,
    },
    {
      id: '3/4-q-q-q',
      label: '3/4: \u2669\u2669\u2669',
      beats: [0, 1, 2],
      timeSignature: 3,
    },
    {
      id: 'trip-q-q',
      label: 'Triplet + \u2669\u2669',
      beats: [0, 0.333, 0.667, 1, 2, 3],
      timeSignature: 4,
    },
    // ── Swing patterns (L6) ──
    // Swing feel: "and" of each beat at 0.67 instead of 0.5
    {
      id: 'swing-basic',
      label: 'Swing \u266A\u266A',
      beats: [0, 0.67, 1, 1.67, 2, 2.67, 3, 3.67],
      timeSignature: 4,
      swing: true,
    },
    {
      id: 'swing-mixed',
      label: 'Swing mixed',
      beats: [0, 0.67, 1, 2, 2.67, 3],
      timeSignature: 4,
      swing: true,
    },
    // ── Reading patterns (L7) ──
    // No audio playback — user reads the visual blocks and taps the rhythm
    {
      id: 'read-quarters',
      label: 'Read: \u2669\u2669\u2669\u2669',
      beats: [0, 1, 2, 3],
      timeSignature: 4,
      reading: true,
    },
    {
      id: 'read-eighths',
      label: 'Read: \u266A\u266A\u266A\u266A',
      beats: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
      timeSignature: 4,
      reading: true,
    },
    {
      id: 'read-mixed',
      label: 'Read: mixed',
      beats: [0, 1, 1.5, 2, 3],
      timeSignature: 4,
      reading: true,
    },
  ];

  const LEVELS = [
    ['q-q-q-q', 'h-h', 'q-q-h', 'h-q-q'],
    ['e-e-q-q-q', 'q-e-e-q-q', 'q-q-e-e-q'],
    ['dq-e-q-q'],
    ['3/4-q-q-q'],
    ['trip-q-q'],
    ['swing-basic', 'swing-mixed'],
    ['read-quarters', 'read-eighths', 'read-mixed'],
  ];

  /* ── Closure state ─────────────────────────────────────── */

  let currentPattern = null;
  let trialStartTime = null;
  let phase = 'idle';      // 'countdown' | 'listening' | 'tapping' | 'scoring' | 'idle'
  let tapTimes = [];       // wall-clock timestamps (ms) of user taps
  let tapRecordStart = null; // wall-clock time when tapping phase began (ms)
  let autoSubmitTimer = null;
  let spacebarHandler = null;
  let scheduledNodes = [];    // track oscillator nodes for cleanup
  let pendingTimeouts = [];   // track setTimeout IDs for cleanup

  /* ── Helpers ────────────────────────────────────────────── */

  function getItems() {
    return RHYTHM_PATTERNS.map(p => ({ id: p.id, label: p.label }));
  }

  function getLevels() {
    return LEVELS;
  }

  function getPatternById(id) {
    return RHYTHM_PATTERNS.find(p => p.id === id) || null;
  }

  /* ── Keyboard listener management ──────────────────────── */

  function addSpacebarListener() {
    removeSpacebarListener();
    spacebarHandler = function (e) {
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        handleTap();
      }
    };
    document.addEventListener('keydown', spacebarHandler);
  }

  function removeSpacebarListener() {
    if (spacebarHandler) {
      document.removeEventListener('keydown', spacebarHandler);
      spacebarHandler = null;
    }
  }

  /* ── Audio: play rhythm pattern via oscillator ──────────── */

  function stopAllScheduled() {
    scheduledNodes.forEach(osc => {
      try { osc.stop(); } catch (e) { /* already stopped */ }
    });
    scheduledNodes = [];
    pendingTimeouts.forEach(id => clearTimeout(id));
    pendingTimeouts = [];
  }

  function playRhythmPattern(pattern) {
    const ctx = getAudioCtx();
    const startTime = ctx.currentTime + 0.05;

    pattern.beats.forEach(beat => {
      const time = startTime + beat * BEAT_DUR;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = 900;
      gain.gain.setValueAtTime(0.45, time);
      gain.gain.exponentialRampToValueAtTime(0.001, time + 0.08);
      osc.connect(gain);
      try { gain.connect(compressor); } catch (e) { gain.connect(ctx.destination); }
      osc.start(time);
      osc.stop(time + 0.09);
      scheduledNodes.push(osc);
      osc.onended = () => {
        const idx = scheduledNodes.indexOf(osc);
        if (idx !== -1) scheduledNodes.splice(idx, 1);
      };
    });

    const lastBeat = pattern.beats[pattern.beats.length - 1];
    const totalMs = (lastBeat * BEAT_DUR + 0.15 + 0.05) * 1000;
    return totalMs;
  }

  /* ── Audio: play swing pattern — uses the beat array directly
       (swing timing is already baked into the beat positions) ── */

  function playSwingPattern(pattern) {
    return playRhythmPattern(pattern);
  }

  /* ── UI rendering ───────────────────────────────────────── */

  function renderUI() {
    const area = document.getElementById('module-area');
    if (!area) return;

    const isReading = currentPattern && currentPattern.reading;
    area.innerHTML = `
      <button id="module-replay-btn">${isReading ? '&#x1F504; Retry' : '&#x1F50A; Replay'}</button>
      <div id="rhythm-visual" style="
        display:flex; gap:0; align-items:flex-end;
        background:#0a0a1a; border:1px solid #1a1a2e;
        border-radius:6px; padding:12px 16px;
        min-height:56px; min-width:320px;
        justify-content:flex-start;
      "></div>
      <div id="rhythm-status" style="
        font-size:0.95rem; color:#888; min-height:1.4em; text-align:center;
      ">Select a pattern to begin...</div>
      <div class="rhythm-tap-area" id="rhythm-tap-area">TAP &mdash; Press Space</div>
      <div id="rhythm-timing-display" style="
        font-size:0.8rem; color:#666; text-align:center; min-height:1.2em;
        font-family: monospace; max-width:400px;
      "></div>
    `;

    document.getElementById('module-replay-btn').addEventListener('click', () => {
      if (phase === 'tapping' || phase === 'listening') return;
      replayPattern();
    });

    document.getElementById('rhythm-tap-area').addEventListener('click', () => {
      handleTap();
    });
  }

  function renderRhythmVisual(pattern, userBeatTimes) {
    const container = document.getElementById('rhythm-visual');
    if (!container || !pattern) return;
    container.innerHTML = '';

    const sig = pattern.timeSignature || 4;
    const totalBeats = sig;
    const containerWidth = 288; // px

    // Build duration segments from beat positions
    // Each beat position marks the start of a note; duration = next beat - current (or end of measure)
    const beats = pattern.beats;
    const segments = beats.map((b, i) => {
      const next = (i + 1 < beats.length) ? beats[i + 1] : totalBeats;
      return { start: b, dur: next - b };
    });

    const unitW = containerWidth / totalBeats;
    const GAP = 3;

    segments.forEach((seg, i) => {
      const w = Math.max(unitW * seg.dur - GAP, 16);
      const isMatched = userBeatTimes
        ? userBeatTimes.some(ut => Math.abs(ut - seg.start) <= TOLERANCE * (1 / BEAT_DUR * 0.001 + 1))
        : false;

      // Colour by note length: whole=blue, half=cyan, quarter=green, eighth=yellow, triplet-e=orange
      let color = '#00ffff';
      if (seg.dur >= 3.5) color = '#4488ff';
      else if (seg.dur >= 1.8) color = '#00ccff';
      else if (seg.dur >= 0.9) color = '#00ff88';
      else if (seg.dur >= 0.45) color = '#ffcc00';
      else color = '#ff8800';

      if (userBeatTimes) {
        color = isMatched ? '#00ff44' : '#ff4444';
      }

      const h = Math.round(20 + seg.dur * 18);

      const block = document.createElement('div');
      block.style.cssText = `
        width:${w}px; height:${h}px;
        background:${color}22;
        border:1px solid ${color};
        border-radius:3px;
        margin-right:${GAP}px;
        flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        font-size:0.7rem; color:${color};
        transition: background 0.2s;
      `;

      // Label: note symbol
      let symbol = '\u2669';
      if (seg.dur >= 3.5) symbol = '\uD834\uDD5D';
      else if (seg.dur >= 1.8) symbol = '\uD834\uDD57';
      else if (seg.dur >= 0.9) symbol = '\u2669';
      else if (seg.dur >= 0.45) symbol = '\u266A';
      else symbol = '\u266C';

      block.textContent = symbol;
      container.appendChild(block);
    });
  }

  function setStatus(msg, color) {
    const el = document.getElementById('rhythm-status');
    if (el) {
      el.textContent = msg;
      el.style.color = color || '#888';
    }
  }

  function setTapAreaState(active) {
    const el = document.getElementById('rhythm-tap-area');
    if (!el) return;
    if (active) {
      el.style.borderColor = '#00ffff';
      el.style.color = '#00ffff';
      el.style.background = '#00ffff11';
    } else {
      el.style.borderColor = '#333';
      el.style.color = '#888';
      el.style.background = '#0a0a1a';
    }
  }

  function flashTap() {
    const el = document.getElementById('rhythm-tap-area');
    if (!el) return;
    el.classList.add('tapped');
    setTimeout(() => el.classList.remove('tapped'), 80);
  }

  /* ── Trial flow ─────────────────────────────────────────── */

  function startTrial(prevItemId) {
    // Clean up any previous state
    clearAutoSubmit();
    removeSpacebarListener();

    const allItems = getItems();
    const unlocked = getUnlockedItems(MODULE_ID, LEVELS, allItems);
    const selected = selectItem(MODULE_ID, unlocked, prevItemId);
    currentPattern = getPatternById(selected.id);
    trialStartTime = Date.now();
    tapTimes = [];
    tapRecordStart = null;

    renderUI();
    renderRhythmVisual(currentPattern, null);

    // Reading patterns: skip audio, go straight to tapping with visual displayed
    if (currentPattern.reading) {
      phase = 'tapping';
      setStatus('Read the rhythm and tap it! (Space / click)', '#00ffff');
      setTapAreaState(true);
      addSpacebarListener();
      // Auto-submit after enough time
      const lastBeat = currentPattern.beats[currentPattern.beats.length - 1];
      const autoSubmitMs = (lastBeat + AUTO_SUBMIT_EXTRA_BEATS + 2) * BEAT_DUR * 1000;
      autoSubmitTimer = setTimeout(() => {
        if (phase === 'tapping') submitTapping();
      }, autoSubmitMs);
      return;
    }

    phase = 'listening';
    setStatus('Listen\u2026', '#aaa');
    setTapAreaState(false);

    // Play the pattern — no count-in, just the rhythm itself
    stopAllScheduled();
    const durationMs = currentPattern.swing
      ? playSwingPattern(currentPattern)
      : playRhythmPattern(currentPattern);

    pendingTimeouts.push(setTimeout(() => {
      beginTapping();
    }, durationMs));
  }

  function replayPattern() {
    if (!currentPattern) return;

    // Reading patterns have no audio — just reset tapping phase
    if (currentPattern.reading) {
      clearAutoSubmit();
      removeSpacebarListener();
      tapTimes = [];
      tapRecordStart = null;
      phase = 'tapping';
      renderRhythmVisual(currentPattern, null);
      setStatus('Read the rhythm and tap it! (Space / click)', '#00ffff');
      setTapAreaState(true);
      addSpacebarListener();
      document.getElementById('rhythm-timing-display').textContent = '';
      const lastBeat = currentPattern.beats[currentPattern.beats.length - 1];
      const autoSubmitMs = (lastBeat + AUTO_SUBMIT_EXTRA_BEATS + 2) * BEAT_DUR * 1000;
      autoSubmitTimer = setTimeout(() => {
        if (phase === 'tapping') submitTapping();
      }, autoSubmitMs);
      return;
    }

    clearAutoSubmit();
    removeSpacebarListener();
    tapTimes = [];
    tapRecordStart = null;
    phase = 'listening';

    renderRhythmVisual(currentPattern, null);
    setStatus('Listen\u2026', '#aaa');
    setTapAreaState(false);
    document.getElementById('rhythm-timing-display').textContent = '';

    stopAllScheduled();
    const durationMs = currentPattern.swing
      ? playSwingPattern(currentPattern)
      : playRhythmPattern(currentPattern);
    pendingTimeouts.push(setTimeout(() => {
      beginTapping();
    }, durationMs));
  }

  function beginTapping() {
    // Go straight to tapping — the user heard the tempo from the pattern.
    // First tap = beat 0, so no count-in needed.
    tapTimes = [];
    tapRecordStart = null;
    phase = 'tapping';
    setStatus('Your turn! Tap the rhythm (Space / click)', '#00ffff');
    setTapAreaState(true);
    addSpacebarListener();

    // Auto-submit after enough time for the full pattern + grace period
    const lastBeat = currentPattern.beats[currentPattern.beats.length - 1];
    const autoSubmitMs = (lastBeat + AUTO_SUBMIT_EXTRA_BEATS) * BEAT_DUR * 1000;
    autoSubmitTimer = setTimeout(() => {
      if (phase === 'tapping') submitTapping();
    }, autoSubmitMs);
  }

  function handleTap() {
    if (phase !== 'tapping') return;
    flashTap();
    const now = Date.now();
    // First tap defines beat 0
    if (tapTimes.length === 0) {
      tapRecordStart = now;
    }
    tapTimes.push(now);
  }

  function clearAutoSubmit() {
    if (autoSubmitTimer !== null) {
      clearTimeout(autoSubmitTimer);
      autoSubmitTimer = null;
    }
  }

  /* ── Scoring ────────────────────────────────────────────── */

  function submitTapping() {
    clearAutoSubmit();
    removeSpacebarListener();
    setTapAreaState(false);

    // No taps at all — treat as AFK / not participating, just replay
    if (tapTimes.length === 0) {
      phase = 'idle';
      setStatus('No taps detected — try again', '#888');
      moduleNextTrial(() => startTrial(currentPattern ? currentPattern.id : null));
      return;
    }

    phase = 'scoring';
    setStatus('Scoring\u2026', '#888');

    const result = scoreRhythm(currentPattern, tapTimes);
    const correct = result.score >= PASS_THRESHOLD;
    const timeTaken = Date.now() - trialStartTime;

    recordModuleAnswer(MODULE_ID, currentPattern.id, correct, timeTaken, correct ? null : currentPattern.id);
    checkModuleLevelUnlock(MODULE_ID, LEVELS);
    updateModuleStatsDisplay(MODULE_ID);
    renderModuleCharts(MODULE_ID);
    renderModuleMasteryGrid(MODULE_ID);
    saveState();

    showFeedback(result, correct);
  }

  function scoreRhythm(pattern, tapTimestamps) {
    // No taps at all
    if (tapTimestamps.length === 0) {
      return { score: 0, matchedCount: 0, total: pattern.beats.length,
        matched: new Array(pattern.beats.length).fill(false), userBeatTimes: [], expected: pattern.beats };
    }

    // First tap = beat 0. Convert all taps to beat positions relative to first tap.
    const firstTap = tapTimestamps[0];
    const userBeatTimes = tapTimestamps.map(t => {
      return (t - firstTap) / 1000 / BEAT_DUR;
    });

    // Swing patterns use the swing-shifted beat positions as expected values.
    // The beats array already has swing timing baked in (0.67 instead of 0.5).
    const expected = pattern.beats;
    const matched = new Array(expected.length).fill(false);
    const usedTaps = new Array(userBeatTimes.length).fill(false);

    // Wider tolerance for swing — the feel is harder to nail precisely
    const baseTolerance = TOLERANCE / BEAT_DUR;
    const toleranceBeats = pattern.swing ? baseTolerance * 1.2 : baseTolerance;

    // Greedy match: for each expected beat find closest unused tap within tolerance
    for (let i = 0; i < expected.length; i++) {
      let bestDist = Infinity;
      let bestJ = -1;
      for (let j = 0; j < userBeatTimes.length; j++) {
        if (usedTaps[j]) continue;
        const dist = Math.abs(userBeatTimes[j] - expected[i]);
        if (dist < bestDist) {
          bestDist = dist;
          bestJ = j;
        }
      }
      if (bestJ !== -1 && bestDist <= toleranceBeats) {
        matched[i] = true;
        usedTaps[bestJ] = true;
      }
    }

    const matchedCount = matched.filter(Boolean).length;
    const score = matchedCount / expected.length;

    return {
      score,
      matchedCount,
      total: expected.length,
      matched,
      userBeatTimes,
      expected,
    };
  }

  /* ── Feedback display ───────────────────────────────────── */

  function showFeedback(result, correct) {
    const { matchedCount, total, matched, userBeatTimes, expected } = result;
    const scoreText = `${matchedCount}/${total} beats matched`;
    showResult(scoreText, correct ? 'correct' : 'incorrect');

    setStatus(
      correct ? `Correct! ${scoreText}` : `Wrong — ${scoreText}`,
      correct ? '#00ff88' : '#ff4444'
    );

    // Re-render visual with match colouring
    // Build a map of expected-beat -> matched status to colour blocks
    renderRhythmVisualWithResults(currentPattern, matched);

    // Show timing detail
    showTimingDetail(expected, userBeatTimes, matched);

    // Wait for user to continue
    moduleNextTrial(() => {
      phase = 'idle';
      startTrial(currentPattern.id);
    });
  }

  function renderRhythmVisualWithResults(pattern, matched) {
    const container = document.getElementById('rhythm-visual');
    if (!container || !pattern) return;
    container.innerHTML = '';

    const sig = pattern.timeSignature || 4;
    const totalBeats = sig;
    const containerWidth = 288;
    const GAP = 3;
    const unitW = containerWidth / totalBeats;

    pattern.beats.forEach((b, i) => {
      const next = (i + 1 < pattern.beats.length) ? pattern.beats[i + 1] : totalBeats;
      const dur = next - b;
      const w = Math.max(unitW * dur - GAP, 16);
      const isOk = matched[i];
      const color = isOk ? '#00ff44' : '#ff4444';
      const h = Math.round(20 + dur * 18);

      let symbol = '\u2669';
      if (dur >= 3.5) symbol = '\uD834\uDD5D';
      else if (dur >= 1.8) symbol = '\uD834\uDD57';
      else if (dur >= 0.9) symbol = '\u2669';
      else if (dur >= 0.45) symbol = '\u266A';
      else symbol = '\u266C';

      const block = document.createElement('div');
      block.style.cssText = `
        width:${w}px; height:${h}px;
        background:${color}22;
        border:2px solid ${color};
        border-radius:3px;
        margin-right:${GAP}px;
        flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        font-size:0.7rem; color:${color};
      `;
      block.textContent = symbol;
      container.appendChild(block);
    });
  }

  function showTimingDetail(expected, userBeatTimes, matched) {
    const el = document.getElementById('rhythm-timing-display');
    if (!el) return;

    const lines = [];

    lines.push('Expected   Your tap   Diff');
    expected.forEach((exp, i) => {
      // Find closest user tap
      let closest = null;
      let closestDist = Infinity;
      userBeatTimes.forEach(ut => {
        const d = Math.abs(ut - exp);
        if (d < closestDist) {
          closestDist = d;
          closest = ut;
        }
      });

      const expStr = exp.toFixed(2).padStart(6);
      const tapStr = closest !== null ? closest.toFixed(2).padStart(6) : ' (none)';
      const diffStr = closest !== null
        ? ((closest - exp >= 0 ? '+' : '') + (closest - exp).toFixed(2)).padStart(6)
        : '      ';
      const mark = matched[i] ? '\u2713' : '\u2717';
      lines.push(`${mark} beat ${expStr}  tap ${tapStr}  ${diffStr}`);
    });

    if (userBeatTimes.length > expected.length) {
      const extra = userBeatTimes.length - expected.length;
      lines.push(`(+${extra} extra tap${extra > 1 ? 's' : ''})`);
    }

    el.textContent = lines.join('\n');
    el.style.whiteSpace = 'pre';
  }

  /* ── Module registration ─────────────────────────────────── */

  registerModule(MODULE_ID, {
    label: 'Rhythm',
    getItems,
    getLevels,
    startTrial,
    renderUI,
    // handleAnswer and showFeedback are managed internally; stubs satisfy interface
    handleAnswer: function () {},
    showFeedback: function () {},
    cleanup: function () {
      stopAllScheduled();
      clearAutoSubmit();
      removeSpacebarListener();
      phase = 'idle';
    },
  });

})();
