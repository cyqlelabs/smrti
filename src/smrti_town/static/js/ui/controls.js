/* ================================================================
   controls.js — Play/pause, speed buttons, skip, keyboard shortcuts
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.initControls = function() {
  /* ── Play / Pause ────────────────────────────────────────────── */
  document.getElementById('btn-play').addEventListener('click', function() {
    TOWN.state.paused = !TOWN.state.paused;
    var btn = document.getElementById('btn-play');
    if (TOWN.state.paused) {
      btn.innerHTML = '\u25B6 Play';
      btn.classList.remove('active');
      TOWN.wsSend({ type: 'command', action: 'pause' });
    } else {
      btn.innerHTML = '\u23F8 Pause';
      btn.classList.add('active');
      TOWN.wsSend({ type: 'command', action: 'resume' });
    }
  });

  /* ── Speed toggle buttons ────────────────────────────────────── */
  var speedBtns = document.querySelectorAll('.speed-btn');
  for (var i = 0; i < speedBtns.length; i++) {
    (function(idx) {
      speedBtns[idx].addEventListener('click', function() {
        TOWN.state.speedIndex = idx;
        TOWN.state.speed = TOWN.SPEED_LEVELS[idx];
        /* Update active state */
        var all = document.querySelectorAll('.speed-btn');
        for (var j = 0; j < all.length; j++) {
          all[j].classList.toggle('speed-active', j === idx);
        }
        TOWN.wsSend({ type: 'command', action: 'set_speed', value: TOWN.SPEED_LEVELS[idx] });
      });
    })(i);
  }

  /* ── Skip buttons ────────────────────────────────────────────── */
  document.getElementById('btn-skip-morning').addEventListener('click', function() {
    TOWN.wsSend({ type: 'command', action: 'skip_morning' });
  });
  document.getElementById('btn-skip-week').addEventListener('click', function() {
    TOWN.wsSend({ type: 'command', action: 'skip_week' });
  });

  /* ── Sidebar toggle ──────────────────────────────────────────── */
  document.getElementById('sidebar-toggle').addEventListener('click', TOWN.toggleSidebar);

  /* ── Keyboard shortcuts ──────────────────────────────────────── */
  document.addEventListener('keydown', function(e) {
    /* Ignore if focused on an input */
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        document.getElementById('btn-play').click();
        break;
      case 'Digit1':
      case 'Numpad1':
        TOWN._setSpeed(0);
        break;
      case 'Digit2':
      case 'Numpad2':
        TOWN._setSpeed(1);
        break;
      case 'Digit3':
      case 'Numpad3':
        TOWN._setSpeed(2);
        break;
      case 'Digit4':
      case 'Numpad4':
        TOWN._setSpeed(3);
        break;
    }
  });
};

TOWN._setSpeed = function(idx) {
  if (idx < 0 || idx >= TOWN.SPEED_LEVELS.length) return;
  TOWN.state.speedIndex = idx;
  TOWN.state.speed = TOWN.SPEED_LEVELS[idx];
  var all = document.querySelectorAll('.speed-btn');
  for (var j = 0; j < all.length; j++) {
    all[j].classList.toggle('speed-active', j === idx);
  }
  TOWN.wsSend({ type: 'command', action: 'set_speed', value: TOWN.SPEED_LEVELS[idx] });
};
