/* ================================================================
   settings.js — LLM settings modal: open/close, fetch, save, regenerate
   ================================================================ */
window.TOWN = window.TOWN || {};

/* ── Field map: DOM id → settings key ───────────────────────────── */
var FIELD_MAP = [
  { id: 'set-enabled',            key: 'enabled',              type: 'bool'   },
  { id: 'set-base-url',           key: 'base_url',             type: 'str'    },
  { id: 'set-model',              key: 'model',                type: 'str'    },
  { id: 'set-dialogue-timeout',   key: 'dialogue_timeout',     type: 'float'  },
  { id: 'set-worldgen-timeout',   key: 'worldgen_timeout',     type: 'float'  },
  { id: 'set-temperature',        key: 'temperature',          type: 'float'  },
  { id: 'set-top-p',              key: 'top_p',                type: 'float'  },
  { id: 'set-max-tokens',         key: 'max_tokens',           type: 'int'    },
  { id: 'set-theme',              key: 'world_theme',          type: 'str'    },
  { id: 'set-worldgen-tokens',    key: 'worldgen_max_tokens',  type: 'int'    },
];

/* ── Open / Close ────────────────────────────────────────────────── */
TOWN.openSettings = function() {
  TOWN._fetchSettings(function() {
    document.getElementById('settings-overlay').classList.remove('hidden');
  });
};

TOWN.closeSettings = function() {
  document.getElementById('settings-overlay').classList.add('hidden');
};

/* ── Fetch settings from server ──────────────────────────────────── */
TOWN._fetchSettings = function(cb) {
  fetch('/settings')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      TOWN._populateForm(data);
      if (cb) cb();
    })
    .catch(function(err) {
      console.warn('Could not load settings:', err);
      if (cb) cb();
    });
};

TOWN._populateForm = function(data) {
  FIELD_MAP.forEach(function(f) {
    var el = document.getElementById(f.id);
    if (!el || data[f.key] === undefined) return;
    if (f.type === 'bool') {
      el.checked = !!data[f.key];
    } else {
      el.value = data[f.key];
    }
  });
};

/* ── Collect form values ─────────────────────────────────────────── */
TOWN._collectForm = function() {
  var out = {};
  FIELD_MAP.forEach(function(f) {
    var el = document.getElementById(f.id);
    if (!el) return;
    if (f.type === 'bool') {
      out[f.key] = el.checked;
    } else if (f.type === 'int') {
      out[f.key] = parseInt(el.value, 10) || 0;
    } else if (f.type === 'float') {
      out[f.key] = parseFloat(el.value) || 0;
    } else {
      out[f.key] = el.value;
    }
  });
  return out;
};

/* ── Save ────────────────────────────────────────────────────────── */
TOWN._saveSettings = function() {
  var data = TOWN._collectForm();
  var btn = document.getElementById('btn-settings-save');
  btn.textContent = 'Saving…';
  btn.disabled = true;

  fetch('/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
    .then(function(r) { return r.json(); })
    .then(function() {
      TOWN.closeSettings();
    })
    .catch(function(err) {
      console.error('Save failed:', err);
      alert('Failed to save settings. See console for details.');
    })
    .finally(function() {
      btn.textContent = 'Save';
      btn.disabled = false;
    });
};

/* ── Regenerate world ────────────────────────────────────────────── */
TOWN._regenerateWorld = function() {
  var regenBtn = document.getElementById('btn-settings-regenerate');
  var saveBtn  = document.getElementById('btn-settings-save');

  /* First save current form values, then regenerate */
  var data = TOWN._collectForm();
  regenBtn.textContent = 'Generating…';
  regenBtn.disabled = true;
  saveBtn.disabled = true;

  fetch('/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
    .then(function() {
      return fetch('/regenerate', { method: 'POST' });
    })
    .then(function(r) { return r.json(); })
    .then(function() {
      /* Server returns 202 immediately; actual generation happens in background.
         Close modal right away — the generating overlay will appear via WebSocket. */
      TOWN.closeSettings();
    })
    .catch(function(err) {
      console.error('Regenerate failed:', err);
      alert('Failed to start world regeneration. See console for details.');
    })
    .finally(function() {
      regenBtn.innerHTML = '&#9851; Regenerate World';
      regenBtn.disabled = false;
      saveBtn.disabled = false;
    });
};

/* ── Handle reset message from server ───────────────────────────── */
TOWN._handleReset = function() {
  /* Clear agent sprites and state so the new world renders cleanly */
  var scene = TOWN.state.scene;
  if (scene) {
    var sprites = TOWN.state.agentSprites;
    for (var n in sprites) {
      var s = sprites[n];
      if (s.circle)   s.circle.destroy();
      if (s.label)    s.label.destroy();
      if (s.selRing)  s.selRing.destroy();
      if (s.bubble)   s.bubble.destroy();
    }
  }
  TOWN.state.agents        = {};
  TOWN.state.agentSprites  = {};
  TOWN.state.agentColorIdx = 0;
  TOWN.state.selectedAgent = null;
  TOWN.state.selectedPlace = null;
  TOWN.state.eventLog      = [];
  TOWN.state.tickNumber    = 0;

  var sb = document.getElementById('sb-content');
  if (sb) sb.innerHTML = '<div class="sb-empty">Generating new world…</div>';

  var log = document.getElementById('event-log');
  if (log) log.innerHTML = '';
};

/* ── Wire up event listeners ─────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('btn-settings').addEventListener('click', TOWN.openSettings);
  document.getElementById('settings-close').addEventListener('click', TOWN.closeSettings);
  document.getElementById('btn-settings-cancel').addEventListener('click', TOWN.closeSettings);
  document.getElementById('btn-settings-save').addEventListener('click', TOWN._saveSettings);
  document.getElementById('btn-settings-regenerate').addEventListener('click', TOWN._regenerateWorld);

  /* Close on overlay click (outside modal) */
  document.getElementById('settings-overlay').addEventListener('click', function(e) {
    if (e.target === this) TOWN.closeSettings();
  });

  /* Keyboard: Escape closes modal */
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      var overlay = document.getElementById('settings-overlay');
      if (!overlay.classList.contains('hidden')) TOWN.closeSettings();
    }
  });
});
