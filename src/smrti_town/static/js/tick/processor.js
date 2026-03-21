/* ================================================================
   processor.js — processTick(), processInit()
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.processTick = function(scene, data) {
  if (data.type === 'state' || data.type === 'init') {
    TOWN._hideGenerating();
    TOWN.processInit(scene, data);
    return Promise.resolve();
  }

  if (data.type === 'generating') {
    TOWN._showGenerating(data.message || 'Generating world…', data.hint || '');
    return Promise.resolve();
  }

  if (data.type === 'dialogue_patch') {
    TOWN._applyDialoguePatch(scene, data);
    return Promise.resolve();
  }

  if (data.type === 'error') {
    TOWN._hideGenerating();
    TOWN.addLogEntry('system', '\u26A0\uFE0F ' + (data.message || 'Server error'));
    return Promise.resolve();
  }

  if (data.type !== 'tick') return Promise.resolve();

  /* ── Calendar ────────────────────────────────────────────────── */
  if (data.calendar) {
    TOWN.state.calendar = data.calendar;
    TOWN.updateClockUI();
  }

  /* ── Director mode ───────────────────────────────────────────── */
  if (data.director_mode) {
    TOWN.state.directorMode = data.director_mode;
    TOWN.updateDirectorBadge();
  }

  /* ── Tick number ─────────────────────────────────────────────── */
  if (data.tick_number !== undefined) {
    TOWN.state.tickNumber = data.tick_number;
  }

  /* ── Agents ──────────────────────────────────────────────────── */
  if (data.agents) {
    for (var i = 0; i < data.agents.length; i++) {
      var a = data.agents[i];
      /* Merge into existing agent data */
      if (!TOWN.state.agents[a.name]) {
        TOWN.state.agents[a.name] = a;
      } else {
        var existing = TOWN.state.agents[a.name];
        for (var key in a) {
          existing[key] = a[key];
        }
      }
      TOWN.updateAgentSprite(scene, TOWN.state.agents[a.name]);
    }
  }

  /* ── Occupant counts ─────────────────────────────────────────── */
  TOWN.updateOccupantCounts();

  /* ── Conversations (staggered for visual pacing) ─────────────── */
  if (data.conversations) {
    for (var ci = 0; ci < data.conversations.length; ci++) {
      (function(conv, delay) {
        setTimeout(function() {
          TOWN.showSpeechBubble(scene, conv.speaker, conv.content);
          TOWN.addLogEntry('talk', conv.speaker + ' to ' + conv.listener + ': \u201C' + conv.content + '\u201D');
        }, delay);
      })(data.conversations[ci], ci * 400);
    }
  }

  /* ── Events ──────────────────────────────────────────────────── */
  if (data.events) {
    for (var ei = 0; ei < data.events.length; ei++) {
      var evt = data.events[ei];
      var evtIcon = evt.icon || '\u2726';
      TOWN.addLogEntry('event', evtIcon + ' ' + evt.description);
    }
  }

  /* ── Births ──────────────────────────────────────────────────── */
  if (data.births) {
    for (var bi = 0; bi < data.births.length; bi++) {
      var b = data.births[bi];
      var birthName = b.child || b.name || 'Baby';
      var parents = b.parents || (b.parent_a && b.parent_b ? [b.parent_a, b.parent_b] : []);
      TOWN.addLogEntry('birth', '\uD83C\uDF1F ' + birthName + ' was born' + (parents.length ? ' to ' + parents.join(' & ') : '') + '!');
      /* Particles at parent location */
      var parentAgent = parents[0] ? TOWN.state.agents[parents[0]] : null;
      if (parentAgent) {
        var pos = TOWN.getPlaceCenter(parentAgent.location);
        TOWN.spawnParticles(scene, pos.x, pos.y, 'birth');
      }
    }
  }

  /* ── Deaths ──────────────────────────────────────────────────── */
  if (data.deaths) {
    for (var di = 0; di < data.deaths.length; di++) {
      var d = data.deaths[di];
      var dName = d.name || d;
      var dAge = d.age || (TOWN.state.agents[dName] ? Math.floor(TOWN.state.agents[dName].age_years) : '?');
      TOWN.addLogEntry('death', '\u271E ' + dName + ' passed away at age ' + dAge);
      var sp = TOWN.state.agentSprites[dName];
      if (sp) {
        TOWN.spawnParticles(scene, sp.x, sp.y, 'death');
      }
      if (TOWN.state.agents[dName]) {
        TOWN.state.agents[dName].alive = false;
      }
    }
  }

  /* ── Update sidebar if relevant ──────────────────────────────── */
  if (TOWN.state.selectedAgent && TOWN.state.agents[TOWN.state.selectedAgent]) {
    TOWN.renderAgentSidebar(TOWN.state.agents[TOWN.state.selectedAgent]);
  } else if (TOWN.state.selectedPlace) {
    TOWN.renderPlaceSidebar(TOWN.state.selectedPlace);
  }

  return TOWN.sleep(50);
};

TOWN.processInit = function(scene, data) {
  /* Stop demo mode if it was running */
  TOWN.state.demoMode = false;
  if (TOWN.state.demoInterval) {
    clearInterval(TOWN.state.demoInterval);
    TOWN.state.demoInterval = null;
  }

  var townData = data.town || data.data || data;

  /* ── Rebuild town from server data ───────────────────────────── */
  if (townData.places) {
    var newTown = {
      places: {},
      connections: townData.connections || TOWN.CONNECTIONS,
    };
    var placeNames = Object.keys(townData.places);
    for (var i = 0; i < placeNames.length; i++) {
      var key = placeNames[i];
      var p = townData.places[key];
      newTown.places[key] = {
        x: p.x || 400, y: p.y || 300,
        w: p.w || 130, h: p.h || 100,
        color: p.color || '#888888',
        icon: p.icon || '',
        label: p.label || key.replace(/_/g, ' '),
      };
    }
    /* Clear and redraw */
    scene.buildingLayer.removeAll(true);
    TOWN.state.placeSprites = {};
    scene.roadLayer.clear();
    TOWN.state.town = newTown;
    TOWN.drawTown(scene, newTown);
  }

  /* ── Spawn agents from init ──────────────────────────────────── */
  var agents = data.agents || townData.agents;
  if (agents) {
    for (var j = 0; j < agents.length; j++) {
      var a = agents[j];
      TOWN.state.agents[a.name] = a;
      TOWN.createAgentSprite(scene, a);
      if (a.drives && TOWN.state.agentSprites[a.name]) {
        TOWN.drawDriveBars(
          TOWN.state.agentSprites[a.name].driveBarContainer,
          TOWN.state.agentSprites[a.name].radius,
          a.drives
        );
      }
    }
    TOWN.updateOccupantCounts();
  }

  TOWN.addLogEntry('system', 'Connected to Smrti Town simulation');
};

/* ── Generating overlay ──────────────────────────────────────────── */

TOWN._showGenerating = function(msg, hint) {
  var el = document.getElementById('generating-overlay');
  if (!el) return;
  var msgEl = document.getElementById('gen-message');
  var hintEl = document.getElementById('gen-hint');
  if (msgEl) msgEl.textContent = msg;
  if (hintEl) hintEl.textContent = hint;
  el.classList.remove('hidden');
};

TOWN._hideGenerating = function() {
  var el = document.getElementById('generating-overlay');
  if (el) el.classList.add('hidden');
};

/* ── Dialogue patch ──────────────────────────────────────────────── */

TOWN._applyDialoguePatch = function(scene, data) {
  /* Update speech bubble if agent is still visible */
  var speaker = data.speaker;
  var content = data.content;
  if (!speaker || !content) return;

  /* Show updated speech bubble */
  if (scene) TOWN.showSpeechBubble(scene, speaker, content);

  /* Append enriched line to event log with a visual marker */
  var target = data.target || '';
  var label = target
    ? speaker + ' \u2192 ' + target + ': \u201C' + content + '\u201D'
    : speaker + ': \u201C' + content + '\u201D';
  TOWN.addLogEntry('talk', '\u2728 ' + label);
};
