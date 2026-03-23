/* ================================================================
   sidebar.js — renderAgentSidebar(), renderPlaceSidebar(), toggle
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.renderAgentSidebar = function(agent) {
  var el = document.getElementById('sb-content');
  var color = TOWN.colorToHex(TOWN.getAgentColor(agent.name));
  var esc = TOWN.escapeHtml;

  var html = '<div class="sb-section">';
  html += '<div class="sb-agent-name" style="color:' + color + '">';
  html += esc(agent.name.replace(/_/g, ' '));
  html += '</div>';

  /* Life stage and age */
  var stage = (agent.life_stage || 'adult');
  html += '<div class="sb-agent-info">' + esc(stage.charAt(0).toUpperCase() + stage.slice(1));
  if (agent.age_years !== undefined) html += ' &middot; Age ' + Math.floor(agent.age_years);
  html += '</div>';

  /* Current action */
  if (agent.action) {
    html += '<div class="sb-agent-info">Action: ' + esc(agent.action);
    if (agent.action_target) html += ' \u2192 ' + esc(agent.action_target);
    if (agent.target) html += ' \u2192 ' + esc(agent.target);
    html += '</div>';
  }

  /* Location */
  if (agent.location) {
    html += '<div class="sb-agent-info">Location: ' + esc(agent.location.replace(/_/g, ' ')) + '</div>';
  }

  /* Deceased marker */
  if (!agent.alive) {
    html += '<div class="sb-agent-info" style="color:#FF6B6B;font-weight:700">Deceased</div>';
  }
  html += '</div>';

  /* ── Drives ──────────────────────────────────────────────────── */
  if (agent.drives) {
    html += '<div class="sb-section"><h3>Drives</h3>';
    var driveKeys = Object.keys(agent.drives);
    for (var i = 0; i < driveKeys.length; i++) {
      var key = driveKeys[i];
      var val = agent.drives[key];
      var pct = Math.max(0, Math.min(100, val));
      var barColor = TOWN.DRIVE_COLORS[key] || '#888';
      html += '<div class="sb-drive-row">';
      html += '<span class="sb-drive-label">' + key + '</span>';
      html += '<div class="sb-drive-bar-bg">';
      html += '<div class="sb-drive-bar" style="width:' + pct + '%;background:' + barColor + '"></div>';
      html += '</div>';
      html += '<span class="sb-drive-val">' + Math.round(val) + '</span>';
      html += '</div>';
    }
    html += '</div>';
  }

  /* ── Relationships ───────────────────────────────────────────── */
  if (agent.relationships && agent.relationships.length > 0) {
    html += '<div class="sb-section"><h3>Relationships</h3>';
    for (var r = 0; r < agent.relationships.length; r++) {
      var rel = agent.relationships[r];
      var relColor;
      if (rel.valence > 0.3) relColor = '#6BCB77';
      else if (rel.valence < -0.3) relColor = '#FF6B6B';
      else relColor = '#D4A03C';
      html += '<div class="sb-rel-item">';
      html += '<div class="sb-rel-dot" style="background:' + relColor + '"></div>';
      html += '<span class="sb-rel-name">' + esc(rel.name.replace(/_/g, ' ')) + '</span>';
      html += '<span class="sb-rel-state">' + esc(rel.state || 'acquaintance') + '</span>';
      html += '</div>';
    }
    html += '</div>';
  }

  /* ── Last dialogue ───────────────────────────────────────────── */
  if (agent.dialogue) {
    html += '<div class="sb-section"><h3>Last Said</h3>';
    html += '<div class="sb-memory-item">\u201C' + esc(agent.dialogue) + '\u201D</div>';
    html += '</div>';
  }

  el.innerHTML = html;
};

TOWN.renderPlaceSidebar = function(placeKey) {
  var el = document.getElementById('sb-content');
  var place = TOWN.state.town && TOWN.state.town.places ? TOWN.state.town.places[placeKey] : null;
  if (!place) return;
  var esc = TOWN.escapeHtml;

  /* Gather occupants */
  var occupants = [];
  var all = TOWN.state.agents;
  for (var n in all) {
    if (all[n].location === placeKey && all[n].alive) occupants.push(all[n]);
  }

  var colorHex = place.color || '#888888';
  var html = '<div class="sb-section">';
  html += '<div class="sb-agent-name" style="color:' + colorHex + '">';
  html += esc(place.icon || '') + ' ' + esc(place.label || placeKey.replace(/_/g, ' '));
  html += '</div>';
  html += '</div>';

  /* Occupant list */
  html += '<div class="sb-section"><h3>Occupants (' + occupants.length + ')</h3>';
  if (occupants.length === 0) {
    html += '<div class="sb-agent-info" style="font-style:italic">Empty</div>';
  } else {
    for (var i = 0; i < occupants.length; i++) {
      var a = occupants[i];
      var ac = TOWN.colorToHex(TOWN.getAgentColor(a.name));
      html += '<div class="sb-rel-item" style="cursor:pointer" onclick="TOWN.selectAgent(\'' + a.name + '\')">';
      html += '<div class="sb-rel-dot" style="background:' + ac + '"></div>';
      html += '<span class="sb-rel-name">' + esc(a.name.replace(/_/g, ' ')) + '</span>';
      html += '<span class="sb-rel-state">' + esc(a.action || 'idle') + '</span>';
      html += '</div>';
    }
  }
  html += '</div>';

  /* Recent events at this place */
  var placeEvents = [];
  var log = TOWN.state.eventLog;
  var lowerKey = placeKey.toLowerCase();
  var lowerLabel = lowerKey.replace(/_/g, ' ');
  for (var j = Math.max(0, log.length - 50); j < log.length; j++) {
    var txt = log[j].text.toLowerCase();
    if (txt.indexOf(lowerKey) !== -1 || txt.indexOf(lowerLabel) !== -1) {
      placeEvents.push(log[j]);
    }
  }
  placeEvents = placeEvents.slice(-8);

  if (placeEvents.length > 0) {
    html += '<div class="sb-section"><h3>Recent Events</h3>';
    for (var k = 0; k < placeEvents.length; k++) {
      var evt = placeEvents[k];
      html += '<div class="sb-memory-item"><span class="log-time">' + evt.time + '</span> ' + esc(evt.text) + '</div>';
    }
    html += '</div>';
  }

  el.innerHTML = html;
};

TOWN.selectAgent = function(name) {
  TOWN.state.selectedAgent = name;
  TOWN.state.selectedPlace = null;
  TOWN.openSidebar();
  if (TOWN.state.agents[name]) {
    TOWN.renderAgentSidebar(TOWN.state.agents[name]);
  }
  /* Update selection ring + camera pan */
  var scene = TOWN.state.scene;
  if (scene) {
    var sprites = TOWN.state.agentSprites;
    for (var n in sprites) {
      if (n === name) {
        TOWN.showSelectionRing(scene, sprites[n]);
        TOWN.highlightAgent(scene, n);
        /* Smoothly pan camera to the selected agent */
        var sp = sprites[n];
        var headOffY = sp.radius * 2 + 14;
        scene.cameras.main.pan(sp.x, sp.y - headOffY / 2, 500, 'Quad.easeOut');
      } else if (sprites[n].selRing.alpha > 0) {
        scene.tweens.killTweensOf(sprites[n].selRing);
        sprites[n].selRing.setAlpha(0);
      }
    }
  }
};

TOWN.openSidebar = function() {
  document.getElementById('sidebar').classList.remove('collapsed');
  var btn = document.getElementById('sidebar-toggle');
  btn.classList.remove('collapsed-pos');
  btn.textContent = '\u203A';
};

TOWN.closeSidebar = function() {
  document.getElementById('sidebar').classList.add('collapsed');
  var btn = document.getElementById('sidebar-toggle');
  btn.classList.add('collapsed-pos');
  btn.textContent = '\u2039';
};

TOWN.toggleSidebar = function() {
  var sb = document.getElementById('sidebar');
  if (sb.classList.contains('collapsed')) {
    TOWN.openSidebar();
  } else {
    TOWN.closeSidebar();
  }
};

TOWN.showCulturePanel = function() {
  TOWN.state.selectedAgent = null;
  TOWN.state.selectedPlace = null;
  TOWN.openSidebar();
  var el = document.getElementById('sb-content');
  el.innerHTML = '<div class="sb-section"><div class="sb-agent-name" style="color:#D4A03C">&#127758; Town Beliefs</div><div class="sb-agent-info" style="font-style:italic">Loading…</div></div>';

  fetch('/culture?top_k=15')
    .then(function(r) { return r.json(); })
    .then(function(atoms) {
      var esc = TOWN.escapeHtml;
      var html = '<div class="sb-section"><div class="sb-agent-name" style="color:#D4A03C">&#127758; Town Beliefs</div>';
      if (!atoms || !atoms.length) {
        html += '<div class="sb-agent-info" style="font-style:italic">No shared beliefs yet — bridge discovery runs every 10th epoch.</div>';
      }
      html += '</div>';
      if (atoms && atoms.length) {
        html += '<div class="sb-section"><h3>Shared Knowledge</h3>';
        for (var i = 0; i < atoms.length; i++) {
          var a = atoms[i];
          var valColor = a.valence > 0.2 ? '#6BCB77' : a.valence < -0.2 ? '#FF6B6B' : '#D4A03C';
          html += '<div class="sb-rel-item">';
          html += '<div class="sb-rel-dot" style="background:' + valColor + '"></div>';
          html += '<span class="sb-rel-name">' + esc(a.content || a.label) + '</span>';
          html += '<span class="sb-rel-state">' + Math.round(a.confidence * 100) + '%</span>';
          html += '</div>';
        }
        html += '</div>';
      }
      document.getElementById('sb-content').innerHTML = html;
    })
    .catch(function() {
      document.getElementById('sb-content').innerHTML =
        '<div class="sb-section"><div class="sb-agent-info" style="color:#FF6B6B">Could not load culture data.</div></div>';
    });
};
