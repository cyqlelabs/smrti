/* ================================================================
   state.js — Global state singleton
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.state = {
  ws: null,
  wsReady: false,
  paused: false,
  speed: 1,
  speedIndex: 0,
  tickQueue: [],
  processing: false,
  selectedAgent: null,
  selectedPlace: null,
  town: null,
  agents: {},
  agentSprites: {},
  placeSprites: {},
  calendar: { hour: 6, day: 1, season: 'spring', year: 1, time_of_day: 'morning' },
  directorMode: 'routine',
  tickNumber: 0,
  eventLog: [],
  scene: null,
  agentColorIdx: 0,
  demoMode: false,
  demoInterval: null,
  stars: [],
  townHealth: 50,
  milestones: new Set(),
  bubblePool: [],
  particlePool: [],
};

/* ── Helpers ─────────────────────────────────────────────────────── */

TOWN.sleep = function(ms) {
  return new Promise(function(r) { setTimeout(r, ms); });
};

TOWN.escapeHtml = function(s) {
  if (!s) return '';
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
          .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
};

TOWN.getAgentColor = function(name) {
  var st = TOWN.state;
  if (!st.agents[name]) return TOWN.AGENT_COLORS[0];
  if (st.agents[name]._color !== undefined) return st.agents[name]._color;
  var c = TOWN.AGENT_COLORS[st.agentColorIdx % TOWN.AGENT_COLORS.length];
  st.agentColorIdx++;
  st.agents[name]._color = c;
  return c;
};

TOWN.getAgentRadius = function(lifeStage) {
  switch (lifeStage) {
    case 'infant': return 12;
    case 'child':  return 20;
    case 'elder':  return 26;
    default:       return 28;
  }
};

/* ── Isometric projection ─────────────────────────────────────────── */

TOWN.ISO = {
  originX: 580,
  originY: 120,
  scaleX: 0.62,
  scaleY: 0.30,
};

/* Convert world (wx, wy, wz) → screen {x, y}.
   wz = height above ground (default 0). */
TOWN.isoProject = function(wx, wy, wz) {
  wz = wz || 0;
  return {
    x: (wx - wy) * TOWN.ISO.scaleX + TOWN.ISO.originX,
    y: (wx + wy) * TOWN.ISO.scaleY + TOWN.ISO.originY - wz * 0.9,
  };
};

/* Get iso screen position for the center of a place (ground level). */
TOWN.getPlaceCenter = function(placeName) {
  var places = TOWN.state.town && TOWN.state.town.places ? TOWN.state.town.places : null;
  var p = places ? places[placeName] : null;
  if (!p) {
    /* Fallback: try Main_Street, then a safe central default */
    var fallback = places && places['Main_Street'];
    if (fallback) {
      var fw = fallback.w || 130, fh = fallback.h || 100;
      return TOWN.isoProject(fallback.x + fw / 2, fallback.y + fh / 2, 0);
    }
    return TOWN.isoProject(440, 290, 0);
  }
  var w = p.w || 130, h = p.h || 100;
  return TOWN.isoProject(p.x + w / 2, p.y + h / 2, 0);
};

/* Location → alive agents cache, rebuilt once per tick number */
TOWN._locAgentsCache = {};
TOWN._locAgentsCacheTick = -1;

TOWN._rebuildLocCache = function() {
  var cache = {};
  var all = TOWN.state.agents;
  for (var n in all) {
    if (all[n].alive) {
      var loc = all[n].location;
      if (!cache[loc]) cache[loc] = [];
      cache[loc].push(n);
    }
  }
  TOWN._locAgentsCache = cache;
  TOWN._locAgentsCacheTick = TOWN.state.tickNumber;
};

TOWN.getAgentOffset = function(agentName, placeName) {
  if (TOWN._locAgentsCacheTick !== TOWN.state.tickNumber) {
    TOWN._rebuildLocCache();
  }
  var agents = TOWN._locAgentsCache[placeName] || [];
  var idx = agents.indexOf(agentName);
  var total = agents.length;
  if (total <= 1) return { dx: 0, dy: 0 };
  var angle = (idx / total) * Math.PI * 2 - Math.PI / 2;
  var spread = Math.min(35, 14 * total);
  return { dx: Math.cos(angle) * spread, dy: Math.sin(angle) * spread };
};

TOWN.colorToHex = function(colorInt) {
  return '#' + colorInt.toString(16).padStart(6, '0');
};
