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

TOWN.getPlaceCenter = function(placeName) {
  var p = TOWN.state.town && TOWN.state.town.places ? TOWN.state.town.places[placeName] : null;
  if (!p) return { x: 500, y: 350 };
  var w = p.w || 130, h = p.h || 100;
  return { x: p.x + w / 2, y: p.y + h / 2 };
};

TOWN.getAgentOffset = function(agentName, placeName) {
  var agents = [];
  var all = TOWN.state.agents;
  for (var n in all) {
    if (all[n].location === placeName && all[n].alive) agents.push(n);
  }
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
