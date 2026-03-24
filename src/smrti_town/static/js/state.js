/**
 * Global mutable state store for smrti-town.
 * All modules read/write this object directly.
 */

/** Escape HTML special characters. */
function _esc(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** Render a two-line stat cell for the building popup grid. */
function _statCell(label, value, color) {
  return '<div style="background:var(--surface);border-radius:4px;padding:4px 6px;">' +
    '<div style="font-size:10px;color:var(--text-dim);">' + _esc(label) + '</div>' +
    '<div style="font-size:13px;font-weight:600;color:' + (color || 'var(--text)') + ';">' + _esc(String(value)) + '</div>' +
    '</div>';
}

var GameState = {
  phase: PHASES.BOOT,
  tick: 0,
  calendar: { hour: 8, day: 1, season: 'spring', year: 1, time_of_day: 'morning' },
  agents: [],
  places: [],
  grid: { buildings: [] },
  economy: { treasury: 0, tax_rates: {}, income: 0, expenses: 0 },
  petitions: [],
  council: { members: [], pending_meeting: null },
  mayorCandidates: [],
  events: [],          // rolling event log [{text, type, time}]
  selectedBuilding: null, // building key for placement mode
  selectedAgent: null,    // agent name for sidebar inspector
  selectedPlace: null,    // place object for sidebar
  paused: false,
  connected: false,
  directorMode: 'routine',
  generatingMessage: '',
  generatingHint: '',
  gameOverReason: '',

  // Phaser scene references (set by scenes on create)
  phaserGame: null,

  // Agent sprite assignments (name -> person sprite key)
  _agentSpriteMap: {},
  _nextSpriteIdx: 0,

  /** Get a stable person sprite for an agent name. */
  spriteForAgent: function(name) {
    if (!this._agentSpriteMap[name]) {
      this._agentSpriteMap[name] = PERSON_SPRITES[this._nextSpriteIdx % PERSON_SPRITES.length];
      this._nextSpriteIdx++;
    }
    return this._agentSpriteMap[name];
  },
};
