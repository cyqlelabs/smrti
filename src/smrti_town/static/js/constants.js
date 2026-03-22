/* ================================================================
   constants.js — Color palettes, drive colors, season/time configs
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.AGENT_COLORS = [
  0xFF6B6B, 0x4ECDC4, 0xFFD93D, 0x6BCB77, 0x4D96FF,
  0xFF6F91, 0x845EC2, 0xFF9671, 0xE8734A, 0x2A7B7F,
  0xD4A03C, 0x5C9E5C, 0x8B4C8B, 0x00B894, 0xF9CA24,
  0xBADC58, 0xE056A0, 0x30336B, 0x22A6B3, 0xBE2EDD,
];

TOWN.DRIVE_COLORS = {
  hunger:    '#FF6B6B',
  energy:    '#6BCB77',
  social:    '#4ECDC4',
  curiosity: '#FFD93D',
  duty:      '#845EC2',
  romance:   '#FF6F91',
};

TOWN.SEASON_COLORS = {
  spring: '#5C9E5C',
  summer: '#E8734A',
  autumn: '#D4A03C',
  winter: '#6BA3BE',
};

TOWN.SEASON_TINTS = {
  spring: { overlay: 0x5C9E5C, alpha: 0.03 },
  summer: { overlay: 0xE8734A, alpha: 0.04 },
  autumn: { overlay: 0xD4A03C, alpha: 0.05 },
  winter: { overlay: 0x6BA3BE, alpha: 0.04 },
};

TOWN.TOD_COLORS = {
  morning:   { bg: 0xF5E6D0, overlay: 0x000000, alpha: 0.0  },
  afternoon: { bg: 0xF0DEC0, overlay: 0xD4A03C, alpha: 0.04 },
  evening:   { bg: 0xC08040, overlay: 0xE07020, alpha: 0.18  },
  night:     { bg: 0x1a1a3e, overlay: 0x000020, alpha: 0.42  },
};

TOWN.TOD_ICONS = {
  morning:   '\u2600\uFE0F',
  afternoon: '\uD83C\uDF24\uFE0F',
  evening:   '\uD83C\uDF05',
  night:     '\uD83C\uDF19',
};

TOWN.LOG_COLORS = {
  talk:   'log-talk',
  move:   'log-move',
  event:  'log-event',
  birth:  'log-birth',
  death:  'log-death',
  system: 'log-system',
};

TOWN.SPEED_LEVELS = [1, 2, 5, 10];
TOWN.SPEED_LABELS = ['1x', '2x', '5x', '10x'];

/* Action → emoji icon shown above agent head */
TOWN.ACTION_ICONS = {
  sleep:   '\uD83D\uDCA4',  /* 💤 */
  eat:     '\uD83C\uDF7D',  /* 🍽 */
  work:    '\uD83D\uDD27',  /* 🔧 */
  talk:    '\uD83D\uDCAC',  /* 💬 */
  walk:    '',
  read:    '\uD83D\uDCDA',  /* 📖 */
  relax:   '\u2615',        /* ☕ */
  romance: '\uD83D\uDC95',  /* 💕 */
  gather:  '\uD83C\uDF89',  /* 🎉 */
  idle:    '',
};

/* Fallback building color by place_type when no explicit color supplied */
TOWN.PLACE_TYPE_COLORS = {
  home:       '#C8854A',
  commercial: '#4D96FF',
  work:       '#845EC2',
  public:     '#8B4C8B',
  outdoor:    '#5C9E5C',
  other:      '#888888',
};

TOWN.PLACES = {
  Cafe_Rosetta:    { x: 320, y: 180, w: 160, h: 110, color: '#E8734A', icon: '\u2615',         label: 'Cafe Rosetta' },
  Public_Library:  { x: 560, y: 160, w: 160, h: 110, color: '#2A7B7F', icon: '\uD83D\uDCDA',   label: 'Library' },
  Central_Park:    { x: 440, y: 400, w: 200, h: 140, color: '#5C9E5C', icon: '\uD83C\uDF33',   label: 'Central Park' },
  Town_Market:     { x: 720, y: 360, w: 160, h: 110, color: '#8B4C8B', icon: '\uD83C\uDFEA',   label: 'Market' },
  Alice_Home:      { x: 140, y: 380, w: 130, h: 100, color: '#D4A03C', icon: '\uD83C\uDFE0',   label: 'Alice Home' },
  Sofia_Home:      { x: 140, y: 160, w: 130, h: 100, color: '#C4873C', icon: '\uD83C\uDFE1',   label: 'Sofia Home' },
  Main_Street:     { x: 440, y: 290, w: 320, h: 36,  color: '#B8A88A', icon: '',                label: 'Main Street' },
  Elm_Street:      { x: 160, y: 290, w: 160, h: 30,  color: '#C4B898', icon: '',                label: 'Elm St' },
};

TOWN.CONNECTIONS = [
  ['Cafe_Rosetta',   'Main_Street'],
  ['Public_Library',  'Main_Street'],
  ['Central_Park',    'Main_Street'],
  ['Town_Market',     'Main_Street'],
  ['Alice_Home',      'Elm_Street'],
  ['Sofia_Home',      'Elm_Street'],
  ['Elm_Street',      'Main_Street'],
];

TOWN.DEFAULT_TOWN = {
  places: TOWN.PLACES,
  connections: TOWN.CONNECTIONS,
};
