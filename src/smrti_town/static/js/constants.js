/**
 * smrti-town constants — shared by all modules.
 */

var TILE_W = 64;
var TILE_H = 32;
var MAP_COLS = 20;
var MAP_ROWS = 20;
var GRID_CELL = 16;

// World pixel dimensions (must match backend config)
var WORLD_W = MAP_COLS * TILE_W;
var WORLD_H = MAP_ROWS * TILE_H + 400; // extra vertical for tall buildings

// Building catalog — mirrors backend gridmap.BUILDING_DEFS
// sprite_key maps to the atlas frame name in sprites.json
var BUILDINGS = {
  town_hall:     { name: 'Town Hall',    cost: 0,     category: 'civic',          sprite: 'town_hall', gridW: 6, gridH: 5, minPop: 0  },
  cottage:       { name: 'Cottage',      cost: 2000,  category: 'residential',    sprite: 'cottage_1', gridW: 3, gridH: 3, minPop: 0  },
  house:         { name: 'House',        cost: 4000,  category: 'residential',    sprite: 'house_1',   gridW: 4, gridH: 3, minPop: 5  },
  well:          { name: 'Well',         cost: 1500,  category: 'infrastructure', sprite: 'well',      gridW: 2, gridH: 2, minPop: 0  },
  farm:          { name: 'Farm',         cost: 5000,  category: 'industrial',     sprite: 'farm_1',    gridW: 5, gridH: 4, minPop: 0  },
  general_store: { name: 'General Store',cost: 3000,  category: 'commercial',     sprite: 'store',     gridW: 4, gridH: 3, minPop: 5  },
  bakery:        { name: 'Bakery',       cost: 4000,  category: 'commercial',     sprite: 'bakery',    gridW: 4, gridH: 3, minPop: 8  },
  park:          { name: 'Park',         cost: 2000,  category: 'cultural',       sprite: 'park',      gridW: 5, gridH: 5, minPop: 5  },
  school:        { name: 'School',       cost: 8000,  category: 'civic',          sprite: 'school',    gridW: 5, gridH: 4, minPop: 10 },
  clinic:        { name: 'Clinic',       cost: 6000,  category: 'civic',          sprite: 'clinic',    gridW: 4, gridH: 3, minPop: 10 },
  windmill:      { name: 'Windmill',     cost: 4000,  category: 'industrial',     sprite: 'windmill',  gridW: 3, gridH: 3, minPop: 10 },
  inn:           { name: 'Inn',          cost: 6000,  category: 'commercial',     sprite: 'inn',       gridW: 5, gridH: 4, minPop: 12 },
  blacksmith:    { name: 'Blacksmith',   cost: 5000,  category: 'commercial',     sprite: 'blacksmith',gridW: 4, gridH: 4, minPop: 12 },
  granary:       { name: 'Granary',      cost: 3000,  category: 'infrastructure', sprite: 'granary',   gridW: 4, gridH: 4, minPop: 10 },
  tavern:        { name: 'Tavern',       cost: 5000,  category: 'commercial',     sprite: 'tavern',    gridW: 5, gridH: 4, minPop: 10 },
  market:        { name: 'Market',       cost: 6000,  category: 'commercial',     sprite: 'market',    gridW: 5, gridH: 4, minPop: 15 },
  church:        { name: 'Church',       cost: 7000,  category: 'civic',          sprite: 'church',    gridW: 5, gridH: 5, minPop: 15 },
  warehouse:     { name: 'Warehouse',    cost: 4000,  category: 'infrastructure', sprite: 'warehouse', gridW: 5, gridH: 4, minPop: 15 },
  library:       { name: 'Library',      cost: 6000,  category: 'civic',          sprite: 'library',   gridW: 5, gridH: 4, minPop: 20 },
};

// Grass tile variants (random per cell)
var GRASS_TILES = ['grass_1', 'grass_2', 'grass_3'];

// Person sprite variants
var PERSON_SPRITES = [];
(function() {
  for (var i = 0; i <= 36; i++) {
    PERSON_SPRITES.push('person_' + (i < 10 ? '0' + i : '' + i));
  }
})();

// Category display order
var CATEGORY_ORDER = ['residential', 'commercial', 'civic', 'cultural', 'industrial', 'infrastructure'];

// Category colors for UI
var CATEGORY_COLORS = {
  residential:    '#58a6ff',
  commercial:     '#e3b341',
  civic:          '#3fb950',
  cultural:       '#bc8cff',
  industrial:     '#f0883e',
  infrastructure: '#8b949e',
};

// Time-of-day tint colors (RGBA)
var TIME_TINTS = {
  night:     { r: 30,  g: 40,  b: 80,  a: 0.45 },
  morning:   { r: 255, g: 220, b: 180, a: 0.08 },
  afternoon: { r: 0,   g: 0,   b: 0,   a: 0.0  },
  evening:   { r: 80,  g: 40,  b: 20,  a: 0.25 },
};


// Game phase FSM
var PHASES = {
  BOOT:                 'boot',
  GENERATING:           'generating',
  OPENING_PLACE_HALL:   'opening_place_hall',
  OPENING_CHOOSE_MAYOR: 'opening_choose_mayor',
  OPENING_COUNCIL:      'opening_council',
  GAMEPLAY:             'gameplay',
  GAME_OVER:            'game_over',
};
