/**
 * OpeningScene — handles the founding sequence:
 *   1. PLACE_HALL: Render iso grass, ghost Town Hall on hover, click to place.
 *   2. CHOOSE_MAYOR: HTML overlay for candidate selection.
 *   3. COUNCIL: HTML overlay for council reveal + "Begin" button.
 */

var OpeningScene = new Phaser.Class({
  Extends: Phaser.Scene,

  initialize: function OpeningScene() {
    Phaser.Scene.call(this, { key: 'OpeningScene' });
    this._ghostSprite = null;
    this._hoverGrid = { gx: -1, gy: -1 };
    this._placed = false;
  },

  create: function() {
    this.cameras.main.setBackgroundColor('#0e1117');

    // Center the iso grid in the viewport
    var centerX = this.cameras.main.width / 2;
    var centerY = 120;
    this._offsetX = centerX;
    this._offsetY = centerY;

    // Draw terrain (reuse Terrain module, offset tiles by camera origin)
    Terrain.create(this, 0.7);
    this._terrainGroup = Terrain.group;
    // Shift all terrain tiles by the scene offset
    var items = this._terrainGroup.getChildren();
    for (var ti = 0; ti < items.length; ti++) {
      items[ti].x += this._offsetX;
      items[ti].y += this._offsetY;
    }

    // Instructions text
    this._instructionText = this.add.text(centerX, 40, 'Click to place the Town Hall', {
      fontSize: '16px',
      fontFamily: 'sans-serif',
      color: '#c9d1d9',
      stroke: '#0e1117',
      strokeThickness: 3,
    });
    this._instructionText.setOrigin(0.5, 0.5);
    this._instructionText.setDepth(20000);

    // Ghost building sprite
    this._ghostSprite = this.add.image(0, 0, 'sprites', 'town_hall');
    this._ghostSprite.setOrigin(0.5, 1.0);
    this._ghostSprite.setAlpha(0.4);
    this._ghostSprite.setDepth(15000);
    this._ghostSprite.setVisible(false);
    var targetW = TILE_W * 6 * 0.6;
    this._ghostSprite.setScale(targetW / this._ghostSprite.width);

    // Input
    this._placed = false;
    this.input.on('pointermove', this._onPointerMove, this);
    this.input.on('pointerdown', this._onPointerDown, this);

    // Hide generating overlay
    document.getElementById('ui-generating').classList.add('hidden');
  },

  _onPointerMove: function(pointer) {
    if (this._placed) return;
    if (GameState.phase !== PHASES.OPENING_PLACE_HALL) return;

    var sx = pointer.x - this._offsetX;
    var sy = pointer.y - this._offsetY;
    var snap = Iso.snapToGrid(sx, sy);

    this._hoverGrid.gx = snap.gx;
    this._hoverGrid.gy = snap.gy;

    if (Iso.inBounds(snap.gx, snap.gy)) {
      this._ghostSprite.setPosition(snap.screenX + this._offsetX, snap.screenY + this._offsetY);
      this._ghostSprite.setVisible(true);
      this._ghostSprite.setTint(0x3fb950);
    } else {
      this._ghostSprite.setVisible(false);
    }
  },

  _onPointerDown: function(pointer) {
    if (this._placed) return;
    if (GameState.phase !== PHASES.OPENING_PLACE_HALL) return;

    var gx = this._hoverGrid.gx;
    var gy = this._hoverGrid.gy;
    if (!Iso.inBounds(gx, gy)) return;

    this._placed = true;
    this._ghostSprite.setAlpha(1.0);
    this._ghostSprite.clearTint();
    this._instructionText.setText('Placing Town Hall...');

    // Send to server
    fetch('/opening/place-hall', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grid_x: gx, grid_y: gy }),
    })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        // Server will send game_phase or mayor_candidates via WS
      })
      .catch(function(err) {
        console.error('Failed to place hall:', err);
        // Allow retry
        this._placed = false;
        this._instructionText.setText('Failed. Click to place the Town Hall.');
      }.bind(this));
  },

  update: function() {
    // Check if phase has advanced to mayor selection or council
    if (GameState.phase === PHASES.OPENING_CHOOSE_MAYOR) {
      this._instructionText.setText('Choose your mayor');
      this._ghostSprite.setVisible(false);
    } else if (GameState.phase === PHASES.OPENING_COUNCIL) {
      this._instructionText.setText('Council formed!');
      this._ghostSprite.setVisible(false);
    } else if (GameState.phase === PHASES.GAMEPLAY) {
      // Transition handled by TickProcessor._switchScene
    }
  },

  shutdown: function() {
    this.input.off('pointermove', this._onPointerMove, this);
    this.input.off('pointerdown', this._onPointerDown, this);
  },
});
