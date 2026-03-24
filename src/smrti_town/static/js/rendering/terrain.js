/**
 * Isometric terrain tile grid rendering.
 */

var Terrain = {
  /** @type {Phaser.GameObjects.Group|null} */
  group: null,

  /** Seeded random for consistent terrain look. */
  _tileMap: null,

  /**
   * Create the isometric grass terrain.
   * @param {Phaser.Scene} scene
   */
  create: function(scene, alpha) {
    if (this.group) this.group.destroy(true);
    this.group = scene.add.group();
    this._tileMap = [];
    var a = (alpha !== undefined) ? alpha : 0.85;

    for (var gy = 0; gy < MAP_ROWS; gy++) {
      this._tileMap[gy] = [];
      for (var gx = 0; gx < MAP_COLS; gx++) {
        var tileKey = GRASS_TILES[(gx * 7 + gy * 13) % GRASS_TILES.length];
        var pos = Iso.toScreen(gx, gy);
        var tile = scene.add.image(pos.x, pos.y, 'sprites', tileKey);
        tile.setOrigin(0.5, 0.5);
        tile.setDisplaySize(TILE_W, TILE_H);
        tile.setDepth(Iso.depthOf(gx, gy) - 1);
        tile.setAlpha(a);
        this.group.add(tile);
        this._tileMap[gy][gx] = tile;
      }
    }
  },

  /**
   * Highlight a tile at grid position.
   * @param {Phaser.Scene} scene
   * @param {number} gx
   * @param {number} gy
   * @param {number} [color=0x58a6ff]
   */
  highlightTile: function(scene, gx, gy, color) {
    if (!Iso.inBounds(gx, gy)) return;
    var tile = this._tileMap[gy][gx];
    if (tile) {
      tile.setTint(color || 0x58a6ff);
    }
  },

  /**
   * Clear all tile highlights.
   */
  clearHighlights: function() {
    if (!this._tileMap) return;
    for (var gy = 0; gy < MAP_ROWS; gy++) {
      for (var gx = 0; gx < MAP_COLS; gx++) {
        var tile = this._tileMap[gy] && this._tileMap[gy][gx];
        if (tile) tile.clearTint();
      }
    }
  },

  /**
   * Destroy all terrain sprites.
   */
  destroy: function() {
    if (this.group) {
      this.group.destroy(true);
      this.group = null;
    }
    this._tileMap = null;
  },
};
