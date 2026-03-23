/**
 * Isometric coordinate utilities.
 * Standard 2:1 isometric projection.
 */

var Iso = {
  /**
   * Convert grid coordinates to screen pixel position.
   * @param {number} gx - Grid column
   * @param {number} gy - Grid row
   * @returns {{x: number, y: number}} Screen position
   */
  toScreen: function(gx, gy) {
    return {
      x: (gx - gy) * (TILE_W / 2),
      y: (gx + gy) * (TILE_H / 2),
    };
  },

  /**
   * Convert screen pixel position to grid coordinates.
   * @param {number} sx - Screen X
   * @param {number} sy - Screen Y
   * @returns {{gx: number, gy: number}} Grid position (fractional)
   */
  toGrid: function(sx, sy) {
    var halfW = TILE_W / 2;
    var halfH = TILE_H / 2;
    return {
      gx: (sx / halfW + sy / halfH) / 2,
      gy: (sy / halfH - sx / halfW) / 2,
    };
  },

  /**
   * Snap a screen position to the nearest grid cell.
   * @param {number} sx - Screen X
   * @param {number} sy - Screen Y
   * @returns {{gx: number, gy: number, screenX: number, screenY: number}}
   */
  snapToGrid: function(sx, sy) {
    var g = this.toGrid(sx, sy);
    var gx = Math.round(g.gx);
    var gy = Math.round(g.gy);
    var s = this.toScreen(gx, gy);
    return { gx: gx, gy: gy, screenX: s.x, screenY: s.y };
  },

  /**
   * Check if grid coordinates are within the map bounds.
   */
  inBounds: function(gx, gy) {
    return gx >= 0 && gx < MAP_COLS && gy >= 0 && gy < MAP_ROWS;
  },

  /**
   * Depth sort value for a given grid position.
   * Higher values render on top.
   */
  depthOf: function(gx, gy) {
    return (gx + gy) * TILE_H / 2;
  },

  /**
   * Convert backend world_pos (pixel coords from navgrid) to iso screen coords.
   * Backend uses a flat coordinate system where positions are in CELL_SIZE units.
   * We map these into our isometric grid.
   */
  worldToScreen: function(worldX, worldY) {
    var gx = worldX / GRID_CELL;
    var gy = worldY / GRID_CELL;
    return this.toScreen(gx, gy);
  },
};
