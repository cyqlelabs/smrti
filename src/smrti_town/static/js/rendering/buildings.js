/**
 * Building sprite management — add, remove, update from grid data.
 */

var BuildingRenderer = {
  /** @type {Object<string, Phaser.GameObjects.Image>} keyed by "gx,gy" */
  sprites: {},

  /** @type {Phaser.Scene|null} */
  scene: null,

  /** Ghost sprite for placement preview. */
  ghost: null,

  init: function(scene) {
    this.scene = scene;
    this.sprites = {};
    this.ghost = null;
  },

  /**
   * Sync building sprites with current GameState.grid.buildings.
   * Adds new buildings, removes demolished ones.
   */
  sync: function() {
    var scene = this.scene;
    if (!scene) return;

    var buildings = GameState.grid.buildings || [];
    var currentKeys = {};

    for (var i = 0; i < buildings.length; i++) {
      var b = buildings[i];
      var gx = b.grid_x;
      var gy = b.grid_y;
      var key = gx + ',' + gy;
      currentKeys[key] = true;

      if (!this.sprites[key]) {
        this._addBuilding(b);
      }
    }

    // Remove sprites for demolished buildings
    var toRemove = [];
    for (var k in this.sprites) {
      if (!currentKeys[k]) {
        toRemove.push(k);
      }
    }
    for (var r = 0; r < toRemove.length; r++) {
      this._removeBuilding(toRemove[r]);
    }
  },

  /**
   * Add a single building sprite.
   */
  /**
   * Atlas frame for a building: the server's sprite_key with the variant
   * suffix it rolled (cottage_1 + variant 2 → cottage_3), then the key
   * itself, then the local catalog's default. sprite_variant is a number,
   * so it was being used as the frame name and every varied building
   * fell back to the town hall.
   */
  _frameFor: function(spriteKey, variant, bKey) {
    var atlas = this.scene.textures.get('sprites');
    var candidates = [];
    if (spriteKey) {
      if (variant > 0) candidates.push(spriteKey.replace(/_1$/, '_' + (variant + 1)));
      candidates.push(spriteKey);
    }
    if (BUILDINGS[bKey]) candidates.push(BUILDINGS[bKey].sprite);
    candidates.push(bKey);
    for (var i = 0; i < candidates.length; i++) {
      if (candidates[i] && atlas.has(candidates[i])) return candidates[i];
    }
    return 'town_hall';
  },

  _addBuilding: function(buildingData) {
    var scene = this.scene;
    var gx = buildingData.grid_x;
    var gy = buildingData.grid_y;
    var bKey = buildingData.building_key || buildingData.type;
    var frame = this._frameFor(buildingData.sprite_key, buildingData.sprite_variant || 0, bKey);

    var pos = Iso.toScreen(gx, gy);
    var sprite = scene.add.image(pos.x, pos.y, 'sprites', frame);
    sprite.setOrigin(0.5, 1.0); // bottom-center anchor for isometric
    sprite.setDepth(Iso.depthOf(gx, gy));

    // Scale building to fit roughly in grid
    var def = BUILDINGS[bKey];
    var targetW = TILE_W * (def ? def.gridW : 3) * 0.6;
    var scale = targetW / sprite.width;
    sprite.setScale(scale);

    sprite.setInteractive({ useHandCursor: true });
    sprite.on('pointerdown', function() {
      GameState.selectedAgent = null;
      GameState.selectedPlace = buildingData;
      Sidebar.showBuilding(buildingData);
    });

    var key = gx + ',' + gy;
    this.sprites[key] = sprite;

    // Construction dust
    Particles.constructionDust(scene, pos.x, pos.y);
  },

  _removeBuilding: function(key) {
    var sprite = this.sprites[key];
    if (sprite) {
      sprite.destroy();
      delete this.sprites[key];
    }
  },

  /**
   * Show ghost building at cursor position for placement preview.
   */
  showGhost: function(gx, gy, buildingKey) {
    var scene = this.scene;
    if (!scene) return;

    var def = BUILDINGS[buildingKey];
    if (!def) return;

    var frame = this._frameFor(def.sprite, 0, buildingKey);

    var pos = Iso.toScreen(gx, gy);

    if (!this.ghost) {
      this.ghost = scene.add.image(pos.x, pos.y, 'sprites', frame);
      this.ghost.setOrigin(0.5, 1.0);
      this.ghost.setAlpha(0.5);
      this.ghost.setDepth(9999);
    } else {
      this.ghost.setPosition(pos.x, pos.y);
      this.ghost.setFrame(frame);
    }

    var targetW = TILE_W * def.gridW * 0.6;
    var scale = targetW / this.ghost.width;
    this.ghost.setScale(scale);
    this.ghost.setVisible(true);

    // Tint green if in bounds, red if out
    if (Iso.inBounds(gx, gy)) {
      this.ghost.setTint(0x3fb950);
    } else {
      this.ghost.setTint(0xf85149);
    }
  },

  hideGhost: function() {
    if (this.ghost) {
      this.ghost.setVisible(false);
    }
  },

  destroy: function() {
    for (var k in this.sprites) {
      this.sprites[k].destroy();
    }
    this.sprites = {};
    if (this.ghost) {
      this.ghost.destroy();
      this.ghost = null;
    }
    this.scene = null;
  },
};
