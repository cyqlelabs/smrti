/**
 * TownScene — main gameplay Phaser scene.
 *
 * Layers:
 *   - Terrain: isometric grass tiles
 *   - Buildings: building sprites at grid positions
 *   - Agents: citizen sprites with tweened movement
 *   - UI: speech bubbles floating above agents
 *   - Day/night overlay
 *
 * Camera: drag to pan, scroll to zoom, bounded to world.
 * Building placement mode via toolbar.
 */

var TownScene = new Phaser.Class({
  Extends: Phaser.Scene,

  initialize: function TownScene() {
    Phaser.Scene.call(this, { key: 'TownScene' });
  },

  create: function() {
    var cam = this.cameras.main;
    cam.setBackgroundColor('#0e1117');

    // World bounds for the iso grid
    // The iso grid extends from roughly (-MAP_COLS * TILE_W/2) to (MAP_COLS * TILE_W/2) in X
    // and 0 to (MAP_ROWS * TILE_H) in Y, plus some padding
    var worldMinX = -MAP_COLS * TILE_W;
    var worldMinY = -200;
    var worldMaxX = MAP_COLS * TILE_W;
    var worldMaxY = MAP_ROWS * TILE_H + 600;
    cam.setBounds(worldMinX, worldMinY, worldMaxX - worldMinX, worldMaxY - worldMinY);

    // Center camera on the middle of the grid
    var center = Iso.toScreen(MAP_COLS / 2, MAP_ROWS / 2);
    cam.centerOn(center.x, center.y);

    // Zoom
    cam.setZoom(1.2);

    // ── Terrain ──
    Terrain.create(this);

    // ── Buildings ──
    BuildingRenderer.init(this);
    BuildingRenderer.sync();

    // ── Agents ──
    AgentRenderer.init(this);
    AgentRenderer.sync();

    // ── Speech bubbles ──
    SpeechBubbles.init(this);

    // ── Day/night overlay ──
    DayNight.create(this);

    // ── Camera controls: drag to pan + building placement ──
    this._dragging = false;
    this._dragStartX = 0;
    this._dragStartY = 0;

    this.input.on('pointerdown', function(pointer) {
      if (pointer.button === 0) {
        this._dragging = true;
        this._dragStartX = pointer.x;
        this._dragStartY = pointer.y;
      }
    }, this);

    this.input.on('pointermove', function(pointer) {
      // Middle mouse drag to pan
      if (pointer.isDown && pointer.button === 1) {
        cam.scrollX -= (pointer.x - pointer.prevPosition.x) / cam.zoom;
        cam.scrollY -= (pointer.y - pointer.prevPosition.y) / cam.zoom;
      }

      // Left-click drag to pan (only when not in placement mode)
      if (this._dragging && pointer.isDown && !GameState.selectedBuilding) {
        var dx = pointer.x - pointer.prevPosition.x;
        var dy = pointer.y - pointer.prevPosition.y;
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
          cam.scrollX -= dx / cam.zoom;
          cam.scrollY -= dy / cam.zoom;
        }
      }

      // Building placement ghost
      if (GameState.selectedBuilding) {
        var worldPoint = cam.getWorldPoint(pointer.x, pointer.y);
        var snap = Iso.snapToGrid(worldPoint.x, worldPoint.y);
        BuildingRenderer.showGhost(snap.gx, snap.gy, GameState.selectedBuilding);
      }
    }, this);

    this.input.on('pointerup', function(pointer) {
      if (this._dragging) {
        var dist = Phaser.Math.Distance.Between(this._dragStartX, this._dragStartY, pointer.x, pointer.y);

        // If it was a click (not a drag) and in placement mode, place building
        if (dist < 5 && GameState.selectedBuilding) {
          var worldPoint = cam.getWorldPoint(pointer.x, pointer.y);
          var snap = Iso.snapToGrid(worldPoint.x, worldPoint.y);
          if (Iso.inBounds(snap.gx, snap.gy)) {
            this._placeBuilding(snap.gx, snap.gy, GameState.selectedBuilding);
          }
        }

        this._dragging = false;
      }
    }, this);

    // Scroll to zoom
    this.input.on('wheel', function(pointer, gameObjects, deltaX, deltaY) {
      var newZoom = cam.zoom - deltaY * 0.001;
      cam.setZoom(Phaser.Math.Clamp(newZoom, 0.4, 3.0));
    }, this);

    // ESC to cancel placement
    this.input.keyboard.on('keydown-ESC', function() {
      if (GameState.selectedBuilding) {
        GameState.selectedBuilding = null;
        BuildingRenderer.hideGhost();
        Toolbar.render();
      } else {
        Sidebar.hide();
      }
    }, this);

    // Show gameplay UI
    Topbar.show();
    Controls.show();
    Toolbar.show();
    EventLog.show();
    Topbar.update();
    Controls.updateButtons();

    // Hide opening overlays
    document.getElementById('ui-generating').classList.add('hidden');
    document.getElementById('ui-mayor-select').classList.add('hidden');
    document.getElementById('ui-council-reveal').classList.add('hidden');
  },

  _placeBuilding: function(gx, gy, buildingKey) {
    fetch('/place-building', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ building_key: buildingKey, grid_x: gx, grid_y: gy }),
    })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.error) {
          EventLog.add('Cannot place: ' + data.error, 'crisis');
        } else {
          // Deselect after placing
          GameState.selectedBuilding = null;
          BuildingRenderer.hideGhost();
          Toolbar.render();
        }
      })
      .catch(function(err) {
        EventLog.add('Placement failed: ' + err, 'crisis');
      });
  },

  update: function(time, delta) {
    // Update day/night tint
    DayNight.update(this);

    // Update speech bubble positions
    SpeechBubbles.update();

    // Depth sort agents (update depth based on current Y)
    for (var name in AgentRenderer.agents) {
      var entry = AgentRenderer.agents[name];
      if (entry.sprite) {
        entry.sprite.setDepth(entry.sprite.y + 100);
        entry.label.setDepth(entry.sprite.y + 101);
      }
    }
  },

  shutdown: function() {
    // Clean up renderers
    Terrain.destroy();
    BuildingRenderer.destroy();
    AgentRenderer.destroy();
    SpeechBubbles.destroy();
    DayNight.destroy();

    // Hide UI
    Topbar.hide();
    Controls.hide();
    Toolbar.hide();
    EventLog.hide();
    Sidebar.hide();
  },
});
