/**
 * Citizen sprite management — create, update positions with tweens, remove.
 */

var AgentRenderer = {
  /** @type {Object<string, {sprite: Phaser.GameObjects.Image, label: Phaser.GameObjects.Text, tween: Phaser.Tweens.Tween|null}>} */
  agents: {},

  /** @type {Phaser.Scene|null} */
  scene: null,

  init: function(scene) {
    this.scene = scene;
    this.agents = {};
  },

  /**
   * Full sync from GameState.agents array.
   * Creates new sprites, updates positions, removes dead agents.
   */
  sync: function() {
    var scene = this.scene;
    if (!scene) return;

    var agentList = GameState.agents || [];
    var activeNames = {};

    for (var i = 0; i < agentList.length; i++) {
      var a = agentList[i];
      if (!a.alive) continue;
      activeNames[a.name] = true;

      var existing = this.agents[a.name];
      if (!existing) {
        this._createAgent(a);
      } else {
        this._updateAgent(a, existing);
      }
    }

    // Remove agents no longer present
    var toRemove = [];
    for (var name in this.agents) {
      if (!activeNames[name]) {
        toRemove.push(name);
      }
    }
    for (var r = 0; r < toRemove.length; r++) {
      this._removeAgent(toRemove[r]);
    }
  },

  _createAgent: function(agentData) {
    var scene = this.scene;
    var spriteKey = GameState.spriteForAgent(agentData.name);
    var pos = this._agentScreenPos(agentData);

    var sprite = scene.add.image(pos.x, pos.y, 'sprites', spriteKey);
    sprite.setOrigin(0.5, 1.0);
    sprite.setScale(0.4);
    sprite.setDepth(pos.y + 100); // agents render above terrain/buildings at same Y
    sprite.setInteractive({ useHandCursor: true });

    sprite.on('pointerdown', function() {
      GameState.selectedPlace = null;
      GameState.selectedAgent = agentData.name;
      Sidebar.showAgent(agentData);
    });

    var label = scene.add.text(pos.x, pos.y - sprite.displayHeight - 2, agentData.name, {
      fontSize: '10px',
      fontFamily: 'sans-serif',
      color: '#c9d1d9',
      stroke: '#0e1117',
      strokeThickness: 2,
      align: 'center',
    });
    label.setOrigin(0.5, 1.0);
    label.setDepth(pos.y + 101);

    this.agents[agentData.name] = {
      sprite: sprite,
      label: label,
      tween: null,
      lastX: pos.x,
      lastY: pos.y,
    };
  },

  _updateAgent: function(agentData, entry) {
    var targetPos = this._agentScreenPos(agentData);
    var dx = Math.abs(targetPos.x - entry.lastX);
    var dy = Math.abs(targetPos.y - entry.lastY);

    // Only tween if position changed significantly
    if (dx > 2 || dy > 2) {
      if (entry.tween) {
        entry.tween.stop();
      }

      // Flip sprite based on movement direction
      if (targetPos.x < entry.lastX) {
        entry.sprite.setFlipX(true);
      } else if (targetPos.x > entry.lastX) {
        entry.sprite.setFlipX(false);
      }

      var duration = Math.max(300, Math.min(1200, Math.sqrt(dx * dx + dy * dy) * 8));

      entry.tween = this.scene.tweens.add({
        targets: [entry.sprite, entry.label],
        x: targetPos.x,
        duration: duration,
        ease: 'Sine.easeInOut',
      });

      // Separate Y tween for sprite (bottom-anchored)
      this.scene.tweens.add({
        targets: entry.sprite,
        y: targetPos.y,
        duration: duration,
        ease: 'Sine.easeInOut',
        onUpdate: function() {
          entry.sprite.setDepth(entry.sprite.y + 100);
          entry.label.setDepth(entry.sprite.y + 101);
        },
      });

      // Label Y follows sprite top
      this.scene.tweens.add({
        targets: entry.label,
        y: targetPos.y - entry.sprite.displayHeight - 2,
        duration: duration,
        ease: 'Sine.easeInOut',
      });

      entry.lastX = targetPos.x;
      entry.lastY = targetPos.y;
    }

    // Update sidebar if this agent is selected
    if (GameState.selectedAgent === agentData.name) {
      Sidebar.showAgent(agentData);
    }
  },

  _removeAgent: function(name) {
    var entry = this.agents[name];
    if (!entry) return;
    if (entry.tween) entry.tween.stop();
    entry.sprite.destroy();
    entry.label.destroy();
    delete this.agents[name];
  },

  /**
   * Compute screen position for an agent.
   * Uses world_pos if available, otherwise looks up place grid position.
   */
  _agentScreenPos: function(agentData) {
    // If backend provides world_pos (pixel coords from navgrid)
    if (agentData.world_pos) {
      var wp = agentData.world_pos;
      return Iso.worldToScreen(wp[0] || wp.x || 0, wp[1] || wp.y || 0);
    }

    // Fall back to place lookup
    var places = GameState.places || [];
    for (var i = 0; i < places.length; i++) {
      var p = places[i];
      if (p.name === agentData.location) {
        var gx = p.grid_x || 0;
        var gy = p.grid_y || 0;
        // Add small offset per agent to avoid stacking
        var hash = 0;
        for (var c = 0; c < agentData.name.length; c++) {
          hash = ((hash << 5) - hash + agentData.name.charCodeAt(c)) | 0;
        }
        var offsetX = ((hash & 0xff) / 255 - 0.5) * TILE_W * 0.6;
        var offsetY = (((hash >> 8) & 0xff) / 255 - 0.5) * TILE_H * 0.4;
        var pos = Iso.toScreen(gx, gy);
        return { x: pos.x + offsetX, y: pos.y + offsetY };
      }
    }

    // Default: center of map
    return Iso.toScreen(MAP_COLS / 2, MAP_ROWS / 2);
  },

  destroy: function() {
    for (var name in this.agents) {
      var entry = this.agents[name];
      if (entry.tween) entry.tween.stop();
      entry.sprite.destroy();
      entry.label.destroy();
    }
    this.agents = {};
    this.scene = null;
  },
};
