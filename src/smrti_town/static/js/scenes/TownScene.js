/* ================================================================
   TownScene.js — Main Phaser scene with camera pan/zoom
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.TownScene = new Phaser.Class({
  Extends: Phaser.Scene,

  initialize: function TownScene() {
    Phaser.Scene.call(this, { key: 'TownScene' });
  },

  preload: function() {},

  create: function() {
    TOWN.state.scene = this;

    /* ── Camera ────────────────────────────────────────────────── */
    this.cameras.main.setScroll(0, 0);
    this.cameras.main.setZoom(1.0);
    this.cameras.main.setBounds(-900, -400, 3400, 2200);

    /* ── Layers ────────────────────────────────────────────────── */
    this.bgLayer       = this.add.graphics().setDepth(0);
    this.roadLayer     = this.add.graphics().setDepth(1);
    this.buildingLayer = this.add.container(0, 0).setDepth(2);

    /* Night overlay — scrollFactor(0) keeps it fixed to the screen */
    this.nightOverlay = this.add.rectangle(
      this.scale.width / 2, this.scale.height / 2,
      this.scale.width * 4, this.scale.height * 4,
      0x000020, 0
    ).setDepth(5).setScrollFactor(0);

    this.relOverlayLayer = this.add.graphics().setDepth(9);
    TOWN._relOverlayGfx  = this.relOverlayLayer;
    TOWN.state.showRelOverlay = false;

    this.agentLayer    = this.add.container(0, 0).setDepth(10);
    this.speechLayer   = this.add.container(0, 0).setDepth(20);
    this.particleLayer = this.add.container(0, 0).setDepth(25);

    /* Hint text fixed to screen */
    this.add.text(12, this.scale.height - 28, 'Drag to pan  •  Scroll to zoom', {
      fontSize: '11px', fontFamily: 'Fredoka, sans-serif', color: '#8B7355',
    }).setScrollFactor(0).setDepth(30).setAlpha(0.6);

    /* ── Draw initial town ─────────────────────────────────────── */
    TOWN.state.town = TOWN.DEFAULT_TOWN;
    TOWN.drawTown(this, TOWN.DEFAULT_TOWN);

    /* ── Input ─────────────────────────────────────────────────── */
    this._dragStart    = null;
    this._dragCamStart = null;
    this._isDragging   = false;
    var DRAG_THRESH    = 7;
    var scene = this;

    this.input.on('pointerdown', function(ptr) {
      scene._dragStart    = { x: ptr.x, y: ptr.y };
      scene._dragCamStart = { x: scene.cameras.main.scrollX, y: scene.cameras.main.scrollY };
      scene._isDragging   = false;
    });

    this.input.on('pointermove', function(ptr) {
      if (!scene._dragStart) return;
      var dx = ptr.x - scene._dragStart.x;
      var dy = ptr.y - scene._dragStart.y;
      if (!scene._isDragging && Math.sqrt(dx * dx + dy * dy) > DRAG_THRESH) {
        scene._isDragging = true;
      }
      if (scene._isDragging) {
        var z = scene.cameras.main.zoom;
        scene.cameras.main.scrollX = scene._dragCamStart.x - dx / z;
        scene.cameras.main.scrollY = scene._dragCamStart.y - dy / z;
      } else {
        TOWN._handleHover(ptr);
      }
    });

    this.input.on('pointerup', function(ptr) {
      if (!scene._isDragging) TOWN._handleClick(ptr);
      scene._dragStart  = null;
      scene._isDragging = false;
    });

    /* Mouse wheel zoom */
    this.input.on('wheel', function(ptr, objs, dx, dy) {
      var cam = scene.cameras.main;
      cam.zoom = Phaser.Math.Clamp(cam.zoom - dy * 0.0009, 0.35, 2.5);
    });
  },

  update: function(time, delta) {
    /* Dequeue one tick per frame */
    if (TOWN.state.tickQueue.length > 0 && !TOWN.state.processing) {
      TOWN.state.processing = true;
      var tick = TOWN.state.tickQueue.shift();
      TOWN.processTick(this, tick).then(function() {
        TOWN.state.processing = false;
      });
    }
    TOWN.updateDayNight(this);
    TOWN.drawRelationshipOverlay(this);
  },
});

/* ── Click handler ───────────────────────────────────────────────── */
TOWN._handleClick = function(pointer) {
  var scene = TOWN.state.scene;
  if (!scene) return;
  var wp = scene.cameras.main.getWorldPoint(pointer.x, pointer.y);

  /* Agents */
  var sprites = TOWN.state.agentSprites;
  for (var name in sprites) {
    var sp = sprites[name];
    var headOffY = sp.radius * 2 + 14;
    var dist = Phaser.Math.Distance.Between(wp.x, wp.y, sp.x, sp.y - headOffY);
    if (dist < sp.radius + 12) { TOWN.selectAgent(name); return; }
  }

  /* Places */
  var placeSprites = TOWN.state.placeSprites;
  for (var pName in placeSprites) {
    var ps = placeSprites[pName];
    var hc = ps.hitzone;
    var hdx = wp.x - hc.x, hdy = wp.y - hc.y;
    if (Math.abs(hdx) < (hc.width || 80) / 2 && Math.abs(hdy) < (hc.height || 40) / 2) {
      TOWN.state.selectedPlace = pName;
      TOWN.state.selectedAgent = null;
      for (var an in sprites) {
        if (sprites[an].selRing.alpha > 0) {
          scene.tweens.killTweensOf(sprites[an].selRing);
          sprites[an].selRing.setAlpha(0);
        }
      }
      TOWN.openSidebar();
      TOWN.renderPlaceSidebar(pName);
      return;
    }
  }

  /* Deselect */
  TOWN.state.selectedAgent = null;
  TOWN.state.selectedPlace = null;
  for (var sn in sprites) {
    if (sprites[sn].selRing.alpha > 0) {
      scene.tweens.killTweensOf(sprites[sn].selRing);
      sprites[sn].selRing.setAlpha(0);
    }
  }
};

/* ── Hover handler ───────────────────────────────────────────────── */
TOWN._handleHover = function(pointer) {
  var scene = TOWN.state.scene;
  if (!scene) return;
  var tooltip = document.getElementById('tooltip');
  if (!tooltip) return;
  var wp = scene.cameras.main.getWorldPoint(pointer.x, pointer.y);
  var hovered = null;
  var sprites = TOWN.state.agentSprites;
  for (var name in sprites) {
    var sp = sprites[name];
    var headOffY = sp.radius * 2 + 14;
    var dist = Phaser.Math.Distance.Between(wp.x, wp.y, sp.x, sp.y - headOffY);
    if (dist < sp.radius + 12) { hovered = name; break; }
  }
  if (hovered) {
    tooltip.textContent = hovered.replace(/_/g, ' ');
    tooltip.style.left = (pointer.x + 16) + 'px';
    tooltip.style.top  = (pointer.y - 8) + 'px';
    tooltip.classList.add('visible');
  } else {
    tooltip.classList.remove('visible');
  }
};

/* ── Resize handler ──────────────────────────────────────────────── */
TOWN._handleResize = function() {
  var scene = TOWN.state.scene;
  if (scene && scene.nightOverlay) {
    scene.nightOverlay.setPosition(window.innerWidth / 2, window.innerHeight / 2);
    scene.nightOverlay.setSize(window.innerWidth * 4, window.innerHeight * 4);
  }
};
