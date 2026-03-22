/* ================================================================
   TownScene.js — Main Phaser scene: create(), update(), layers
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.TownScene = new Phaser.Class({
  Extends: Phaser.Scene,

  initialize: function TownScene() {
    Phaser.Scene.call(this, { key: 'TownScene' });
  },

  preload: function() {
    /* No assets to preload — all drawn procedurally */
  },

  create: function() {
    TOWN.state.scene = this;
    this.cameras.main.setScroll(0, 0);

    /* ── Layer hierarchy ───────────────────────────────────────── */

    /* Background graphics */
    this.bgLayer = this.add.graphics().setDepth(0);

    /* Night/weather overlay */
    this.nightOverlay = this.add.rectangle(
      this.scale.width / 2, this.scale.height / 2,
      this.scale.width * 2, this.scale.height * 2,
      0x000020, 0
    ).setDepth(5);

    /* Road graphics */
    this.roadLayer = this.add.graphics().setDepth(1);

    /* Building container */
    this.buildingLayer = this.add.container(0, 0).setDepth(2);

    /* Relationship overlay (below agents) */
    this.relOverlayLayer = this.add.graphics().setDepth(9);
    TOWN._relOverlayGfx = this.relOverlayLayer;
    TOWN.state.showRelOverlay = false;

    /* Agent container */
    this.agentLayer = this.add.container(0, 0).setDepth(10);

    /* Speech bubble container */
    this.speechLayer = this.add.container(0, 0).setDepth(20);

    /* Particle container */
    this.particleLayer = this.add.container(0, 0).setDepth(25);

    /* ── Draw initial town ─────────────────────────────────────── */
    TOWN.state.town = TOWN.DEFAULT_TOWN;
    TOWN.drawTown(this, TOWN.DEFAULT_TOWN);

    /* ── Input handling ────────────────────────────────────────── */
    this.input.on('pointerdown', TOWN._handleClick, this);
    this.input.on('pointermove', TOWN._handleHover, this);
  },

  update: function(time, delta) {
    /* Process tick queue */
    if (TOWN.state.tickQueue.length > 0 && !TOWN.state.processing) {
      TOWN.state.processing = true;
      var tick = TOWN.state.tickQueue.shift();
      TOWN.processTick(this, tick).then(function() {
        TOWN.state.processing = false;
      });
    }

    /* Ambient day/night cycle */
    TOWN.updateDayNight(this);

    /* Relationship overlay */
    TOWN.drawRelationshipOverlay(this);
  },
});

/* ── Click Handler ───────────────────────────────────────────────── */
TOWN._handleClick = function(pointer) {
  var scene = TOWN.state.scene;
  if (!scene) return;

  var worldPoint = scene.cameras.main.getWorldPoint(pointer.x, pointer.y);

  /* Check agents first (top layer) */
  var clickedAgent = null;
  var sprites = TOWN.state.agentSprites;
  for (var name in sprites) {
    var sp = sprites[name];
    var dist = Phaser.Math.Distance.Between(worldPoint.x, worldPoint.y, sp.x, sp.y);
    if (dist < sp.radius + 10) {
      clickedAgent = name;
      break;
    }
  }

  if (clickedAgent) {
    TOWN.selectAgent(clickedAgent);
    return;
  }

  /* Check places */
  var clickedPlace = null;
  var placeSprites = TOWN.state.placeSprites;
  for (var pName in placeSprites) {
    var ps = placeSprites[pName];
    if (worldPoint.x >= ps.x && worldPoint.x <= ps.x + ps.w &&
        worldPoint.y >= ps.y && worldPoint.y <= ps.y + ps.h) {
      clickedPlace = pName;
      break;
    }
  }

  if (clickedPlace) {
    TOWN.state.selectedPlace = clickedPlace;
    TOWN.state.selectedAgent = null;
    /* Clear any agent selection rings */
    for (var an in sprites) {
      if (sprites[an].selRing.alpha > 0) {
        scene.tweens.killTweensOf(sprites[an].selRing);
        sprites[an].selRing.setAlpha(0);
      }
    }
    TOWN.openSidebar();
    TOWN.renderPlaceSidebar(clickedPlace);
    return;
  }

  /* Empty space — deselect */
  TOWN.state.selectedAgent = null;
  TOWN.state.selectedPlace = null;
  for (var sn in sprites) {
    if (sprites[sn].selRing.alpha > 0) {
      scene.tweens.killTweensOf(sprites[sn].selRing);
      sprites[sn].selRing.setAlpha(0);
    }
  }
};

/* ── Hover Handler (tooltip) ─────────────────────────────────────── */
TOWN._handleHover = function(pointer) {
  var scene = TOWN.state.scene;
  if (!scene) return;

  var tooltip = document.getElementById('tooltip');
  if (!tooltip) return;

  var worldPoint = scene.cameras.main.getWorldPoint(pointer.x, pointer.y);

  /* Check if hovering over an agent */
  var hovered = null;
  var sprites = TOWN.state.agentSprites;
  for (var name in sprites) {
    var sp = sprites[name];
    var dist = Phaser.Math.Distance.Between(worldPoint.x, worldPoint.y, sp.x, sp.y);
    if (dist < sp.radius + 8) {
      hovered = name;
      break;
    }
  }

  if (hovered) {
    tooltip.textContent = hovered.replace(/_/g, ' ');
    tooltip.style.left = (pointer.x + 16) + 'px';
    tooltip.style.top = (pointer.y - 8) + 'px';
    tooltip.classList.add('visible');
  } else {
    tooltip.classList.remove('visible');
  }
};

/* ── Resize Handler ──────────────────────────────────────────────── */
TOWN._handleResize = function() {
  var scene = TOWN.state.scene;
  if (scene && scene.nightOverlay) {
    scene.nightOverlay.setPosition(window.innerWidth / 2, window.innerHeight / 2);
    scene.nightOverlay.setSize(window.innerWidth * 2, window.innerHeight * 2);
  }
};
