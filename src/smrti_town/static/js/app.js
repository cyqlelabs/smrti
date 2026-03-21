/* ================================================================
   app.js — Entry point: Phaser game, WebSocket, demo mode boot
   ================================================================ */
(function() {
  'use strict';

  /* ── Phaser Configuration ──────────────────────────────────────── */
  var config = {
    type: Phaser.AUTO,
    parent: 'game-container',
    width: window.innerWidth,
    height: window.innerHeight,
    backgroundColor: '#F5E6D0',
    scene: [TOWN.TownScene],
    scale: {
      mode: Phaser.Scale.RESIZE,
      autoCenter: Phaser.Scale.CENTER_BOTH,
    },
  };

  /* ── Create Game ───────────────────────────────────────────────── */
  var game = new Phaser.Game(config);

  /* ── Wire up UI controls ───────────────────────────────────────── */
  TOWN.initControls();

  /* ── Resize handler ────────────────────────────────────────────── */
  window.addEventListener('resize', TOWN._handleResize);

  /* ── Connect WebSocket ─────────────────────────────────────────── */
  TOWN.connectWS();

  /* ── Start demo mode (falls back if no server) ─────────────────── */
  TOWN.startDemoMode();

})();
