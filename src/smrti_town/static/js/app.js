/**
 * Main entry point — creates Phaser.Game, initializes UI modules.
 */

(function() {
  // Initialize all UI modules
  Topbar.init();
  Sidebar.init();
  EventLog.init();
  Controls.init();
  Toolbar.init();
  Petitions.init();
  Economy.init();
  Council.init();
  Settings.init();

  // Phaser game config
  var config = {
    type: Phaser.AUTO,
    parent: 'game-container',
    width: window.innerWidth,
    height: window.innerHeight,
    backgroundColor: '#0e1117',
    scale: {
      mode: Phaser.Scale.RESIZE,
      autoCenter: Phaser.Scale.CENTER_BOTH,
    },
    scene: [BootScene, OpeningScene, TownScene, GameOverScene],
    render: {
      antialias: true,
      pixelArt: false,
      roundPixels: false,
    },
    physics: {
      default: false,
    },
    audio: {
      noAudio: true,
    },
  };

  var game = new Phaser.Game(config);
  GameState.phaserGame = game;

  // Keyboard shortcuts (on the document level for UI panels)
  document.addEventListener('keydown', function(e) {
    // Don't intercept when typing in inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
      return;
    }

    switch (e.key) {
      case 'p':
      case 'P':
        if (GameState.paused) {
          fetch('/resume', { method: 'POST' });
          GameState.paused = false;
        } else {
          fetch('/pause', { method: 'POST' });
          GameState.paused = true;
        }
        Controls.updateButtons();
        break;

      case 'b':
      case 'B':
        Toolbar.el.classList.toggle('hidden');
        break;

      case 'e':
      case 'E':
        Economy.toggle();
        break;

      case 'q':
      case 'Q':
        Petitions.toggle();
        break;

      case 's':
      case 'S':
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          Settings.show();
        }
        break;

      case 'Escape':
        if (GameState.selectedBuilding) {
          GameState.selectedBuilding = null;
          BuildingRenderer.hideGhost();
          Toolbar.render();
        } else {
          Sidebar.hide();
          // Close any open overlays
          document.getElementById('ui-settings').classList.add('hidden');
          document.getElementById('ui-council-overlay').classList.add('hidden');
        }
        break;
    }
  });
})();
