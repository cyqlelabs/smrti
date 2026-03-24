/**
 * BootScene — loads assets and shows a loading bar.
 */

var BootScene = new Phaser.Class({
  Extends: Phaser.Scene,

  initialize: function BootScene() {
    Phaser.Scene.call(this, { key: 'BootScene' });
  },

  preload: function() {
    var self = this;
    var w = this.cameras.main.width;
    var h = this.cameras.main.height;

    // Background
    this.cameras.main.setBackgroundColor('#0e1117');

    // Loading bar
    var barW = 300;
    var barH = 12;
    var barX = (w - barW) / 2;
    var barY = h / 2;

    var bgBar = this.add.rectangle(w / 2, barY, barW, barH, 0x1e2433);
    bgBar.setOrigin(0.5, 0.5);

    var fillBar = this.add.rectangle(barX, barY, 0, barH, 0x58a6ff);
    fillBar.setOrigin(0, 0.5);

    var loadingText = this.add.text(w / 2, barY - 24, 'Loading...', {
      fontSize: '14px',
      fontFamily: 'sans-serif',
      color: '#8b949e',
    });
    loadingText.setOrigin(0.5, 0.5);

    this.load.on('progress', function(value) {
      fillBar.width = barW * value;
    });

    this.load.on('complete', function() {
      loadingText.setText('Ready');
    });

    // Load sprite atlas
    this.load.atlas('sprites', 'smrti-sprite.png', 'sprites.json');
  },

  create: function() {
    // Transition: show the generating overlay and connect WS
    // The WS connection will trigger the appropriate scene based on server state
    var genEl = document.getElementById('ui-generating');
    document.getElementById('generating-message').textContent = 'Connecting to server...';
    document.getElementById('generating-hint').textContent = '';
    genEl.classList.remove('hidden');

    WS.connect();

    // Wait for first state message to determine which scene to go to.
    // The TickProcessor._handleState will call _switchScene.
    // If no response in 2s, show connecting message
    var self = this;
    this.time.delayedCall(2000, function() {
      if (GameState.phase === PHASES.BOOT) {
        document.getElementById('generating-message').textContent = 'Waiting for server...';
        document.getElementById('generating-hint').textContent = 'Make sure smrti serve town is running on port 8430';
      }
    });
  },
});
