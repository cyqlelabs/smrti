/**
 * GameOverScene — shows failure reason and a restart button.
 */

var GameOverScene = new Phaser.Class({
  Extends: Phaser.Scene,

  initialize: function GameOverScene() {
    Phaser.Scene.call(this, { key: 'GameOverScene' });
  },

  create: function() {
    var w = this.cameras.main.width;
    var h = this.cameras.main.height;

    this.cameras.main.setBackgroundColor('#0e1117');

    // Hide gameplay UI
    Topbar.hide();
    Controls.hide();
    Toolbar.hide();
    EventLog.hide();
    Sidebar.hide();

    // Title
    this.add.text(w / 2, h / 2 - 60, 'GAME OVER', {
      fontSize: '32px',
      fontFamily: 'sans-serif',
      color: '#f85149',
      fontStyle: 'bold',
    }).setOrigin(0.5, 0.5);

    // Reason
    var reason = GameState.gameOverReason || 'The town has fallen.';
    this.add.text(w / 2, h / 2, reason, {
      fontSize: '16px',
      fontFamily: 'sans-serif',
      color: '#8b949e',
      wordWrap: { width: 400 },
      align: 'center',
    }).setOrigin(0.5, 0.5);

    // Restart button
    var btn = this.add.text(w / 2, h / 2 + 60, '[ Restart ]', {
      fontSize: '18px',
      fontFamily: 'sans-serif',
      color: '#58a6ff',
      fontStyle: 'bold',
    }).setOrigin(0.5, 0.5);

    btn.setInteractive({ useHandCursor: true });

    btn.on('pointerover', function() {
      btn.setColor('#79b8ff');
    });

    btn.on('pointerout', function() {
      btn.setColor('#58a6ff');
    });

    btn.on('pointerdown', function() {
      // Request world regeneration
      fetch('/regenerate', { method: 'POST' })
        .then(function() {
          GameState.phase = PHASES.GENERATING;
          document.getElementById('generating-message').textContent = 'Regenerating world...';
          document.getElementById('generating-hint').textContent = '';
          document.getElementById('ui-generating').classList.remove('hidden');
        });
    });
  },
});
