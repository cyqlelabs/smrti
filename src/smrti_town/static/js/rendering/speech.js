/**
 * Speech bubble management — floating text above agents.
 */

var SpeechBubbles = {
  /** @type {Object<string, {bg: Phaser.GameObjects.Graphics, text: Phaser.GameObjects.Text, timer: Phaser.Time.TimerEvent}>} */
  bubbles: {},

  /** @type {Phaser.Scene|null} */
  scene: null,

  init: function(scene) {
    this.scene = scene;
    this.bubbles = {};
  },

  /**
   * Show a speech bubble above an agent.
   * @param {string} speakerName
   * @param {string} dialogue - Text to show (truncated to 80 chars)
   * @param {number} [duration=4000] - How long to show (ms)
   */
  show: function(speakerName, dialogue, duration) {
    var scene = this.scene;
    if (!scene) return;

    // Remove existing bubble for this speaker
    this.remove(speakerName);

    var agentEntry = AgentRenderer.agents[speakerName];
    if (!agentEntry) return;

    var sx = agentEntry.sprite.x;
    var sy = agentEntry.sprite.y - agentEntry.sprite.displayHeight - 14;

    // Truncate long text
    var displayText = dialogue;
    if (displayText.length > 80) {
      displayText = displayText.substring(0, 77) + '...';
    }

    var text = scene.add.text(sx, sy, displayText, {
      fontSize: '10px',
      fontFamily: 'sans-serif',
      color: '#e6edf3',
      backgroundColor: '#161b26ee',
      padding: { x: 6, y: 4 },
      wordWrap: { width: 140 },
      align: 'center',
      stroke: '#0e1117',
      strokeThickness: 1,
    });
    text.setOrigin(0.5, 1.0);
    text.setDepth(10002);

    // Fade in
    text.setAlpha(0);
    scene.tweens.add({
      targets: text,
      alpha: 1,
      duration: 200,
      ease: 'Quad.easeOut',
    });

    // Auto-remove after duration
    var dur = duration || 4000;
    var timer = scene.time.delayedCall(dur, function() {
      this._fadeAndRemove(speakerName);
    }, [], this);

    this.bubbles[speakerName] = { text: text, timer: timer };
  },

  _fadeAndRemove: function(speakerName) {
    var entry = this.bubbles[speakerName];
    if (!entry || !this.scene) return;

    this.scene.tweens.add({
      targets: entry.text,
      alpha: 0,
      duration: 300,
      ease: 'Quad.easeIn',
      onComplete: function() {
        if (entry.text) entry.text.destroy();
        delete this.bubbles[speakerName];
      }.bind(this),
    });
  },

  /**
   * Remove a speech bubble immediately.
   */
  remove: function(speakerName) {
    var entry = this.bubbles[speakerName];
    if (!entry) return;
    if (entry.timer) entry.timer.remove();
    if (entry.text) entry.text.destroy();
    delete this.bubbles[speakerName];
  },

  /**
   * Update bubble positions to follow agents.
   */
  update: function() {
    for (var name in this.bubbles) {
      var entry = this.bubbles[name];
      var agentEntry = AgentRenderer.agents[name];
      if (agentEntry && entry.text) {
        entry.text.setPosition(
          agentEntry.sprite.x,
          agentEntry.sprite.y - agentEntry.sprite.displayHeight - 14
        );
      }
    }
  },

  destroy: function() {
    for (var name in this.bubbles) {
      this.remove(name);
    }
    this.bubbles = {};
    this.scene = null;
  },
};
