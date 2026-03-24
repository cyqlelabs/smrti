/**
 * Day/night tint overlay based on calendar time_of_day.
 */

var DayNight = {
  /** @type {Phaser.GameObjects.Rectangle|null} */
  overlay: null,

  /** Current time of day key. */
  _current: 'afternoon',

  /**
   * Create the overlay rectangle covering the whole world.
   * @param {Phaser.Scene} scene
   */
  create: function(scene) {
    // Use camera viewport size; will be repositioned in update
    this.overlay = scene.add.rectangle(0, 0, WORLD_W * 2, WORLD_H * 2, 0x000000, 0);
    this.overlay.setOrigin(0.5, 0.5);
    this.overlay.setDepth(10000);
    this.overlay.setScrollFactor(0); // fixed to camera
    this.overlay.setBlendMode(Phaser.BlendModes.MULTIPLY);
    this._current = 'afternoon';
  },

  /**
   * Update tint based on current calendar.
   */
  update: function(scene) {
    if (!this.overlay) return;

    var tod = GameState.calendar.time_of_day || 'afternoon';
    if (tod === this._current) return;
    this._current = tod;

    var tint = TIME_TINTS[tod] || TIME_TINTS.afternoon;

    if (tint.a <= 0.01) {
      this.overlay.setAlpha(0);
      return;
    }

    var color = Phaser.Display.Color.GetColor(
      255 - Math.round(tint.r * tint.a),
      255 - Math.round(tint.g * tint.a),
      255 - Math.round(tint.b * tint.a)
    );

    this.overlay.setFillStyle(color, tint.a);
    this.overlay.setAlpha(1);

    // Resize to camera viewport
    var cam = scene.cameras.main;
    this.overlay.setPosition(cam.width / 2, cam.height / 2);
    this.overlay.setSize(cam.width + 100, cam.height + 100);
  },

  destroy: function() {
    if (this.overlay) {
      this.overlay.destroy();
      this.overlay = null;
    }
  },
};
