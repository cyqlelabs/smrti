/**
 * Particle effects — construction dust and celebration sparkles.
 * Uses simple Phaser graphics objects since Phaser 3.60+ removed
 * the particle manager in favour of ParticleEmitter on GameObjects.
 */

var Particles = {
  /**
   * Emit construction dust at a screen position.
   * @param {Phaser.Scene} scene
   * @param {number} x
   * @param {number} y
   */
  constructionDust: function(scene, x, y) {
    var count = 8;
    for (var i = 0; i < count; i++) {
      (function() {
        var dot = scene.add.circle(
          x + (Math.random() - 0.5) * 30,
          y + (Math.random() - 0.5) * 20,
          2 + Math.random() * 3,
          0xb8a080,
          0.7
        );
        dot.setDepth(10001);

        scene.tweens.add({
          targets: dot,
          x: dot.x + (Math.random() - 0.5) * 40,
          y: dot.y - 10 - Math.random() * 30,
          alpha: 0,
          scale: 0.3,
          duration: 600 + Math.random() * 400,
          ease: 'Quad.easeOut',
          onComplete: function() {
            dot.destroy();
          },
        });
      })();
    }
  },

  /**
   * Emit celebration sparkles at a screen position.
   * @param {Phaser.Scene} scene
   * @param {number} x
   * @param {number} y
   */
  celebrate: function(scene, x, y) {
    var colors = [0xffe066, 0xff6b6b, 0x58a6ff, 0x3fb950, 0xbc8cff];
    var count = 12;
    for (var i = 0; i < count; i++) {
      (function(idx) {
        var color = colors[idx % colors.length];
        var dot = scene.add.circle(
          x,
          y,
          2 + Math.random() * 2,
          color,
          0.9
        );
        dot.setDepth(10001);

        var angle = (Math.PI * 2 * idx) / count + (Math.random() - 0.5) * 0.5;
        var dist = 20 + Math.random() * 40;

        scene.tweens.add({
          targets: dot,
          x: x + Math.cos(angle) * dist,
          y: y + Math.sin(angle) * dist - 20,
          alpha: 0,
          scale: 0.2,
          duration: 800 + Math.random() * 400,
          ease: 'Quad.easeOut',
          onComplete: function() {
            dot.destroy();
          },
        });
      })(i);
    }
  },
};
