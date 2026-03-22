/* ================================================================
   particles.js — spawnParticles() for births, deaths, romance, weather
   ================================================================ */
window.TOWN = window.TOWN || {};

/* ── Particle pool helpers ─────────────────────────────────────────── */
TOWN._getParticle = function(scene) {
  var pool = TOWN.state.particlePool;
  for (var i = 0; i < pool.length; i++) {
    if (!pool[i].inUse) {
      pool[i].inUse = true;
      pool[i].txt.setVisible(true);
      return pool[i];
    }
  }
  var entry = {
    txt: scene.add.text(0, 0, '', { fontSize: '20px' })
           .setOrigin(0.5).setDepth(26).setAlpha(0).setVisible(false),
    inUse: true,
  };
  scene.particleLayer.add(entry.txt);
  pool.push(entry);
  return entry;
};

TOWN._returnParticle = function(entry) {
  entry.txt.setVisible(false).setAlpha(0).setScale(1).setText('');
  entry.inUse = false;
};

/* ── spawnParticles ─────────────────────────────────────────────────── */
TOWN.spawnParticles = function(scene, x, y, type) {
  var configs = {
    birth: {
      emojis: ['\u2B50', '\u2728', '\u2606'],
      colors: ['#FF6F91', '#FFD93D', '#FF9671'],
      count: 10, speed: 50, life: 2200, gravity: -30,
    },
    death: {
      emojis: ['\u00B7', '\u2727', '\u2022'],
      colors: ['#8B8B8B', '#A0A0A0', '#707070'],
      count: 8, speed: 20, life: 3000, gravity: 15,
    },
    romance: {
      emojis: ['\u2764\uFE0F', '\uD83D\uDC95', '\u2665'],
      colors: ['#FF6B6B', '#FF6F91', '#FF9671'],
      count: 8, speed: 35, life: 2000, gravity: -40,
    },
    rain: {
      emojis: ['\u2502', '\u2503'],
      colors: ['#6BA3BE', '#4D96FF'],
      count: 15, speed: 60, life: 1000, gravity: 80,
    },
    sunshine: {
      emojis: ['\u2726', '\u2605', '\u2736'],
      colors: ['#FFD93D', '#FF9671', '#FFFFFF'],
      count: 8, speed: 25, life: 1800, gravity: -10,
    },
  };

  var cfg = configs[type] || configs.birth;

  for (var i = 0; i < cfg.count; i++) {
    (function(idx) {
      var entry = TOWN._getParticle(scene);
      var txt   = entry.txt;

      var emoji = cfg.emojis[idx % cfg.emojis.length];
      var color = cfg.colors[idx % cfg.colors.length];
      var sizeBase = type === 'rain' ? 14 : 20;
      var sizePx   = sizeBase + Math.floor(Math.random() * 8 - 3);

      txt.setText(emoji).setStyle({ fontSize: sizePx + 'px', color: color });

      var angle  = (idx / cfg.count) * Math.PI * 2 + (Math.random() - 0.5) * 0.6;
      var spawnR = 4 + Math.random() * 8;
      var px = x + Math.cos(angle) * spawnR;
      var py = y + Math.sin(angle) * spawnR;
      txt.setPosition(px, py).setScale(0).setAlpha(0);

      var dist    = cfg.speed * (0.7 + Math.random() * 0.6);
      var dxP     = Math.cos(angle) * dist + (Math.random() - 0.5) * 25;
      var dyP     = Math.sin(angle) * dist + cfg.gravity + (Math.random() - 0.5) * 20;
      var life    = cfg.life * (0.7 + Math.random() * 0.5);
      var peakScale = 0.8 + Math.random() * 0.6;
      var stagger   = idx * 40 + Math.random() * 30;

      scene.tweens.add({
        targets: txt,
        x: px + dxP, y: py + dyP,
        alpha: { from: 1, to: 0 },
        scaleX: { from: 0, to: peakScale },
        scaleY: { from: 0, to: peakScale },
        duration: life, delay: stagger,
        ease: 'Cubic.easeOut',
        onComplete: function() { TOWN._returnParticle(entry); },
      });
    })(i);
  }
};
