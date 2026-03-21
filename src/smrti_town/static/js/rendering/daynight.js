/* ================================================================
   daynight.js — updateDayNight() cycle, season tinting, stars
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.updateDayNight = function(scene) {
  var tod = TOWN.state.calendar.time_of_day || 'morning';
  var todCfg = TOWN.TOD_COLORS[tod] || TOWN.TOD_COLORS.morning;

  /* Background color — smooth transition */
  scene.cameras.main.setBackgroundColor(todCfg.bg);

  /* Night/evening overlay alpha — lerp toward target */
  if (scene.nightOverlay) {
    var target = todCfg.alpha;
    var current = scene.nightOverlay.alpha;
    var newAlpha = current + (target - current) * 0.04;
    scene.nightOverlay.setAlpha(newAlpha);

    /* Overlay tint color */
    scene.nightOverlay.setFillStyle(todCfg.overlay, newAlpha);
  }

  /* Stars at night */
  if (tod === 'night') {
    TOWN._ensureStars(scene);
    TOWN._twinkleStars(scene);
  } else {
    TOWN._hideStars(scene);
  }
};

TOWN._ensureStars = function(scene) {
  if (TOWN.state.stars.length > 0) return;

  /* Create random star positions across the game area */
  var gameW = scene.scale.width || 1000;
  var gameH = scene.scale.height || 700;

  for (var i = 0; i < 40; i++) {
    var sx = 30 + Math.random() * (gameW - 60);
    var sy = 30 + Math.random() * (gameH * 0.5);
    var size = 1 + Math.random() * 2.5;

    var star = scene.add.graphics().setDepth(6);
    star.fillStyle(0xFFFFFF, 0.7);
    star.fillCircle(sx, sy, size);
    star.setAlpha(0);
    TOWN.state.stars.push(star);

    /* Fade in */
    scene.tweens.add({
      targets: star,
      alpha: 0.3 + Math.random() * 0.6,
      duration: 1000 + Math.random() * 2000,
      ease: 'Sine.easeInOut',
    });
  }
};

TOWN._twinkleStars = function(scene) {
  /* Twinkle 2-3 stars per frame call for a livelier sky */
  var stars = TOWN.state.stars;
  if (stars.length === 0) return;

  /* Higher probability → more stars twinkling at any moment */
  for (var t = 0; t < 2; t++) {
    if (Math.random() < 0.08) {
      var idx = Math.floor(Math.random() * stars.length);
      var star = stars[idx];
      if (star && star.alpha > 0.05 && !star.getData('twinkling')) {
        star.setData('twinkling', true);
        var dip = 0.05 + Math.random() * 0.1;
        var dur = 300 + Math.random() * 400;
        scene.tweens.add({
          targets: star,
          alpha: { from: star.alpha, to: dip },
          duration: dur,
          yoyo: true,
          ease: 'Sine.easeInOut',
          onComplete: function() { this.targets[0].setData('twinkling', false); },
        });
      }
    }
  }
};

TOWN._hideStars = function(scene) {
  var stars = TOWN.state.stars;
  if (stars.length === 0) return;

  for (var i = 0; i < stars.length; i++) {
    var star = stars[i];
    if (star.alpha > 0) {
      scene.tweens.add({
        targets: star,
        alpha: 0,
        duration: 800,
        ease: 'Sine.easeIn',
        onComplete: function() {
          this.targets[0].destroy();
        },
      });
    } else {
      star.destroy();
    }
  }
  TOWN.state.stars = [];
};
