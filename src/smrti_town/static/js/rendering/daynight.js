/* ================================================================
   daynight.js — updateDayNight() cycle, season tinting, stars
   ================================================================ */
window.TOWN = window.TOWN || {};

/* Module-level TOD/season change trackers */
TOWN._lastWindowTod   = null;
TOWN._lastSeasonTint  = null;

TOWN.updateDayNight = function(scene) {
  var tod    = TOWN.state.calendar.time_of_day || 'morning';
  var season = TOWN.state.calendar.season      || 'spring';
  var todCfg = TOWN.TOD_COLORS[tod] || TOWN.TOD_COLORS.morning;

  /* Background color — smooth transition */
  scene.cameras.main.setBackgroundColor(todCfg.bg);

  /* Night/evening overlay alpha — lerp toward target */
  if (scene.nightOverlay) {
    var target = todCfg.alpha;
    var current = scene.nightOverlay.alpha;
    var newAlpha = current + (target - current) * 0.04;
    scene.nightOverlay.setAlpha(newAlpha);
    scene.nightOverlay.setFillStyle(todCfg.overlay, newAlpha);
  }

  /* Season tint overlay — update when season changes */
  if (season !== TOWN._lastSeasonTint && TOWN._seasonOverlay) {
    TOWN._lastSeasonTint = season;
    var sc = TOWN.SEASON_TINTS[season];
    if (sc) {
      TOWN._seasonOverlay.setFillStyle(sc.overlay, 1);
      scene.tweens.add({
        targets: TOWN._seasonOverlay,
        alpha: sc.alpha,
        duration: 3000,
        ease: 'Sine.easeInOut',
      });
    }
  }

  /* Window lighting — update when TOD changes */
  if (tod !== TOWN._lastWindowTod) {
    TOWN.updateWindowLighting();
  }

  /* Stars at night */
  if (tod === 'night') {
    TOWN._ensureStars(scene);
    TOWN._twinkleStars(scene);
  } else {
    TOWN._hideStars(scene);
  }
};

/* ── Lit window overlay ───────────────────────────────────────────── */
TOWN.updateWindowLighting = function() {
  var gfx = TOWN._windowLayerGfx;
  if (!gfx) return;
  var tod = TOWN.state.calendar && TOWN.state.calendar.time_of_day;
  TOWN._lastWindowTod = tod;
  gfx.clear();

  var isNight = (tod === 'night' || tod === 'evening');
  if (!isNight) return;

  var town = TOWN.state.town;
  if (!town || !town.places) return;
  var sprites = TOWN.state.placeSprites;

  for (var pName in sprites) {
    var sp   = sprites[pName];
    var place = town.places[pName];
    if (!place) continue;
    var pt = sp.placeType || place.place_type || 'other';
    if (pt === 'outdoor') continue;
    if (pName.toLowerCase().indexOf('street') !== -1) continue;

    var w = sp.w, h = sp.h, bH = sp.boxH;
    var wx = sp.x, wy = sp.y;
    var winRows = (pt === 'public') ? 2 : 1;
    var winCols = (pt === 'public') ? 3 : 2;

    /* Deterministic per-window hash for consistent lit pattern */
    var nameHash = 0;
    for (var ci = 0; ci < pName.length; ci++) {
      nameHash = (nameHash * 31 + pName.charCodeAt(ci)) & 0xFFFF;
    }
    var litThreshold = (tod === 'night') ? 0.65 : 0.42;

    for (var row = 0; row < winRows; row++) {
      for (var col = 0; col < winCols; col++) {
        var winHash = (nameHash + row * 7919 + col * 1009) & 0xFFFF;
        if ((winHash / 0xFFFF) > litThreshold) continue;

        var wX  = wx + w  * (0.15 + col * (0.7 / Math.max(winCols - 1, 1)));
        var wY  = wy + h;
        var wZb = bH * (0.2 + row * 0.38);
        var wZt = wZb + bH * 0.22;
        var wD  = h * 0.10;
        var alpha = (tod === 'night') ? 0.88 : 0.58;

        gfx.fillStyle(0xFFE87A, alpha);
        gfx.fillPoints([
          TOWN.isoProject(wX,      wY, wZb),
          TOWN.isoProject(wX + wD, wY, wZb),
          TOWN.isoProject(wX + wD, wY, wZt),
          TOWN.isoProject(wX,      wY, wZt),
        ], true);
      }
    }
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
