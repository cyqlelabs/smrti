/* ================================================================
   buildings.js — Isometric 3-D town rendering
   ================================================================ */
window.TOWN = window.TOWN || {};

/* ── Color helpers ───────────────────────────────────────────────── */
TOWN._darken = function(hex, amount) {
  var c = Phaser.Display.Color.HexStringToColor(hex);
  return Phaser.Display.Color.GetColor(
    Math.max(0, c.red - amount),
    Math.max(0, c.green - amount),
    Math.max(0, c.blue - amount)
  );
};
TOWN._lighten = function(hex, amount) {
  var c = Phaser.Display.Color.HexStringToColor(hex);
  return Phaser.Display.Color.GetColor(
    Math.min(255, c.red + amount),
    Math.min(255, c.green + amount),
    Math.min(255, c.blue + amount)
  );
};
TOWN._hexToInt = function(hex) {
  return Phaser.Display.Color.HexStringToColor(hex).color;
};

/* ── Isometric box ───────────────────────────────────────────────── */
/* Draw a solid isometric box at world position (wx,wy),
   ww=world-width (x axis), wd=world-depth (y axis), wh=visual height. */
TOWN._drawIsoBox = function(gfx, wx, wy, ww, wd, wh,
                             topColor, leftColor, rightColor, borderAlpha) {
  var x1 = wx,      y1 = wy;
  var x2 = wx + ww, y2 = wy;
  var x3 = wx + ww, y3 = wy + wd;
  var x4 = wx,      y4 = wy + wd;

  /* Right face (east wall) */
  gfx.fillStyle(rightColor, 1.0);
  gfx.fillPoints([
    TOWN.isoProject(x2, y2, 0),
    TOWN.isoProject(x3, y3, 0),
    TOWN.isoProject(x3, y3, wh),
    TOWN.isoProject(x2, y2, wh),
  ], true);

  /* Left face (south wall) */
  gfx.fillStyle(leftColor, 1.0);
  gfx.fillPoints([
    TOWN.isoProject(x4, y4, 0),
    TOWN.isoProject(x3, y3, 0),
    TOWN.isoProject(x3, y3, wh),
    TOWN.isoProject(x4, y4, wh),
  ], true);

  /* Top face (roof base) */
  gfx.fillStyle(topColor, 1.0);
  gfx.fillPoints([
    TOWN.isoProject(x1, y1, wh),
    TOWN.isoProject(x2, y2, wh),
    TOWN.isoProject(x3, y3, wh),
    TOWN.isoProject(x4, y4, wh),
  ], true);

  /* Outline */
  if (borderAlpha > 0) {
    var bc = 0x2A1A0A;
    gfx.lineStyle(1.2, bc, borderAlpha);
    gfx.strokePoints([
      TOWN.isoProject(x1, y1, wh),
      TOWN.isoProject(x2, y2, wh),
      TOWN.isoProject(x3, y3, wh),
      TOWN.isoProject(x4, y4, wh),
    ], true);
    /* Vertical edges */
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].forEach(function(c) {
      gfx.lineStyle(1.0, bc, borderAlpha * 0.7);
      gfx.beginPath();
      var b = TOWN.isoProject(c[0], c[1], 0);
      var t = TOWN.isoProject(c[0], c[1], wh);
      gfx.moveTo(b.x, b.y); gfx.lineTo(t.x, t.y);
      gfx.strokePath();
    });
  }
};

/* ── Road strip ──────────────────────────────────────────────────── */
TOWN._drawRoad = function(gfx, pa, pb) {
  var aw = pa.w || 130, ah = pa.h || 100;
  var bw = pb.w || 130, bh = pb.h || 100;
  var ax = pa.x + aw / 2, ay = pa.y + ah / 2;
  var bx = pb.x + bw / 2, by = pb.y + bh / 2;
  var RW = 20;
  var dx = bx - ax, dy = by - ay;
  var len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return;
  var nx = -dy / len * RW / 2, ny = dx / len * RW / 2;
  var corners = [
    [ax + nx, ay + ny],
    [bx + nx, by + ny],
    [bx - nx, by - ny],
    [ax - nx, ay - ny],
  ];
  var pts = corners.map(function(c) { return TOWN.isoProject(c[0], c[1], 1); });
  gfx.fillStyle(0xC4B090, 0.9);
  gfx.fillPoints(pts, true);
  /* Pavements */
  var paveL = [
    [ax + nx * 1.3, ay + ny * 1.3],
    [bx + nx * 1.3, by + ny * 1.3],
    [bx + nx, by + ny],
    [ax + nx, ay + ny],
  ];
  var paveR = [
    [ax - nx, ay - ny],
    [bx - nx, by - ny],
    [bx - nx * 1.3, by - ny * 1.3],
    [ax - nx * 1.3, ay - ny * 1.3],
  ];
  gfx.fillStyle(0xD8C8A8, 0.7);
  gfx.fillPoints(paveL.map(function(c) { return TOWN.isoProject(c[0], c[1], 1); }), true);
  gfx.fillPoints(paveR.map(function(c) { return TOWN.isoProject(c[0], c[1], 1); }), true);
  /* Dashes */
  gfx.lineStyle(2, 0xF5EDD8, 0.3);
  gfx.beginPath();
  var ms = TOWN.isoProject(ax, ay, 1), me = TOWN.isoProject(bx, by, 1);
  gfx.moveTo(ms.x, ms.y); gfx.lineTo(me.x, me.y);
  gfx.strokePath();
};

/* ── Ground plane ─────────────────────────────────────────────────── */
TOWN._drawGround = function(gfx) {
  var pts = [
    TOWN.isoProject(-50,  -50,  0),
    TOWN.isoProject(1050, -50,  0),
    TOWN.isoProject(1050, 750,  0),
    TOWN.isoProject(-50,  750,  0),
  ];
  gfx.fillStyle(0xC8B898, 0.6);
  gfx.fillPoints(pts, true);
  /* Grid */
  gfx.lineStyle(1, 0xB0A080, 0.15);
  for (var gx = 0; gx <= 1000; gx += 80) {
    gfx.beginPath();
    var s = TOWN.isoProject(gx, -50, 0), e = TOWN.isoProject(gx, 750, 0);
    gfx.moveTo(s.x, s.y); gfx.lineTo(e.x, e.y); gfx.strokePath();
  }
  for (var gy = 0; gy <= 700; gy += 80) {
    gfx.beginPath();
    var s = TOWN.isoProject(-50, gy, 0), e = TOWN.isoProject(1050, gy, 0);
    gfx.moveTo(s.x, s.y); gfx.lineTo(e.x, e.y); gfx.strokePath();
  }
};

/* ── Window helper ───────────────────────────────────────────────── */
TOWN._drawWindows = function(gfx, wx, wy, ww, wd, wh, rows, cols) {
  var winColor = 0xADD8FF;
  /* Windows on south-facing wall (left face) */
  for (var row = 0; row < rows; row++) {
    for (var col = 0; col < cols; col++) {
      var wX  = wx + ww * (0.15 + col * (0.7 / Math.max(cols - 1, 1)));
      var wY  = wy + wd;
      var wZb = wh * (0.2 + row * 0.38);
      var wZt = wZb + wh * 0.22;
      var wD  = wd * 0.10;
      gfx.fillStyle(winColor, 0.55);
      gfx.fillPoints([
        TOWN.isoProject(wX,       wY,      wZb),
        TOWN.isoProject(wX + wD,  wY,      wZb),
        TOWN.isoProject(wX + wD,  wY,      wZt),
        TOWN.isoProject(wX,       wY,      wZt),
      ], true);
      gfx.lineStyle(0.6, 0x4A7090, 0.4);
      gfx.strokePoints([
        TOWN.isoProject(wX,       wY,      wZb),
        TOWN.isoProject(wX + wD,  wY,      wZb),
        TOWN.isoProject(wX + wD,  wY,      wZt),
        TOWN.isoProject(wX,       wY,      wZt),
      ], true);
    }
  }
};

/* ── Main entry points ───────────────────────────────────────────── */

TOWN.drawTown = function(scene, town) {
  var gfx = scene.roadLayer;
  gfx.clear();

  TOWN._drawGround(gfx);

  /* Roads */
  if (town.connections) {
    town.connections.forEach(function(pair) {
      var pa = town.places[pair[0]], pb = town.places[pair[1]];
      if (!pa || !pb) return;
      var isStreetA = pair[0].toLowerCase().indexOf('street') !== -1;
      var isStreetB = pair[1].toLowerCase().indexOf('street') !== -1;
      if (isStreetA || isStreetB) TOWN._drawRoad(gfx, pa, pb);
    });
  }

  /* Buildings sorted by iso depth (x+y ascending = draw back-to-front) */
  var names = Object.keys(town.places);
  names.sort(function(a, b) {
    var pa = town.places[a], pb = town.places[b];
    return (pa.x + pa.y) - (pb.x + pb.y);
  });
  for (var i = 0; i < names.length; i++) {
    TOWN.drawBuilding(scene, names[i], town.places[names[i]]);
  }
};

TOWN.drawBuilding = function(scene, name, place) {
  var w = place.w || 130, h = place.h || 100;
  var isStreet = name.toLowerCase().indexOf('street') !== -1;
  if (isStreet) return; /* streets drawn as roads */

  var placeType = place.place_type || 'other';
  var colorHex  = (place.color && place.color !== '#888888')
    ? place.color
    : (TOWN.PLACE_TYPE_COLORS[placeType] || '#888888');

  var topColor   = TOWN._lighten(colorHex, 35);
  var leftColor  = TOWN._hexToInt(colorHex);
  var rightColor = TOWN._darken(colorHex, 30);

  /* Height varies by place type */
  var boxH = 55;
  if (placeType === 'home')    boxH = 48;
  if (placeType === 'outdoor') boxH = 6;
  if (placeType === 'public')  boxH = 68;

  var bg = scene.add.graphics().setDepth(2);

  /* Drop shadow */
  var shadowPts = [
    TOWN.isoProject(place.x + 10, place.y + 10, 0),
    TOWN.isoProject(place.x + w + 10, place.y + 10, 0),
    TOWN.isoProject(place.x + w + 10, place.y + h + 10, 0),
    TOWN.isoProject(place.x + 10, place.y + h + 10, 0),
  ];
  bg.fillStyle(0x1A0A00, 0.20);
  bg.fillPoints(shadowPts, true);

  if (placeType === 'outdoor') {
    /* Park: green ground patch with path and tree positions */
    var groundPts = [
      TOWN.isoProject(place.x,     place.y,     0),
      TOWN.isoProject(place.x + w, place.y,     0),
      TOWN.isoProject(place.x + w, place.y + h, 0),
      TOWN.isoProject(place.x,     place.y + h, 0),
    ];
    bg.fillStyle(0x3A8C3A, 0.9);
    bg.fillPoints(groundPts, true);
    /* Inner lighter patch */
    var margin = 12;
    bg.fillStyle(0x4EA84E, 0.6);
    bg.fillPoints([
      TOWN.isoProject(place.x + margin,     place.y + margin,     0),
      TOWN.isoProject(place.x + w - margin, place.y + margin,     0),
      TOWN.isoProject(place.x + w - margin, place.y + h - margin, 0),
      TOWN.isoProject(place.x + margin,     place.y + h - margin, 0),
    ], true);
    /* Path diagonal */
    bg.lineStyle(8, 0xC8A870, 0.35);
    bg.beginPath();
    var pA = TOWN.isoProject(place.x + w * 0.1, place.y + h * 0.5, 1);
    var pB = TOWN.isoProject(place.x + w * 0.9, place.y + h * 0.5, 1);
    bg.moveTo(pA.x, pA.y); bg.lineTo(pB.x, pB.y); bg.strokePath();
    bg.lineStyle(2, 0x2A6A2A, 0.5);
    bg.strokePoints(groundPts, true);

  } else {
    /* Standard isometric building */
    TOWN._drawIsoBox(bg, place.x, place.y, w, h, boxH,
                     topColor, leftColor, rightColor, 0.28);

    /* Windows */
    var winRows = (placeType === 'public') ? 2 : 1;
    var winCols = (placeType === 'public') ? 3 : 2;
    TOWN._drawWindows(bg, place.x, place.y, w, h, boxH, winRows, winCols);

    /* Door on south face */
    var doorX  = place.x + w * 0.42;
    var doorY  = place.y + h;
    var doorW  = w * 0.12;
    var doorHt = boxH * 0.38;
    bg.fillStyle(TOWN._darken(colorHex, 55), 1.0);
    bg.fillPoints([
      TOWN.isoProject(doorX,        doorY, 0),
      TOWN.isoProject(doorX + doorW,doorY, 0),
      TOWN.isoProject(doorX + doorW,doorY, doorHt),
      TOWN.isoProject(doorX,        doorY, doorHt),
    ], true);
    bg.lineStyle(0.7, 0x1A0A00, 0.4);
    bg.strokePoints([
      TOWN.isoProject(doorX,        doorY, 0),
      TOWN.isoProject(doorX + doorW,doorY, 0),
      TOWN.isoProject(doorX + doorW,doorY, doorHt),
      TOWN.isoProject(doorX,        doorY, doorHt),
    ], true);

    if (placeType === 'home') {
      /* Pitched roof */
      var peakX = place.x + w / 2, peakY = place.y + h / 2;
      var peakZ = boxH + 26;
      var roofColor  = TOWN._darken(colorHex, 45);
      var roofColor2 = TOWN._darken(colorHex, 60);
      var c1 = TOWN.isoProject(place.x,     place.y,     boxH);
      var c2 = TOWN.isoProject(place.x + w, place.y,     boxH);
      var c3 = TOWN.isoProject(place.x + w, place.y + h, boxH);
      var c4 = TOWN.isoProject(place.x,     place.y + h, boxH);
      var peak = TOWN.isoProject(peakX, peakY, peakZ);
      /* Front slope */
      bg.fillStyle(roofColor2, 1.0);
      bg.fillTriangle(c4.x, c4.y, c3.x, c3.y, peak.x, peak.y);
      /* Left slope */
      bg.fillStyle(roofColor, 1.0);
      bg.fillTriangle(c1.x, c1.y, c4.x, c4.y, peak.x, peak.y);
      /* Back slope (lighter, top face) */
      bg.fillStyle(TOWN._lighten(colorHex, -20), 1.0);
      bg.fillTriangle(c1.x, c1.y, c2.x, c2.y, peak.x, peak.y);
      /* Right slope */
      bg.fillStyle(TOWN._darken(colorHex, 35), 1.0);
      bg.fillTriangle(c2.x, c2.y, c3.x, c3.y, peak.x, peak.y);
      /* Ridge outline */
      bg.lineStyle(1, 0x1A0A00, 0.3);
      bg.strokeTriangle(c4.x, c4.y, c3.x, c3.y, peak.x, peak.y);
      bg.strokeTriangle(c1.x, c1.y, c4.x, c4.y, peak.x, peak.y);

      /* Chimney */
      var chiX = place.x + w * 0.72, chiY = place.y + h * 0.28;
      TOWN._drawIsoBox(bg, chiX, chiY, w * 0.08, h * 0.07, peakZ - 10,
                       TOWN._darken(colorHex, 50), TOWN._darken(colorHex, 65), TOWN._darken(colorHex, 75), 0.2);

    } else if (placeType === 'public') {
      /* Flat roof with parapet crenellations */
      var mW = w * 0.10, mD = h * 0.10, mH = 10;
      var merColor = topColor;
      for (var m = 0; m < 4; m++) {
        var mx = place.x + w * 0.08 + m * (w * 0.23);
        TOWN._drawIsoBox(bg, mx, place.y + h * 0.05, mW, mD, boxH + mH,
                         merColor, leftColor, rightColor, 0.12);
      }
      /* Roof detail stripe */
      bg.fillStyle(TOWN._darken(colorHex, 15), 0.6);
      bg.fillPoints([
        TOWN.isoProject(place.x,          place.y + 5, boxH),
        TOWN.isoProject(place.x + w,      place.y + 5, boxH),
        TOWN.isoProject(place.x + w,      place.y + h * 0.1, boxH),
        TOWN.isoProject(place.x,          place.y + h * 0.1, boxH),
      ], true);
    }
  }

  scene.buildingLayer.add(bg);

  /* Icon floating above building */
  if (place.icon) {
    var iconZ = (placeType === 'outdoor') ? 6 : boxH + 14;
    var iconPos = TOWN.isoProject(place.x + w / 2, place.y + h / 2, iconZ);
    var icon = scene.add.text(iconPos.x, iconPos.y, place.icon, {
      fontSize: (placeType === 'outdoor') ? '26px' : '22px',
      align: 'center',
    }).setOrigin(0.5, 1).setDepth(3);
    scene.buildingLayer.add(icon);
    scene.tweens.add({
      targets: icon, y: iconPos.y - 5,
      duration: 2600 + Math.random() * 800,
      yoyo: true, repeat: -1,
      delay: Math.random() * 2000,
      ease: 'Sine.easeInOut',
    });

    /* Extra trees for parks */
    if (placeType === 'outdoor') {
      var treePos = [
        [place.x + w * 0.18, place.y + h * 0.22],
        [place.x + w * 0.75, place.y + h * 0.18],
        [place.x + w * 0.55, place.y + h * 0.72],
        [place.x + w * 0.12, place.y + h * 0.68],
        [place.x + w * 0.82, place.y + h * 0.62],
      ];
      treePos.forEach(function(tp, ti) {
        var tP = TOWN.isoProject(tp[0], tp[1], 2);
        var tree = scene.add.text(tP.x, tP.y, '🌳', {
          fontSize: (16 + (ti % 3) * 5) + 'px',
        }).setOrigin(0.5, 1).setDepth(3);
        scene.buildingLayer.add(tree);
        scene.tweens.add({
          targets: tree, y: tP.y - 4,
          duration: 2800 + ti * 450,
          yoyo: true, repeat: -1,
          delay: ti * 310,
          ease: 'Sine.easeInOut',
        });
      });
    }
  }

  /* Label */
  var labelPos = TOWN.isoProject(place.x + w / 2, place.y + h + 8, 0);
  var label = scene.add.text(labelPos.x, labelPos.y, place.label || name.replace(/_/g, ' '), {
    fontSize: '13px',
    fontFamily: 'Fredoka, sans-serif',
    fontStyle: 'bold',
    color: '#FFF8F0',
    align: 'center',
    stroke: '#3D2B1F',
    strokeThickness: 3,
  }).setOrigin(0.5, 0).setDepth(4);
  scene.buildingLayer.add(label);

  /* Occupant counter */
  var counterPos = TOWN.isoProject(place.x + w, place.y, boxH + 8);
  var counter = scene.add.text(counterPos.x, counterPos.y, '', {
    fontSize: '11px',
    fontFamily: 'Fredoka, sans-serif',
    fontStyle: 'bold',
    color: '#FFFFFF',
    backgroundColor: '#5D3A1A',
    padding: { x: 5, y: 2 },
  }).setOrigin(0.5, 0.5).setVisible(false).setDepth(5);
  scene.buildingLayer.add(counter);

  /* Glow */
  var glow = scene.add.graphics().setDepth(1).setAlpha(0);
  scene.buildingLayer.add(glow);

  /* Hitzone — iso-projected area */
  var hzC = TOWN.isoProject(place.x + w / 2, place.y + h / 2, boxH / 2);
  var hitzone = scene.add.rectangle(hzC.x, hzC.y, w * 0.9, h * 0.5, 0x000000, 0)
    .setInteractive({ useHandCursor: true }).setDepth(6);
  hitzone.setData('placeKey', name);
  scene.buildingLayer.add(hitzone);

  TOWN.state.placeSprites[name] = {
    bg: bg,
    glow: glow,
    label: label,
    counter: counter,
    hitzone: hitzone,
    x: place.x, y: place.y,
    w: w, h: h,
    boxH: boxH,
    placeType: placeType,
    color: TOWN._hexToInt(colorHex),
    colorHex: colorHex,
  };
};

TOWN.updateOccupantCounts = function() {
  var counts = {};
  var all = TOWN.state.agents;
  for (var n in all) {
    if (all[n].alive && all[n].location) {
      counts[all[n].location] = (counts[all[n].location] || 0) + 1;
    }
  }
  var sprites = TOWN.state.placeSprites;
  for (var name in sprites) {
    var sp = sprites[name];
    var c = counts[name] || 0;
    if (c > 0) {
      sp.counter.setText(c + ' \uD83D\uDC64');
      sp.counter.setVisible(true);
      /* Glow */
      var gi = Math.min(0.06 + c * 0.03, 0.18);
      sp.glow.clear();
      sp.glow.fillStyle(0xFFD93D, gi);
      var gpad = 4 + Math.min(c, 5) * 1.5;
      /* Draw glow as flat iso diamond */
      sp.glow.fillPoints([
        TOWN.isoProject(sp.x - gpad,     sp.y - gpad,     0),
        TOWN.isoProject(sp.x + sp.w + gpad, sp.y - gpad,  0),
        TOWN.isoProject(sp.x + sp.w + gpad, sp.y + sp.h + gpad, 0),
        TOWN.isoProject(sp.x - gpad,     sp.y + sp.h + gpad, 0),
      ], true);
      if (sp.glow.alpha < 0.05 && TOWN.state.scene) {
        TOWN.state.scene.tweens.add({
          targets: sp.glow,
          alpha: { from: gi * 0.5, to: gi },
          duration: 1600,
          yoyo: true, repeat: -1,
          ease: 'Sine.easeInOut',
        });
      }
    } else {
      sp.counter.setVisible(false);
      if (sp.glow.alpha > 0 && TOWN.state.scene) {
        TOWN.state.scene.tweens.killTweensOf(sp.glow);
        TOWN.state.scene.tweens.add({ targets: sp.glow, alpha: 0, duration: 600 });
      }
    }
  }
};
