/* ================================================================
   buildings.js — drawTown(), drawBuilding(), updateOccupantCounts()
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.drawTown = function(scene, town) {
  var gfx = scene.roadLayer;
  gfx.clear();

  /* ── Roads ─────────────────────────────────────────────────────── */
  if (town.connections) {
    /* Road fill — warm stone color, thick */
    gfx.lineStyle(14, 0xB8A88A, 0.7);
    town.connections.forEach(function(pair) {
      var pa = town.places[pair[0]], pb = town.places[pair[1]];
      if (!pa || !pb) return;
      var ax = pa.x + (pa.w || 130) / 2, ay = pa.y + (pa.h || 100) / 2;
      var bx = pb.x + (pb.w || 130) / 2, by = pb.y + (pb.h || 100) / 2;
      gfx.beginPath();
      gfx.moveTo(ax, ay);
      gfx.lineTo(bx, by);
      gfx.strokePath();
    });

    /* Center dashes — lighter */
    gfx.lineStyle(2, 0xF5E6D0, 0.35);
    town.connections.forEach(function(pair) {
      var pa = town.places[pair[0]], pb = town.places[pair[1]];
      if (!pa || !pb) return;
      var ax = pa.x + (pa.w || 130) / 2, ay = pa.y + (pa.h || 100) / 2;
      var bx = pb.x + (pb.w || 130) / 2, by = pb.y + (pb.h || 100) / 2;
      gfx.beginPath();
      gfx.moveTo(ax, ay);
      gfx.lineTo(bx, by);
      gfx.strokePath();
    });
  }

  /* ── Buildings ──────────────────────────────────────────────────── */
  var names = Object.keys(town.places);
  for (var i = 0; i < names.length; i++) {
    TOWN.drawBuilding(scene, names[i], town.places[names[i]]);
  }
};

TOWN.drawBuilding = function(scene, name, place) {
  var w = place.w || 130, h = place.h || 100;
  var colorHex = place.color || '#888888';
  var color = Phaser.Display.Color.HexStringToColor(colorHex).color;
  var isStreet = name.toLowerCase().indexOf('street') !== -1;
  var radius = isStreet ? 6 : 16;
  var roofH = isStreet ? 0 : 15;

  /* Drop shadow */
  var shadow = scene.add.graphics();
  shadow.fillStyle(0x3D2B1F, 0.2);
  shadow.fillRoundedRect(place.x + 4, place.y + 4, w, h, radius);
  scene.buildingLayer.add(shadow);

  /* Main body */
  var bg = scene.add.graphics();
  bg.fillStyle(color, isStreet ? 0.5 : 1.0);
  bg.fillRoundedRect(place.x, place.y, w, h, radius);
  scene.buildingLayer.add(bg);

  /* Roof strip (darker) */
  if (roofH > 0) {
    var darker = Phaser.Display.Color.IntegerToColor(color);
    var roofColor = Phaser.Display.Color.GetColor(
      Math.max(0, darker.red - 40),
      Math.max(0, darker.green - 40),
      Math.max(0, darker.blue - 40)
    );
    var roofGfx = scene.add.graphics();
    /* Draw roof with top corners rounded only */
    roofGfx.fillStyle(roofColor, 1.0);
    roofGfx.fillRoundedRect(place.x, place.y, w, roofH + radius, { tl: radius, tr: radius, bl: 0, br: 0 });
    /* Clip the bottom part that extends past the roof */
    roofGfx.fillStyle(color, 1.0);
    roofGfx.fillRect(place.x, place.y + roofH, w, radius);
    scene.buildingLayer.add(roofGfx);
  }

  /* Soft border */
  var border = scene.add.graphics();
  border.lineStyle(2, 0x3D2B1F, 0.15);
  border.strokeRoundedRect(place.x, place.y, w, h, radius);
  scene.buildingLayer.add(border);

  /* Icon (large, centered) with gentle idle float */
  if (place.icon) {
    var iconY = place.y + h / 2 - 14;
    var icon = scene.add.text(place.x + w / 2, iconY, place.icon, {
      fontSize: '36px', align: 'center',
    }).setOrigin(0.5);
    scene.buildingLayer.add(icon);

    /* Subtle vertical bob — each building offset so they don't sync */
    scene.tweens.add({
      targets: icon,
      y: iconY - 3,
      duration: 2600 + Math.random() * 800,
      yoyo: true,
      repeat: -1,
      delay: Math.random() * 2000,
      ease: 'Sine.easeInOut',
    });
  }

  /* Label */
  var labelY = place.icon ? place.y + h / 2 + 18 : place.y + h / 2;
  var label = scene.add.text(
    place.x + w / 2, labelY,
    place.label || name.replace(/_/g, ' '),
    {
      fontSize: isStreet ? '12px' : '16px',
      fontFamily: 'Fredoka, sans-serif',
      fontStyle: 'bold',
      color: '#FFFFFF',
      align: 'center',
      stroke: '#3D2B1F',
      strokeThickness: isStreet ? 2 : 3,
    }
  ).setOrigin(0.5);
  scene.buildingLayer.add(label);

  /* Occupant counter badge */
  var counter = scene.add.text(place.x + w - 10, place.y + 10, '', {
    fontSize: '13px',
    fontFamily: 'Fredoka, sans-serif',
    fontStyle: 'bold',
    color: '#FFFFFF',
    backgroundColor: '#5D3A1A',
    padding: { x: 6, y: 3 },
  }).setOrigin(1, 0).setVisible(false);
  scene.buildingLayer.add(counter);

  /* Glow graphic (hidden by default, shown when occupied) */
  var glow = scene.add.graphics();
  glow.setAlpha(0);
  scene.buildingLayer.add(glow);

  /* Click hitzone */
  var hitzone = scene.add.rectangle(
    place.x + w / 2, place.y + h / 2, w, h, 0x000000, 0
  ).setInteractive({ useHandCursor: true });
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
    color: color,
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

      /* Glow intensity scales with occupants: 1→subtle, 3+→warm */
      var glowIntensity = Math.min(0.06 + c * 0.03, 0.18);
      var glowPad = 3 + Math.min(c, 5) * 1.5;

      /* Redraw glow at new intensity / spread */
      sp.glow.clear();
      sp.glow.fillStyle(0xFFD93D, glowIntensity);
      sp.glow.fillRoundedRect(
        sp.x - glowPad, sp.y - glowPad,
        sp.w + glowPad * 2, sp.h + glowPad * 2, 20
      );

      /* Pulse — only start if not already pulsing */
      if (sp.glow.alpha < 0.05 && TOWN.state.scene) {
        TOWN.state.scene.tweens.add({
          targets: sp.glow,
          alpha: { from: glowIntensity * 0.6, to: glowIntensity },
          duration: 1600 - Math.min(c, 4) * 150,
          yoyo: true,
          repeat: -1,
          ease: 'Sine.easeInOut',
        });
      }
    } else {
      sp.counter.setVisible(false);
      if (sp.glow.alpha > 0) {
        /* Fade out glow gracefully */
        if (TOWN.state.scene) {
          TOWN.state.scene.tweens.killTweensOf(sp.glow);
          TOWN.state.scene.tweens.add({
            targets: sp.glow,
            alpha: 0,
            duration: 600,
            ease: 'Sine.easeIn',
          });
        }
      }
    }
  }
};
