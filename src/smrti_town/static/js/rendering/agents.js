/* ================================================================
   agents.js — createAgentSprite(), updateAgentSprite(),
               drawAgentCircle(), drawDriveBars()
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.createAgentSprite = function(scene, agent) {
  var name = agent.name;
  var color = TOWN.getAgentColor(name);
  var radius = TOWN.getAgentRadius(agent.life_stage);
  var pos = TOWN.getPlaceCenter(agent.location || 'Main_Street');
  var off = TOWN.getAgentOffset(name, agent.location);
  var x = pos.x + off.dx, y = pos.y + off.dy;

  /* Agent body circle */
  var circle = scene.add.graphics();
  TOWN.drawAgentCircle(circle, 0, 0, radius, color, agent.life_stage, agent.alive, 0, 0, agent.mood_valence || 0);
  circle.setPosition(x, y);
  circle.setDepth(11);

  /* Hitzone for click detection */
  var hitzone = scene.add.circle(x, y, radius + 6, 0x000000, 0)
    .setInteractive({ useHandCursor: true }).setDepth(12);
  hitzone.setData('agentName', name);

  /* Name label */
  var nameText = scene.add.text(x, y - radius - 12, name.replace(/_/g, ' '), {
    fontSize: '16px',
    fontFamily: 'Fredoka, sans-serif',
    fontStyle: 'bold',
    color: '#FFFFFF',
    align: 'center',
    stroke: '#5D3A1A',
    strokeThickness: 3,
  }).setOrigin(0.5, 1).setDepth(13);

  /* Drive bars (mini bar under agent) */
  var driveBarContainer = scene.add.graphics().setDepth(12);
  driveBarContainer.setPosition(x, y);

  /* Selection ring (hidden by default) */
  var selRing = scene.add.graphics().setDepth(10);
  selRing.setPosition(x, y);
  selRing.setAlpha(0);

  scene.agentLayer.add(circle);
  scene.agentLayer.add(hitzone);
  scene.agentLayer.add(nameText);
  scene.agentLayer.add(driveBarContainer);
  scene.agentLayer.add(selRing);

  var sprite = {
    circle: circle,
    hitzone: hitzone,
    nameText: nameText,
    driveBarContainer: driveBarContainer,
    selRing: selRing,
    speechBubble: null,
    _speechTimer: null,
    x: x, y: y,
    radius: radius,
    color: color,
    prevX: x, prevY: y,
    talking: false,
    _talkTween: null,
    _breatheTween: null,
  };
  TOWN.state.agentSprites[name] = sprite;

  /* ── Idle breathing animation ─────────────────────────────────── */
  /* Gentle scale oscillation so agents feel alive even when still.
     Each agent gets a randomized phase offset and slightly different
     duration so they don't pulse in unison. */
  if (agent.alive) {
    var breatheDur = 2200 + Math.random() * 800;
    var breatheDelay = Math.random() * breatheDur;
    sprite._breatheTween = scene.tweens.add({
      targets: circle,
      scaleX: { from: 1.0, to: 1.04 },
      scaleY: { from: 1.0, to: 1.04 },
      duration: breatheDur,
      yoyo: true,
      repeat: -1,
      delay: breatheDelay,
      ease: 'Sine.easeInOut',
    });
  }

  /* ── Pop-in entrance for newly spawned agents (births) ─────── */
  circle.setScale(0);
  nameText.setAlpha(0);
  driveBarContainer.setAlpha(0);
  scene.tweens.add({
    targets: circle,
    scaleX: 1, scaleY: 1,
    duration: 450,
    ease: 'Back.easeOut',
  });
  scene.tweens.add({
    targets: [nameText, driveBarContainer],
    alpha: 1,
    duration: 350,
    delay: 200,
    ease: 'Sine.easeOut',
  });

  return sprite;
};

TOWN.drawAgentCircle = function(gfx, x, y, radius, color, lifeStage, alive, dx, dy, moodValence) {
  gfx.clear();

  if (!alive) {
    /* Ghost / memorial — faded gray */
    gfx.fillStyle(0x999999, 0.35);
    gfx.fillCircle(x, y, radius);
    gfx.lineStyle(2, 0xaaaaaa, 0.25);
    gfx.strokeCircle(x, y, radius);
    return;
  }

  /* Outer glow */
  gfx.fillStyle(color, 0.12);
  gfx.fillCircle(x, y, radius + 6);

  /* Body */
  gfx.fillStyle(color, 1.0);
  gfx.fillCircle(x, y, radius);

  /* Mood tint — green for positive memories, red for negative */
  if (moodValence !== undefined && Math.abs(moodValence) > 0.15) {
    var tintColor = moodValence > 0 ? 0x44EE88 : 0xFF5555;
    var tintAlpha = Math.min(0.22, Math.abs(moodValence) * 0.3);
    gfx.fillStyle(tintColor, tintAlpha);
    gfx.fillCircle(x, y, radius);
  }

  /* Highlight (specular) */
  gfx.fillStyle(0xFFFFFF, 0.3);
  gfx.fillCircle(x - radius * 0.22, y - radius * 0.22, radius * 0.3);

  /* Life stage ring */
  if (lifeStage === 'elder') {
    gfx.lineStyle(3, 0xFFFFFF, 0.35);
    gfx.strokeCircle(x, y, radius + 1);
  } else if (lifeStage === 'child') {
    gfx.lineStyle(2, 0xFFD93D, 0.5);
    gfx.strokeCircle(x, y, radius);
  } else if (lifeStage === 'infant') {
    gfx.lineStyle(2, 0xFF6F91, 0.5);
    gfx.strokeCircle(x, y, radius);
  } else {
    gfx.lineStyle(2, 0x3D2B1F, 0.2);
    gfx.strokeCircle(x, y, radius + 1);
  }

  /* ── Eyes ──────────────────────────────────────────────────────── */
  /* Direction: normalize dx/dy so eyes look toward movement */
  var eyeSpread = radius * 0.32;
  var eyeY = y - radius * 0.12;
  var eyeR = Math.max(2.5, radius * 0.16);
  var pupilR = Math.max(1.5, eyeR * 0.55);

  /* Eye direction offset based on movement */
  var lookX = 0, lookY = 0;
  if (dx !== 0 || dy !== 0) {
    var mag = Math.sqrt(dx * dx + dy * dy);
    if (mag > 0) {
      lookX = (dx / mag) * eyeR * 0.35;
      lookY = (dy / mag) * eyeR * 0.35;
    }
  }

  /* Left eye */
  gfx.fillStyle(0xFFFFFF, 0.95);
  gfx.fillCircle(x - eyeSpread, eyeY, eyeR);
  gfx.fillStyle(0x1a1a1a, 1.0);
  gfx.fillCircle(x - eyeSpread + lookX, eyeY + lookY, pupilR);

  /* Right eye */
  gfx.fillStyle(0xFFFFFF, 0.95);
  gfx.fillCircle(x + eyeSpread, eyeY, eyeR);
  gfx.fillStyle(0x1a1a1a, 1.0);
  gfx.fillCircle(x + eyeSpread + lookX, eyeY + lookY, pupilR);
};

TOWN.drawDriveBars = function(gfx, radius, drives) {
  gfx.clear();
  if (!drives) return;

  var keys = Object.keys(drives);
  var barW = 46, barH = 4, gap = 2;
  var startY = radius + 10;
  var startX = -barW / 2;

  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    var val = Math.max(0, Math.min(100, drives[key]));
    var y = startY + i * (barH + gap);
    var c = Phaser.Display.Color.HexStringToColor(TOWN.DRIVE_COLORS[key] || '#888888').color;

    /* Bar background */
    gfx.fillStyle(0x3D2B1F, 0.2);
    gfx.fillRoundedRect(startX, y, barW, barH, 2);

    /* Bar fill */
    if (val > 0) {
      gfx.fillStyle(c, 0.85);
      gfx.fillRoundedRect(startX, y, barW * (val / 100), barH, 2);
    }
  }
};

TOWN.updateAgentSprite = function(scene, agent) {
  var name = agent.name;
  var sprite = TOWN.state.agentSprites[name];

  if (!sprite) {
    sprite = TOWN.createAgentSprite(scene, agent);
  }

  var radius = TOWN.getAgentRadius(agent.life_stage);
  var color = TOWN.getAgentColor(name);
  sprite.radius = radius;
  sprite.color = color;

  /* Target position */
  var pos = TOWN.getPlaceCenter(agent.location || 'Main_Street');
  var off = TOWN.getAgentOffset(name, agent.location);
  var tx = pos.x + off.dx, ty = pos.y + off.dy;

  /* Movement direction for eye tracking */
  var dx = tx - sprite.x;
  var dy = ty - sprite.y;

  /* Redraw circle with eye direction and mood tint */
  TOWN.drawAgentCircle(sprite.circle, 0, 0, radius, color, agent.life_stage, agent.alive, dx, dy, agent.mood_valence || 0);

  /* Redraw drive bars */
  TOWN.drawDriveBars(sprite.driveBarContainer, radius, agent.drives);

  /* Tween to new position if moved */
  if (Math.abs(sprite.x - tx) > 2 || Math.abs(sprite.y - ty) > 2) {
    var dur = 800;
    scene.tweens.add({ targets: sprite.circle, x: tx, y: ty, duration: dur, ease: 'Quad.easeInOut' });
    scene.tweens.add({ targets: sprite.hitzone, x: tx, y: ty, duration: dur, ease: 'Quad.easeInOut' });
    scene.tweens.add({ targets: sprite.nameText, x: tx, y: ty - radius - 12, duration: dur, ease: 'Quad.easeInOut' });
    scene.tweens.add({ targets: sprite.driveBarContainer, x: tx, y: ty, duration: dur, ease: 'Quad.easeInOut' });
    scene.tweens.add({ targets: sprite.selRing, x: tx, y: ty, duration: dur, ease: 'Quad.easeInOut' });
    sprite.prevX = sprite.x;
    sprite.prevY = sprite.y;
    sprite.x = tx;
    sprite.y = ty;
  }

  /* Dead agent — float up and fade */
  if (!agent.alive && sprite.circle.alpha > 0.3) {
    /* Kill the breathing tween so it doesn't fight the death anim */
    if (sprite._breatheTween) {
      sprite._breatheTween.stop();
      sprite._breatheTween = null;
    }
    scene.tweens.add({
      targets: [sprite.circle, sprite.selRing],
      alpha: 0.25,
      scaleX: 0.7, scaleY: 0.7,
      y: '-=18',
      duration: 2400,
      ease: 'Sine.easeInOut',
    });
    scene.tweens.add({
      targets: [sprite.nameText],
      alpha: 0.2,
      y: '-=18',
      duration: 2400,
      ease: 'Sine.easeInOut',
    });
    scene.tweens.add({
      targets: [sprite.driveBarContainer],
      alpha: 0, duration: 800,
    });
  }

  /* Selection ring update */
  if (TOWN.state.selectedAgent === name) {
    TOWN.showSelectionRing(scene, sprite);
  } else if (sprite.selRing.alpha > 0) {
    scene.tweens.killTweensOf(sprite.selRing);
    sprite.selRing.setAlpha(0);
  }
};

TOWN.showSelectionRing = function(scene, sprite) {
  if (sprite.selRing.alpha > 0) return;
  sprite.selRing.clear();
  sprite.selRing.lineStyle(3, 0xFFD93D, 0.7);
  sprite.selRing.strokeCircle(0, 0, sprite.radius + 10);
  scene.tweens.add({
    targets: sprite.selRing,
    alpha: { from: 0.3, to: 0.8 },
    duration: 800,
    yoyo: true,
    repeat: -1,
    ease: 'Sine.easeInOut',
  });
};

TOWN.startTalkBounce = function(scene, agentName) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite || sprite.talking) return;
  sprite.talking = true;
  sprite._talkTween = scene.tweens.add({
    targets: sprite.circle,
    scaleX: 1.08, scaleY: 1.08,
    duration: 400,
    yoyo: true,
    repeat: -1,
    ease: 'Sine.easeInOut',
  });
};

TOWN.stopTalkBounce = function(scene, agentName) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite || !sprite.talking) return;
  sprite.talking = false;
  if (sprite._talkTween) {
    sprite._talkTween.stop();
    sprite._talkTween = null;
  }
  sprite.circle.setScale(1, 1);
};

/* Relationship state → line color */
TOWN._REL_COLORS = {
  married:      0xFFD700,  /* gold */
  romantic:     0xFF69B4,  /* pink */
  close_friend: 0x6BCB77,  /* green */
  friend:       0x87CEEB,  /* sky blue */
  acquaintance: 0xBBBBBB,  /* light gray */
};

TOWN.drawRelationshipOverlay = function(scene) {
  var gfx = TOWN._relOverlayGfx;
  if (!gfx) return;
  gfx.clear();
  if (!TOWN.state.showRelOverlay) return;

  var agents = TOWN.state.agents;
  var sprites = TOWN.state.agentSprites;
  var drawn = {};

  for (var nameA in agents) {
    var a = agents[nameA];
    if (!a.alive || !a.relationships || !a.relationships.length) continue;
    var spA = sprites[nameA];
    if (!spA) continue;

    for (var r = 0; r < a.relationships.length; r++) {
      var rel = a.relationships[r];
      var nameB = rel.name;
      if (!nameB) continue;
      var pairKey = nameA < nameB ? nameA + ':' + nameB : nameB + ':' + nameA;
      if (drawn[pairKey]) continue;
      drawn[pairKey] = true;

      var spB = sprites[nameB];
      var agentB = agents[nameB];
      if (!spB || !agentB || !agentB.alive) continue;

      var lineColor = TOWN._REL_COLORS[rel.state] || 0xAAAAAA;
      var alpha = rel.state === 'acquaintance' ? 0.2 : 0.45;
      gfx.lineStyle(rel.state === 'married' || rel.state === 'romantic' ? 2.5 : 1.5, lineColor, alpha);
      gfx.beginPath();
      gfx.moveTo(spA.x, spA.y);
      gfx.lineTo(spB.x, spB.y);
      gfx.strokePath();
    }
  }
};

TOWN.highlightAgent = function(scene, name) {
  var sp = TOWN.state.agentSprites[name];
  if (!sp) return;
  scene.tweens.add({
    targets: sp.circle,
    scaleX: 1.2, scaleY: 1.2,
    duration: 150, yoyo: true,
    ease: 'Sine.easeOut',
  });
};
