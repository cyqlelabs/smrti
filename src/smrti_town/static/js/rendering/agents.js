/* ================================================================
   agents.js — Isometric person sprites with walk animation
   ================================================================ */
window.TOWN = window.TOWN || {};

/* ── Color helpers ───────────────────────────────────────────────── */
TOWN._darkenInt = function(colorInt, amount) {
  var r = (colorInt >> 16) & 0xFF;
  var g = (colorInt >> 8)  & 0xFF;
  var b =  colorInt        & 0xFF;
  return Phaser.Display.Color.GetColor(
    Math.max(0, r - amount),
    Math.max(0, g - amount),
    Math.max(0, b - amount)
  );
};
TOWN._lightenInt = function(colorInt, amount) {
  var r = (colorInt >> 16) & 0xFF;
  var g = (colorInt >> 8)  & 0xFF;
  var b =  colorInt        & 0xFF;
  return Phaser.Display.Color.GetColor(
    Math.min(255, r + amount),
    Math.min(255, g + amount),
    Math.min(255, b + amount)
  );
};

/* ── Person renderer ─────────────────────────────────────────────── */
/* Draw a miniature person at canvas pixel position (sx, sy — base feet).
   Walking: if true, animate limbs based on walkPhase (0–1). */
TOWN.drawPerson = function(gfx, sx, sy, color, lifeStage, alive, walking, walkPhase, moodValence) {
  gfx.clear();

  if (!alive) {
    gfx.fillStyle(0x888888, 0.25);
    gfx.fillEllipse(sx, sy - 4, 18, 8);
    return;
  }

  var s = 1.0;
  if (lifeStage === 'infant') s = 0.42;
  else if (lifeStage === 'child')  s = 0.62;
  else if (lifeStage === 'elder')  s = 0.85;

  var headR = 9  * s;
  var bodyH = 14 * s;
  var bodyW = 8  * s;
  var legLen = 10 * s;
  var armLen = 8  * s;
  var fw = 4 * s;  /* foot width */

  /* Walk cycle */
  var lSwing = walking ? Math.sin(walkPhase * Math.PI * 2) * 5 * s : 0;
  var aSwing = -lSwing * 0.7;

  /* Base y = sy (feet level) */
  var feetY  = sy;
  var bodyY  = feetY - legLen - bodyH;
  var headY  = bodyY - headR;

  /* Shadow */
  gfx.fillStyle(0x1A0A00, 0.16);
  gfx.fillEllipse(sx, feetY + 2, 16 * s, 6 * s);

  /* Legs */
  var legColor = TOWN._darkenInt(color, 45);
  gfx.fillStyle(legColor, 1.0);
  /* Left leg */
  gfx.fillRect(sx - bodyW * 0.48 - 1, feetY - legLen + lSwing, fw, legLen - Math.abs(lSwing) * 0.4);
  /* Right leg */
  gfx.fillRect(sx + bodyW * 0.48 - fw + 1, feetY - legLen - lSwing, fw, legLen - Math.abs(lSwing) * 0.4);
  /* Feet */
  gfx.fillStyle(TOWN._darkenInt(color, 65), 1.0);
  gfx.fillRect(sx - bodyW * 0.48 - 2, feetY + lSwing * 0.3 - 3, fw + 3, 3 * s);
  gfx.fillRect(sx + bodyW * 0.48 - fw, feetY - lSwing * 0.3 - 3, fw + 3, 3 * s);

  /* Body */
  gfx.fillStyle(color, 1.0);
  gfx.fillRoundedRect(sx - bodyW / 2, bodyY, bodyW, bodyH, 3 * s);
  /* Collar line */
  gfx.fillStyle(TOWN._lightenInt(color, 30), 0.45);
  gfx.fillRect(sx - bodyW * 0.28, bodyY, bodyW * 0.56, bodyH * 0.22);
  /* Mood body tint */
  if (moodValence && Math.abs(moodValence) > 0.15) {
    var tint = moodValence > 0 ? 0x44EE88 : 0xFF5555;
    gfx.fillStyle(tint, Math.min(0.20, Math.abs(moodValence) * 0.28));
    gfx.fillRoundedRect(sx - bodyW / 2, bodyY, bodyW, bodyH, 3 * s);
  }

  /* Arms */
  gfx.lineStyle(2.5 * s, legColor, 1.0);
  gfx.beginPath();
  gfx.moveTo(sx - bodyW / 2, bodyY + bodyH * 0.18);
  gfx.lineTo(sx - bodyW / 2 - armLen * 0.75 + aSwing, bodyY + bodyH * 0.62 + aSwing * 0.4);
  gfx.strokePath();
  gfx.beginPath();
  gfx.moveTo(sx + bodyW / 2, bodyY + bodyH * 0.18);
  gfx.lineTo(sx + bodyW / 2 + armLen * 0.75 - aSwing, bodyY + bodyH * 0.62 - aSwing * 0.4);
  gfx.strokePath();

  /* Neck */
  gfx.fillStyle(0xF5CBA7, 1.0);
  gfx.fillRect(sx - 1.5 * s, bodyY - 3 * s, 3 * s, 4 * s);

  /* Head */
  var skinColor = (lifeStage === 'elder') ? 0xE8B898 : 0xF5CBA7;
  gfx.fillStyle(skinColor, 1.0);
  gfx.fillCircle(sx, headY + headR, headR);

  /* Hair */
  var hairColor = TOWN._darkenInt(color, 28);
  if (lifeStage === 'elder') hairColor = 0xCCCCCC;
  if (lifeStage === 'child') hairColor = TOWN._darkenInt(color, 10);
  gfx.fillStyle(hairColor, 1.0);
  gfx.fillRect(sx - headR * 0.85, headY, headR * 1.7, headR * 0.55);
  gfx.fillCircle(sx, headY + headR * 0.28, headR * 0.87);

  /* Eyes */
  gfx.fillStyle(0x1A1A1A, 1.0);
  gfx.fillCircle(sx - headR * 0.30, headY + headR * 0.88, 1.7 * s);
  gfx.fillCircle(sx + headR * 0.30, headY + headR * 0.88, 1.7 * s);
  /* Shine */
  gfx.fillStyle(0xFFFFFF, 0.8);
  gfx.fillCircle(sx - headR * 0.22, headY + headR * 0.78, 0.8 * s);
  gfx.fillCircle(sx + headR * 0.38, headY + headR * 0.78, 0.8 * s);

  /* Expression */
  gfx.lineStyle(1.4 * s, 0x8B5E3C, 0.85);
  gfx.beginPath();
  if (moodValence > 0.12) {
    gfx.arc(sx, headY + headR * 1.05, headR * 0.28, 0.25, Math.PI - 0.25);
  } else if (moodValence < -0.12) {
    gfx.arc(sx, headY + headR * 1.28, headR * 0.28, Math.PI + 0.25, -0.25);
  } else {
    gfx.moveTo(sx - headR * 0.22, headY + headR * 1.12);
    gfx.lineTo(sx + headR * 0.22, headY + headR * 1.12);
  }
  gfx.strokePath();

  /* Life stage ring */
  if (lifeStage === 'child') {
    gfx.lineStyle(2, 0xFFD93D, 0.75);
    gfx.strokeCircle(sx, headY + headR, headR + 3);
  } else if (lifeStage === 'infant') {
    gfx.lineStyle(2, 0xFF6F91, 0.75);
    gfx.strokeCircle(sx, headY + headR, headR + 3);
  } else if (lifeStage === 'elder') {
    gfx.lineStyle(2, 0xCCCCCC, 0.55);
    gfx.strokeCircle(sx, headY + headR, headR + 3);
  }
};

/* ── Radius / color helpers ─────────────────────────────────────── */
TOWN.getAgentRadius = function(lifeStage) {
  switch (lifeStage) {
    case 'infant': return 9;
    case 'child':  return 14;
    case 'elder':  return 20;
    default:       return 22;
  }
};

/* ── Create sprite ───────────────────────────────────────────────── */
TOWN.createAgentSprite = function(scene, agent) {
  var name   = agent.name;
  var color  = TOWN.getAgentColor(name);
  var radius = TOWN.getAgentRadius(agent.life_stage);
  var pos    = TOWN.getPlaceCenter(agent.location || 'Main_Street');
  var off    = TOWN.getAgentOffset(name, agent.location);
  var x = pos.x + off.dx, y = pos.y + off.dy;

  /* Graphics */
  var gfx = scene.add.graphics().setDepth(11);
  TOWN.drawPerson(gfx, x, y, color, agent.life_stage,
                  agent.alive !== false, false, 0, agent.mood_valence || 0);

  /* Hitzone centred on the head */
  var headOffY = radius * 2 + 14;
  var hitzone  = scene.add.circle(x, y - headOffY, radius + 8, 0x000000, 0)
    .setInteractive({ useHandCursor: true }).setDepth(12);
  hitzone.setData('agentName', name);

  /* Name label */
  var nameText = scene.add.text(x, y - headOffY - radius - 4,
    name.replace(/_/g, ' '), {
      fontSize: '12px',
      fontFamily: 'Fredoka, sans-serif',
      fontStyle: 'bold',
      color: '#FFFFFF',
      align: 'center',
      stroke: '#3D2B1F',
      strokeThickness: 3,
    }).setOrigin(0.5, 1).setDepth(13);

  /* Drive bars */
  var driveBarContainer = scene.add.graphics().setDepth(12);

  /* Selection ring */
  var selRing = scene.add.graphics().setDepth(10);
  selRing.setPosition(x, y - headOffY);
  selRing.setAlpha(0);

  scene.agentLayer.add(gfx);
  scene.agentLayer.add(hitzone);
  scene.agentLayer.add(nameText);
  scene.agentLayer.add(driveBarContainer);
  scene.agentLayer.add(selRing);

  var sprite = {
    gfx: gfx,
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
    _walkPhase: Math.random(),
    _walking: false,
  };
  TOWN.state.agentSprites[name] = sprite;

  /* Pop-in */
  gfx.setAlpha(0);
  nameText.setAlpha(0);
  scene.tweens.add({ targets: gfx,  alpha: 1, duration: 450, ease: 'Back.easeOut' });
  scene.tweens.add({ targets: [nameText, driveBarContainer], alpha: 1, duration: 350, delay: 200 });

  return sprite;
};

/* ── Drive bars ─────────────────────────────────────────────────── */
TOWN.drawDriveBars = function(gfx, x, y, bottomOffset, drives) {
  gfx.clear();
  if (!drives) return;
  var keys = Object.keys(drives);
  var barW = 38, barH = 3, gap = 2;
  var startX = x - barW / 2;
  var startY = y + bottomOffset;
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    var val = Math.max(0, Math.min(100, drives[key]));
    var barY = startY + i * (barH + gap);
    var c = Phaser.Display.Color.HexStringToColor(TOWN.DRIVE_COLORS[key] || '#888888').color;
    gfx.fillStyle(0x3D2B1F, 0.2);
    gfx.fillRoundedRect(startX, barY, barW, barH, 1);
    if (val > 0) {
      gfx.fillStyle(c, 0.85);
      gfx.fillRoundedRect(startX, barY, barW * (val / 100), barH, 1);
    }
  }
};

/* ── Update sprite ───────────────────────────────────────────────── */
TOWN.updateAgentSprite = function(scene, agent) {
  var name = agent.name;
  var sprite = TOWN.state.agentSprites[name];
  if (!sprite) sprite = TOWN.createAgentSprite(scene, agent);

  var radius = TOWN.getAgentRadius(agent.life_stage);
  var color  = TOWN.getAgentColor(name);
  sprite.radius = radius;
  sprite.color  = color;

  var headOffY = radius * 2 + 14;

  /* Target iso position */
  var wx, wy;
  if (agent.world_pos && agent.world_pos[0] !== 0) {
    wx = agent.world_pos[0];
    wy = agent.world_pos[1];
  } else {
    var place = TOWN.state.town.places[agent.location];
    if (place) {
      wx = place.x + place.w / 2;
      wy = place.y + place.h / 2;
    } else {
      var pos_ = TOWN.getPlaceCenter(agent.location || 'Main_Street');
      wx = pos_.x;
      wy = pos_.y;
    }
  }
  var off = TOWN.getAgentOffset(name, agent.location);
  var tx = wx + off.dx, ty = wy + off.dy;

  var moving = Math.abs(sprite.x - tx) > 3 || Math.abs(sprite.y - ty) > 3;

  if (moving) {
    /* Animate movement + walk cycle */
    var fromX = sprite.x, fromY = sprite.y;
    sprite.x = tx; sprite.y = ty;
    sprite._walking = true;

    var startPhase = sprite._walkPhase;
    var dur = 900;
    var tweenObj = { t: 0 };
    var _sprite = sprite, _color = color, _agent = agent;
    scene.tweens.add({
      targets: tweenObj,
      t: 1,
      duration: dur,
      ease: 'Quad.easeInOut',
      onUpdate: function() {
        var cx = fromX + (tx - fromX) * tweenObj.t;
        var cy = fromY + (ty - fromY) * tweenObj.t;
        _sprite._walkPhase = (startPhase + tweenObj.t * 1.8) % 1.0;
        TOWN.drawPerson(_sprite.gfx, cx, cy, _color, _agent.life_stage,
                        _agent.alive !== false, true, _sprite._walkPhase, _agent.mood_valence || 0);
        TOWN.drawDriveBars(_sprite.driveBarContainer, cx, cy, 6, _agent.drives);
        _sprite.hitzone.setPosition(cx, cy - headOffY);
        _sprite.nameText.setPosition(cx, cy - headOffY - radius - 4);
        _sprite.selRing.setPosition(cx, cy - headOffY);
      },
      onComplete: function() {
        _sprite._walking = false;
        TOWN.drawPerson(_sprite.gfx, tx, ty, _color, _agent.life_stage,
                        _agent.alive !== false, false, _sprite._walkPhase, _agent.mood_valence || 0);
        TOWN.drawDriveBars(_sprite.driveBarContainer, tx, ty, 6, _agent.drives);
      },
    });
    scene.tweens.add({ targets: sprite.hitzone, x: tx, y: ty - headOffY, duration: dur, ease: 'Quad.easeInOut' });
    scene.tweens.add({ targets: sprite.nameText, x: tx, y: ty - headOffY - radius - 4, duration: dur, ease: 'Quad.easeInOut' });
    scene.tweens.add({ targets: sprite.selRing,  x: tx, y: ty - headOffY, duration: dur, ease: 'Quad.easeInOut' });
  } else {
    /* Idle — redraw in place */
    TOWN.drawPerson(sprite.gfx, sprite.x, sprite.y, color, agent.life_stage,
                    agent.alive !== false, false, sprite._walkPhase, agent.mood_valence || 0);
    TOWN.drawDriveBars(sprite.driveBarContainer, sprite.x, sprite.y, 6, agent.drives);
    /* Walking wobble from server-side moving flag */
    if (agent.moving) {
      var wobblePhase = (Date.now() / 150) % (Math.PI * 2);
      sprite.gfx.rotation = Math.sin(wobblePhase) * 0.08;
      sprite.gfx.y += Math.sin(wobblePhase * 2) * 2;
    }
  }

  /* Dead */
  if (!agent.alive && sprite.gfx.alpha > 0.3) {
    if (sprite._breatheTween) { sprite._breatheTween.stop(); sprite._breatheTween = null; }
    scene.tweens.add({ targets: [sprite.gfx, sprite.nameText], alpha: 0.18, y: '-=22', duration: 2400, ease: 'Sine.easeInOut' });
  }

  /* Selection ring */
  if (TOWN.state.selectedAgent === name) {
    TOWN.showSelectionRing(scene, sprite);
  } else if (sprite.selRing.alpha > 0) {
    scene.tweens.killTweensOf(sprite.selRing);
    sprite.selRing.setAlpha(0);
  }
};

/* ── Selection ring ─────────────────────────────────────────────── */
TOWN.showSelectionRing = function(scene, sprite) {
  if (sprite.selRing.alpha > 0) return;
  sprite.selRing.clear();
  sprite.selRing.lineStyle(3, 0xFFD93D, 0.8);
  sprite.selRing.strokeCircle(0, 0, sprite.radius + 10);
  scene.tweens.add({
    targets: sprite.selRing,
    alpha: { from: 0.3, to: 0.85 },
    duration: 800, yoyo: true, repeat: -1,
    ease: 'Sine.easeInOut',
  });
};

/* ── Talk bounce ────────────────────────────────────────────────── */
TOWN.startTalkBounce = function(scene, agentName) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite) return;
  sprite.talking = true;
};

TOWN.stopTalkBounce = function(scene, agentName) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite) return;
  sprite.talking = false;
};

/* ── Relationship overlay ────────────────────────────────────────── */
TOWN._REL_COLORS = {
  married:      0xFFD700,
  romantic:     0xFF69B4,
  close_friend: 0x6BCB77,
  friend:       0x87CEEB,
  acquaintance: 0xBBBBBB,
};

TOWN.drawRelationshipOverlay = function(scene) {
  var gfx = TOWN._relOverlayGfx;
  if (!gfx) return;
  gfx.clear();
  if (!TOWN.state.showRelOverlay) return;
  var agents  = TOWN.state.agents;
  var sprites = TOWN.state.agentSprites;
  var drawn   = {};
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
      if (!spB || !agents[nameB] || !agents[nameB].alive) continue;
      var lc = TOWN._REL_COLORS[rel.state] || 0xAAAAAA;
      var la = rel.state === 'acquaintance' ? 0.2 : 0.45;
      var lw = (rel.state === 'married' || rel.state === 'romantic') ? 2.5 : 1.5;
      gfx.lineStyle(lw, lc, la);
      gfx.beginPath();
      gfx.moveTo(spA.x, spA.y); gfx.lineTo(spB.x, spB.y);
      gfx.strokePath();
    }
  }
};

/* ── Highlight flash ─────────────────────────────────────────────── */
TOWN.highlightAgent = function(scene, name) {
  var sp = TOWN.state.agentSprites[name];
  if (!sp) return;
  scene.tweens.add({ targets: sp.gfx, alpha: 0.4, duration: 80, yoyo: true });
};
