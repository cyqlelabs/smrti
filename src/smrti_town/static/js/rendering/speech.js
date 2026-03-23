/* ================================================================
   speech.js — showSpeechBubble(), hideSpeechBubble()
   ================================================================ */
window.TOWN = window.TOWN || {};

/* ── Bubble pool helpers ───────────────────────────────────────────── */
TOWN._getBubble = function(scene) {
  var pool = TOWN.state.bubblePool;
  for (var i = 0; i < pool.length; i++) {
    if (!pool[i].inUse) {
      pool[i].inUse = true;
      pool[i].bubble.setVisible(true);
      pool[i].txt.setVisible(true);
      return pool[i];
    }
  }
  var entry = {
    bubble: scene.add.graphics().setDepth(21).setVisible(false),
    txt: scene.add.text(0, 0, '', {
      fontSize: '14px', fontFamily: 'Nunito, sans-serif',
      color: '#3D2B1F', wordWrap: { width: 206 }, lineSpacing: 2,
    }).setDepth(22).setVisible(false),
    inUse: true,
  };
  scene.speechLayer.add(entry.bubble);
  scene.speechLayer.add(entry.txt);
  pool.push(entry);
  return entry;
};

TOWN._returnBubble = function(entry) {
  if (!entry) return;
  entry.bubble.setAlpha(0).setVisible(false);
  entry.txt.setAlpha(0).setVisible(false);
  entry.inUse = false;
};

/* ── showSpeechBubble ─────────────────────────────────────────────── */
TOWN.showSpeechBubble = function(scene, agentName, text) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite) return;

  /* Dismiss existing bubble first */
  TOWN.hideSpeechBubble(scene, agentName);

  var maxW    = 230;
  var padding = 12;
  var tailH   = 10;
  var displayText = text.length > 120 ? text.substring(0, 117) + '...' : text;

  /* Grab pooled objects */
  var entry  = TOWN._getBubble(scene);
  var bubble = entry.bubble;
  var txt    = entry.txt;

  /* Update text content + measure */
  txt.setText(displayText).setWordWrapWidth(maxW - padding * 2);
  var tw = txt.width + padding * 2;
  var th = txt.height + padding * 2;

  /* Position above agent */
  var bx = sprite.x - tw / 2;
  var by = sprite.y - sprite.radius - 20 - th - tailH;
  if (bx < 10) bx = 10;
  if (bx + tw > 990) bx = 990 - tw;
  if (by < 10) by = 10;

  /* Redraw bubble graphics */
  bubble.clear();
  bubble.fillStyle(0xFFF8F0, 0.97);
  bubble.fillRoundedRect(bx, by, tw, th, 12);
  bubble.lineStyle(2, 0xD4A03C, 0.8);
  bubble.strokeRoundedRect(bx, by, tw, th, 12);
  var tailX = Math.max(bx + 15, Math.min(sprite.x, bx + tw - 15));
  bubble.fillStyle(0xFFF8F0, 0.97);
  bubble.fillTriangle(tailX - 7, by + th, tailX + 7, by + th, tailX, by + th + tailH);
  bubble.lineStyle(2, 0xD4A03C, 0.8);
  bubble.lineBetween(tailX - 7, by + th, tailX, by + th + tailH);
  bubble.lineBetween(tailX + 7, by + th, tailX, by + th + tailH);

  txt.setPosition(bx + padding, by + padding);

  /* Entrance animation */
  bubble.setAlpha(0).setScale(0.8);
  txt.setAlpha(0).setScale(0.8);
  scene.tweens.add({
    targets: [bubble, txt],
    alpha: 1, scaleX: 1, scaleY: 1,
    duration: 250, ease: 'Back.easeOut',
  });

  sprite.speechBubble = entry;
  TOWN.startTalkBounce(scene, agentName);

  sprite._speechTimer = scene.time.delayedCall(5000, function() {
    TOWN.hideSpeechBubble(scene, agentName);
  });
};

/* ── hideSpeechBubble ─────────────────────────────────────────────── */
TOWN.hideSpeechBubble = function(scene, agentName) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite || !sprite.speechBubble) return;

  var entry = sprite.speechBubble;
  sprite.speechBubble = null;

  scene.tweens.add({
    targets: [entry.bubble, entry.txt],
    alpha: 0,
    duration: 300,
    ease: 'Sine.easeIn',
    onComplete: function() { TOWN._returnBubble(entry); },
  });

  if (sprite._speechTimer) {
    sprite._speechTimer.remove();
    sprite._speechTimer = null;
  }

  TOWN.stopTalkBounce(scene, agentName);
};
