/* ================================================================
   speech.js — showSpeechBubble(), hideSpeechBubble()
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.showSpeechBubble = function(scene, agentName, text) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite) return;

  /* Remove existing bubble */
  TOWN.hideSpeechBubble(scene, agentName);

  var maxW = 230;
  var padding = 12;
  var tailH = 10;

  /* Truncate long text */
  var displayText = text.length > 120 ? text.substring(0, 117) + '...' : text;

  /* Text object */
  var txt = scene.add.text(0, 0, displayText, {
    fontSize: '14px',
    fontFamily: 'Nunito, sans-serif',
    color: '#3D2B1F',
    wordWrap: { width: maxW - padding * 2 },
    lineSpacing: 2,
  });

  var tw = txt.width + padding * 2;
  var th = txt.height + padding * 2;

  /* Position above agent */
  var bx = sprite.x - tw / 2;
  var by = sprite.y - sprite.radius - 20 - th - tailH;

  /* Clamp to game bounds */
  if (bx < 10) bx = 10;
  if (bx + tw > 990) bx = 990 - tw;
  if (by < 10) by = 10;

  /* Bubble background */
  var bubble = scene.add.graphics().setDepth(21);

  /* Cream background with warm border */
  bubble.fillStyle(0xFFF8F0, 0.97);
  bubble.fillRoundedRect(bx, by, tw, th, 12);

  /* Border */
  bubble.lineStyle(2, 0xD4A03C, 0.8);
  bubble.strokeRoundedRect(bx, by, tw, th, 12);

  /* Tail triangle pointing down to speaker */
  var tailX = Math.max(bx + 15, Math.min(sprite.x, bx + tw - 15));
  bubble.fillStyle(0xFFF8F0, 0.97);
  bubble.fillTriangle(
    tailX - 7, by + th,
    tailX + 7, by + th,
    tailX, by + th + tailH
  );
  /* Tail border lines */
  bubble.lineStyle(2, 0xD4A03C, 0.8);
  bubble.lineBetween(tailX - 7, by + th, tailX, by + th + tailH);
  bubble.lineBetween(tailX + 7, by + th, tailX, by + th + tailH);

  /* Position text inside bubble */
  txt.setPosition(bx + padding, by + padding);
  txt.setDepth(22);

  scene.speechLayer.add(bubble);
  scene.speechLayer.add(txt);

  /* Entrance animation: fade + scale-up */
  bubble.setAlpha(0);
  bubble.setScale(0.8);
  txt.setAlpha(0);
  txt.setScale(0.8);

  scene.tweens.add({
    targets: [bubble, txt],
    alpha: 1,
    scaleX: 1, scaleY: 1,
    duration: 250,
    ease: 'Back.easeOut',
  });

  sprite.speechBubble = { bubble: bubble, txt: txt };

  /* Talk bounce while bubble is shown */
  TOWN.startTalkBounce(scene, agentName);

  /* Auto-dismiss after 5 seconds */
  sprite._speechTimer = scene.time.delayedCall(5000, function() {
    TOWN.hideSpeechBubble(scene, agentName);
  });
};

TOWN.hideSpeechBubble = function(scene, agentName) {
  var sprite = TOWN.state.agentSprites[agentName];
  if (!sprite || !sprite.speechBubble) return;

  var bubbleRef = sprite.speechBubble.bubble;
  var txtRef = sprite.speechBubble.txt;

  scene.tweens.add({
    targets: [bubbleRef, txtRef],
    alpha: 0,
    duration: 300,
    ease: 'Sine.easeIn',
    onComplete: function() {
      bubbleRef.destroy();
      txtRef.destroy();
    },
  });

  sprite.speechBubble = null;

  if (sprite._speechTimer) {
    sprite._speechTimer.remove();
    sprite._speechTimer = null;
  }

  TOWN.stopTalkBounce(scene, agentName);
};
