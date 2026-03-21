/* ================================================================
   ws.js — WebSocket connection, reconnection, send helper
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.connectWS = function() {
  TOWN.setWSStatus('connecting');

  var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var host = location.hostname || 'localhost';
  var port = location.port || '8420';
  var wsUrl = protocol + '//' + host + ':' + port + '/ws';

  var ws;
  try {
    ws = new WebSocket(wsUrl);
  } catch (e) {
    TOWN.setWSStatus('disconnected');
    setTimeout(TOWN.connectWS, 3000);
    return;
  }

  ws.onopen = function() {
    TOWN.state.ws = ws;
    TOWN.state.wsReady = true;
    TOWN.setWSStatus('connected');
    TOWN.addLogEntry('system', 'WebSocket connected');

    /* If demo mode was running, we keep it — server data will override */
  };

  ws.onmessage = function(evt) {
    try {
      var data = JSON.parse(evt.data);
      TOWN.state.tickQueue.push(data);
    } catch (e) {
      /* Ignore malformed messages */
    }
  };

  ws.onclose = function() {
    TOWN.state.ws = null;
    TOWN.state.wsReady = false;
    TOWN.setWSStatus('disconnected');
    TOWN.addLogEntry('system', 'Disconnected. Reconnecting in 3s\u2026');
    setTimeout(TOWN.connectWS, 3000);
  };

  ws.onerror = function() {
    ws.close();
  };
};

TOWN.wsSend = function(msg) {
  if (TOWN.state.ws && TOWN.state.wsReady) {
    TOWN.state.ws.send(JSON.stringify(msg));
  }
};

TOWN.setWSStatus = function(status) {
  var dot = document.getElementById('ws-dot');
  var label = document.getElementById('ws-label');
  if (!dot || !label) return;
  dot.className = 'ws-dot ws-' + status;
  label.textContent = status.charAt(0).toUpperCase() + status.slice(1);
};
