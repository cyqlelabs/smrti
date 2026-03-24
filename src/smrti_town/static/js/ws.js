/**
 * WebSocket connection with auto-reconnect.
 */

var WS = {
  /** @type {WebSocket|null} */
  socket: null,
  _reconnectDelay: 1000,
  _maxReconnectDelay: 15000,
  _reconnectTimer: null,

  /**
   * Connect to the server WebSocket.
   */
  connect: function() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = protocol + '//' + window.location.host + '/ws';

    try {
      this.socket = new WebSocket(url);
    } catch (e) {
      console.error('WebSocket creation failed:', e);
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = function() {
      console.log('WS connected');
      GameState.connected = true;
      WS._reconnectDelay = 1000;

      // Hide generating overlay if it was shown
      var genEl = document.getElementById('ui-generating');
      if (genEl && !genEl.classList.contains('hidden')) {
        // Keep it shown until we receive state data
      }
    };

    this.socket.onmessage = function(event) {
      try {
        var msg = JSON.parse(event.data);
        TickProcessor.handle(msg);

        // Hide generating on first state/tick
        if (msg.type === 'state' || msg.type === 'tick') {
          var genEl = document.getElementById('ui-generating');
          if (genEl) genEl.classList.add('hidden');
        }
      } catch (e) {
        console.error('WS parse error:', e, event.data);
      }
    };

    this.socket.onclose = function(event) {
      console.log('WS closed:', event.code, event.reason);
      GameState.connected = false;
      WS._scheduleReconnect();
    };

    this.socket.onerror = function(err) {
      console.error('WS error:', err);
      GameState.connected = false;
    };
  },

  /**
   * Send a JSON message to the server.
   * @param {object} data
   */
  send: function(data) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    } else {
      console.warn('WS not connected, cannot send:', data);
    }
  },

  _scheduleReconnect: function() {
    if (this._reconnectTimer) return;

    var delay = this._reconnectDelay;
    console.log('WS reconnecting in', delay, 'ms');

    this._reconnectTimer = setTimeout(function() {
      WS._reconnectTimer = null;
      WS.connect();
    }, delay);

    // Exponential backoff
    this._reconnectDelay = Math.min(this._reconnectDelay * 1.5, this._maxReconnectDelay);
  },

  disconnect: function() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    GameState.connected = false;
  },
};
