/**
 * Bottom-left scrolling event log.
 */

var EventLog = {
  el: null,
  entriesEl: null,
  _maxEntries: 80,

  init: function() {
    this.el = document.getElementById('ui-eventlog');
    this.entriesEl = document.getElementById('eventlog-entries');
  },

  show: function() {
    this.el.classList.remove('hidden');
  },

  hide: function() {
    this.el.classList.add('hidden');
  },

  /**
   * Add an event entry.
   * @param {string} text
   * @param {string} [type='event'] - 'event', 'crisis', 'milestone', 'dialogue'
   */
  add: function(text, type) {
    var cls = 'event-entry';
    if (type === 'crisis') cls += ' event-crisis';
    else if (type === 'milestone') cls += ' event-milestone';
    else if (type === 'dialogue') cls += ' event-dialogue';

    var cal = GameState.calendar;
    var timeStr = 'D' + (cal.day || 1) + ' ' +
      (cal.hour < 10 ? '0' : '') + Math.floor(cal.hour || 0) + ':00';

    var entry = document.createElement('div');
    entry.className = cls;
    entry.innerHTML = '<span class="event-time">' + _esc(timeStr) + '</span>' + _esc(text);

    this.entriesEl.appendChild(entry);

    // Store in GameState for persistence
    GameState.events.push({ text: text, type: type || 'event', time: timeStr });
    if (GameState.events.length > this._maxEntries) {
      GameState.events.shift();
    }

    // Trim DOM
    while (this.entriesEl.children.length > this._maxEntries) {
      this.entriesEl.removeChild(this.entriesEl.firstChild);
    }

    // Auto-scroll to bottom
    this.el.scrollTop = this.el.scrollHeight;
  },

  clear: function() {
    this.entriesEl.innerHTML = '';
    GameState.events = [];
  },
};
