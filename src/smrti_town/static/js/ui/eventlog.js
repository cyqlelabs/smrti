/* ================================================================
   eventlog.js — addLogEntry(), log management
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.addLogEntry = function(type, text) {
  var log = document.getElementById('event-log');
  var entry = document.createElement('div');
  entry.className = 'log-entry ' + (TOWN.LOG_COLORS[type] || 'log-system');

  var c = TOWN.state.calendar;
  var h = Math.floor(c.hour || 6);
  var m = String(Math.floor(((c.hour || 6) % 1) * 60)).padStart(2, '0');
  var timeStr = String(h).padStart(2, '0') + ':' + m;

  entry.innerHTML = '<span class="log-time">' + timeStr + '</span>' + TOWN.escapeHtml(text);
  log.appendChild(entry);

  /* Keep max 80 entries */
  while (log.children.length > 80) {
    log.removeChild(log.firstChild);
  }
  log.scrollTop = log.scrollHeight;

  TOWN.state.eventLog.push({ type: type, text: text, time: timeStr });
};
