/* ================================================================
   topbar.js — updateClockUI(), updateDirectorBadge()
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.updateClockUI = function() {
  var c = TOWN.state.calendar;
  var h = (c.hour !== undefined ? c.hour : 6);
  var hh = String(Math.floor(h)).padStart(2, '0');
  var mm = String(Math.floor((h % 1) * 60)).padStart(2, '0');

  /* Time with sun/moon icon */
  var tod = c.time_of_day || 'morning';
  var icon = TOWN.TOD_ICONS[tod] || '\u2600\uFE0F';
  document.getElementById('sim-time').textContent = icon + ' ' + hh + ':' + mm;

  /* Day */
  document.getElementById('sim-day').textContent = 'Day ' + (c.day || 1);

  /* Season with color */
  var seasonEl = document.getElementById('sim-season');
  var season = c.season || 'spring';
  var seasonColor = TOWN.SEASON_COLORS[season] || '#5C9E5C';
  seasonEl.textContent = season.charAt(0).toUpperCase() + season.slice(1);
  seasonEl.style.color = seasonColor;

  /* Year */
  document.getElementById('sim-year').textContent = 'Year ' + (c.year || 1);

  /* Tick counter */
  document.getElementById('tick-counter').textContent = 'Tick ' + TOWN.state.tickNumber;
};

TOWN.updateDirectorBadge = function() {
  var badge = document.getElementById('director-badge');
  var mode = TOWN.state.directorMode || 'routine';
  badge.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
  badge.className = 'badge badge-' + mode;
};
