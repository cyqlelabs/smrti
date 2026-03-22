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

TOWN.computeTownHealth = function() {
  var agents = TOWN.state.agents;
  var alive = 0, totalMood = 0, totalEnergy = 0;
  for (var n in agents) {
    var a = agents[n];
    if (!a.alive) continue;
    alive++;
    totalMood   += (a.mood_valence || 0);
    totalEnergy += (a.drives && a.drives.energy !== undefined ? a.drives.energy : 50);
  }
  if (alive === 0) return 50;
  var moodScore   = ((totalMood / alive) + 1) / 2 * 100;
  var popScore    = Math.min(alive / 6, 1) * 100;
  var energyScore = totalEnergy / alive;
  return Math.round(moodScore * 0.4 + popScore * 0.3 + energyScore * 0.3);
};

TOWN.updateTownHealth = function() {
  var score = TOWN.computeTownHealth();
  TOWN.state.townHealth = score;
  var el = document.getElementById('town-health');
  if (!el) return;
  var icon  = score >= 70 ? '\uD83D\uDC9A' : score >= 40 ? '\uD83D\uDC9B' : '\u2764\uFE0F';
  el.textContent = icon + ' ' + score;
  el.style.color = score >= 70 ? '#6BCB77' : score >= 40 ? '#FFD93D' : '#FF6B6B';
};
