/**
 * Top bar UI — clock, treasury, population, speed.
 */

var Topbar = {
  el: null,
  clockEl: null,
  treasuryEl: null,
  popEl: null,
  speedEl: null,

  init: function() {
    this.el = document.getElementById('ui-topbar');
    this.clockEl = document.getElementById('topbar-clock');
    this.treasuryEl = document.getElementById('topbar-treasury');
    this.popEl = document.getElementById('topbar-population');
    this.speedEl = document.getElementById('topbar-speed');
  },

  show: function() {
    this.el.classList.remove('hidden');
  },

  hide: function() {
    this.el.classList.add('hidden');
  },

  update: function() {
    var cal = GameState.calendar;
    var h = cal.hour || 0;
    var hourStr = (h < 10 ? '0' : '') + Math.floor(h) + ':00';
    var seasonCap = (cal.season || 'spring');
    seasonCap = seasonCap.charAt(0).toUpperCase() + seasonCap.slice(1);

    this.clockEl.textContent = 'Day ' + (cal.day || 1) + ' ' + hourStr +
      ' | ' + seasonCap + ' Y' + (cal.year || 1);

    var treasury = GameState.economy.treasury || 0;
    this.treasuryEl.textContent = treasury + 'g';

    var alive = 0;
    var agents = GameState.agents || [];
    for (var i = 0; i < agents.length; i++) {
      if (agents[i].alive !== false) alive++;
    }
    this.popEl.textContent = alive + ' citizens';

    var mode = GameState.directorMode || 'routine';
    var pauseStr = GameState.paused ? ' [PAUSED]' : '';
    this.speedEl.textContent = mode + pauseStr;
  },
};
