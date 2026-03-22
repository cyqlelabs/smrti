/**
 * Economy UI — wallet display and transaction notifications.
 */
(function() {
  'use strict';

  TOWN.updateEconomyDisplay = function() {
    var container = document.getElementById('economy-display');
    if (!container) return;

    var selected = TOWN.state.selectedAgent;
    if (!selected) {
      container.style.display = 'none';
      return;
    }

    var agent = TOWN.state.agents[selected];
    if (!agent) {
      container.style.display = 'none';
      return;
    }

    container.style.display = 'block';
    var wallet = agent.wallet !== undefined ? agent.wallet : '?';
    container.innerHTML = '<span class="wallet-icon">\uD83D\uDCB0</span> <span class="wallet-amount">' + wallet + '</span>';
  };

})();
