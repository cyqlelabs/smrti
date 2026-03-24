/**
 * Economy dashboard panel.
 */

var Economy = {
  el: null,
  contentEl: null,

  init: function() {
    this.el = document.getElementById('ui-economy');
    this.contentEl = document.getElementById('economy-content');
    document.getElementById('economy-close').addEventListener('click', function() {
      Economy.hide();
    });
  },

  show: function() {
    this.el.classList.remove('hidden');
    this.render();
  },

  hide: function() {
    this.el.classList.add('hidden');
  },

  toggle: function() {
    if (this.el.classList.contains('hidden')) {
      this.show();
    } else {
      this.hide();
    }
  },

  render: function() {
    var eco = GameState.economy || {};
    var html = '';

    html += this._row('Treasury', (eco.treasury || 0) + 'g', '');
    html += this._row('Income', '+' + (eco.income || 0) + 'g/tick', 'positive');
    html += this._row('Expenses', '-' + (eco.expenses || 0) + 'g/tick', 'negative');

    // Tax rates
    var taxes = eco.tax_rates || {};
    var taxKeys = Object.keys(taxes);
    if (taxKeys.length > 0) {
      html += '<div style="margin-top:10px;">';
      html += '<div class="sidebar-label">Tax Rates</div>';
      for (var i = 0; i < taxKeys.length; i++) {
        var rate = taxes[taxKeys[i]];
        var pct = typeof rate === 'number' ? Math.round(rate * 100) + '%' : rate;
        html += this._row(taxKeys[i], pct, '');
      }
      html += '</div>';
    }

    this.contentEl.innerHTML = html;
  },

  _row: function(label, value, cls) {
    return '<div class="economy-row">' +
      '<span class="label">' + _esc(label) + '</span>' +
      '<span class="value ' + (cls || '') + '">' + _esc(String(value)) + '</span>' +
      '</div>';
  },
};
