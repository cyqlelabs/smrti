/**
 * Right-side inspector sidebar — shows agent or building details.
 */

var Sidebar = {
  el: null,
  contentEl: null,

  init: function() {
    this.el = document.getElementById('ui-sidebar');
    this.contentEl = document.getElementById('sidebar-content');
    document.getElementById('sidebar-close').addEventListener('click', function() {
      Sidebar.hide();
    });
  },

  show: function() {
    this.el.classList.remove('hidden');
  },

  hide: function() {
    this.el.classList.add('hidden');
    GameState.selectedAgent = null;
    GameState.selectedPlace = null;
  },

  /**
   * Display agent details in the sidebar.
   */
  showAgent: function(agent) {
    this.show();
    var html = '<h3>' + _esc(agent.name) + '</h3>';

    // Life info
    html += '<div class="sidebar-section">';
    html += '<div class="sidebar-label">Status</div>';
    html += '<div class="sidebar-value">' +
      _esc(agent.life_stage || 'adult') + ', age ' + (agent.age_years || '?') +
      '</div>';
    if (agent.location) {
      html += '<div class="sidebar-value" style="color:var(--text-dim);font-size:11px;">at ' +
        _esc(agent.location) + '</div>';
    }
    if (agent.council_role) {
      html += '<div class="sidebar-value" style="color:var(--accent);font-size:11px;">' +
        _esc(agent.council_role) + '</div>';
    }
    html += '</div>';

    // Wallet
    if (agent.wallet !== undefined) {
      html += '<div class="sidebar-section">';
      html += '<div class="sidebar-label">Wallet</div>';
      html += '<div class="sidebar-value" style="color:var(--gold);">' + agent.wallet + 'g</div>';
      html += '</div>';
    }

    // Needs / drives
    var needs = agent.needs || agent.drives || {};
    var needKeys = Object.keys(needs);
    if (needKeys.length > 0) {
      html += '<div class="sidebar-section">';
      html += '<div class="sidebar-label">Needs</div>';
      for (var i = 0; i < needKeys.length; i++) {
        var key = needKeys[i];
        var val = needs[key];
        if (typeof val !== 'number') continue;
        var pct = Math.min(100, Math.max(0, Math.round(val)));
        var color = pct > 70 ? 'var(--danger)' : pct > 40 ? 'var(--warning)' : 'var(--success)';
        html += '<div class="need-bar">' +
          '<span class="need-bar-label">' + _esc(key) + '</span>' +
          '<div class="need-bar-track"><div class="need-bar-fill" style="width:' +
          pct + '%;background:' + color + ';"></div></div>' +
          '<span class="need-bar-value">' + pct + '</span>' +
          '</div>';
      }
      html += '</div>';
    }

    // Traits
    if (agent.traits) {
      var traitKeys = Object.keys(agent.traits);
      if (traitKeys.length > 0) {
        html += '<div class="sidebar-section">';
        html += '<div class="sidebar-label">Traits</div>';
        for (var t = 0; t < traitKeys.length; t++) {
          var tk = traitKeys[t];
          var tv = agent.traits[tk];
          if (typeof tv !== 'number') continue;
          var tpct = Math.round(tv * 100);
          html += '<div class="need-bar">' +
            '<span class="need-bar-label">' + _esc(tk) + '</span>' +
            '<div class="need-bar-track"><div class="need-bar-fill" style="width:' +
            tpct + '%;background:var(--accent);"></div></div>' +
            '<span class="need-bar-value">' + tpct + '</span>' +
            '</div>';
        }
        html += '</div>';
      }
    }

    this.contentEl.innerHTML = html;
  },

  /**
   * Display building details in the sidebar.
   */
  showBuilding: function(building) {
    this.show();
    var bKey = building.building_key || building.type || 'unknown';
    var def = BUILDINGS[bKey];
    var name = (def && def.name) || bKey;

    var html = '<h3>' + _esc(name) + '</h3>';

    html += '<div class="sidebar-section">';
    html += '<div class="sidebar-label">Type</div>';
    html += '<div class="sidebar-value">' + _esc(bKey) + '</div>';
    html += '</div>';

    if (def) {
      html += '<div class="sidebar-section">';
      html += '<div class="sidebar-label">Category</div>';
      html += '<div class="sidebar-value" style="color:' +
        (CATEGORY_COLORS[def.category] || 'var(--text)') + ';">' +
        _esc(def.category) + '</div>';
      html += '</div>';
    }

    html += '<div class="sidebar-section">';
    html += '<div class="sidebar-label">Position</div>';
    html += '<div class="sidebar-value">(' + building.grid_x + ', ' + building.grid_y + ')</div>';
    html += '</div>';

    // List occupants from GameState
    if (building.occupants && building.occupants.length > 0) {
      html += '<div class="sidebar-section">';
      html += '<div class="sidebar-label">Occupants</div>';
      for (var i = 0; i < building.occupants.length; i++) {
        html += '<div class="sidebar-value">' + _esc(building.occupants[i]) + '</div>';
      }
      html += '</div>';
    }

    this.contentEl.innerHTML = html;
  },
};
