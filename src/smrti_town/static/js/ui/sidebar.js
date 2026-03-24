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
   * Display building details in the sidebar with live stats.
   */
  showBuilding: function(building) {
    this.show();
    var bKey = building.building_key || building.type || 'unknown';
    var def = BUILDINGS[bKey];
    var name = (def && def.name) || bKey;
    var category = building.category || (def && def.category) || '';
    var catColor = CATEGORY_COLORS[category] || 'var(--text)';

    var html = '<h3>' + _esc(name) + '</h3>';

    // Category badge
    if (category) {
      html += '<div style="margin-bottom:8px;">';
      html += '<span style="background:' + catColor + '22;color:' + catColor +
        ';border:1px solid ' + catColor + '44;border-radius:4px;padding:2px 7px;font-size:11px;">' +
        _esc(category) + '</span>';
      html += '</div>';
    }

    // Description
    var desc = building.description || (def && def.description) || '';
    if (desc) {
      html += '<div style="color:var(--text-dim);font-size:12px;margin-bottom:10px;line-height:1.4;">' +
        _esc(desc) + '</div>';
    }

    // Purpose tags
    var tags = [];
    if (building.provides_food) tags.push({ label: 'Food', color: '#f0a500' });
    if (building.provides_housing) tags.push({ label: 'Housing', color: '#58a6ff' });
    if (building.provides_goods) tags.push({ label: 'Goods', color: '#7ee787' });
    if (building.revenue_per_hour > 0) tags.push({ label: 'Revenue', color: '#bc8cff' });
    if (building.staff_required > 0) tags.push({ label: 'Employs ' + building.staff_required, color: '#d29922' });
    if (building.capacity > 0) tags.push({ label: 'Cap. ' + building.capacity, color: '#79c0ff' });
    if (tags.length > 0) {
      html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;">';
      for (var t = 0; t < tags.length; t++) {
        html += '<span style="background:' + tags[t].color + '22;color:' + tags[t].color +
          ';border:1px solid ' + tags[t].color + '44;border-radius:4px;padding:1px 6px;font-size:11px;">' +
          _esc(tags[t].label) + '</span>';
      }
      html += '</div>';
    }

    // Live stats
    html += '<div class="sidebar-section">';
    html += '<div class="sidebar-label">Live</div>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-top:4px;">';
    html += _statCell('Here now', building.citizens_here || 0, 'var(--accent)');
    if (building.provides_housing)
      html += _statCell('Residents', building.citizens_home || 0, '#58a6ff');
    if (building.staff_required > 0)
      html += _statCell('Workers', building.citizens_work || 0, '#d29922');
    if (building.provides_food || building.provides_goods) {
      html += _statCell('Transactions', building.transactions || 0, '#7ee787');
      html += _statCell('Revenue', (building.revenue || 0) + 'g', '#bc8cff');
    }
    html += '</div>';
    html += '</div>';

    // Economics
    html += '<div class="sidebar-section">';
    html += '<div class="sidebar-label">Economics</div>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-top:4px;">';
    if (building.cost) html += _statCell('Cost', building.cost + 'g', 'var(--text)');
    if (building.maintenance) html += _statCell('Upkeep/hr', building.maintenance + 'g', '#f85149');
    if (building.revenue_per_hour) html += _statCell('Rev/hr', building.revenue_per_hour + 'g', '#7ee787');
    html += '</div>';
    html += '</div>';

    this.contentEl.innerHTML = html;
  },
};
