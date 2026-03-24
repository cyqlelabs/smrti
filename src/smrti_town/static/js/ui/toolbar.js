/**
 * Left-side building palette toolbar.
 */

var Toolbar = {
  el: null,
  itemsEl: null,

  init: function() {
    this.el = document.getElementById('ui-toolbar');
    this.itemsEl = document.getElementById('toolbar-items');
  },

  show: function() {
    this.el.classList.remove('hidden');
    this.render();
  },

  hide: function() {
    this.el.classList.add('hidden');
  },

  render: function() {
    var html = '';
    var population = 0;
    var agents = GameState.agents || [];
    for (var i = 0; i < agents.length; i++) {
      if (agents[i].alive !== false) population++;
    }

    // Group by category
    var categories = {};
    for (var key in BUILDINGS) {
      var b = BUILDINGS[key];
      if (!categories[b.category]) categories[b.category] = [];
      categories[b.category].push({ key: key, def: b });
    }

    for (var c = 0; c < CATEGORY_ORDER.length; c++) {
      var cat = CATEGORY_ORDER[c];
      var items = categories[cat];
      if (!items || items.length === 0) continue;

      html += '<div class="toolbar-category">';
      html += '<div class="toolbar-category-title" style="color:' +
        (CATEGORY_COLORS[cat] || 'var(--text-dim)') + ';">' + _esc(cat) + '</div>';

      for (var j = 0; j < items.length; j++) {
        var item = items[j];
        var locked = population < item.def.minPop;
        var selected = GameState.selectedBuilding === item.key;
        var cls = 'toolbar-item';
        if (locked) cls += ' locked';
        if (selected) cls += ' selected';

        html += '<div class="' + cls + '" data-building="' + item.key + '">';
        html += '<span class="toolbar-item-name">' + _esc(item.def.name) + '</span>';
        html += '<span class="toolbar-item-cost">' + item.def.cost + 'g</span>';
        html += '</div>';
      }

      html += '</div>';
    }

    this.itemsEl.innerHTML = html;

    // Attach click handlers
    var toolbarItems = this.itemsEl.querySelectorAll('.toolbar-item:not(.locked)');
    for (var t = 0; t < toolbarItems.length; t++) {
      toolbarItems[t].addEventListener('click', function(e) {
        var bKey = this.getAttribute('data-building');
        if (GameState.selectedBuilding === bKey) {
          GameState.selectedBuilding = null;
        } else {
          GameState.selectedBuilding = bKey;
        }
        Toolbar.render();
      });
    }
  },

  /** Cancel placement mode. */
  cancelPlacement: function() {
    GameState.selectedBuilding = null;
    this.render();
  },
};
