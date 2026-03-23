/**
 * Petition list panel with approve/dismiss buttons.
 */

var Petitions = {
  el: null,
  listEl: null,

  init: function() {
    this.el = document.getElementById('ui-petitions');
    this.listEl = document.getElementById('petitions-list');
    document.getElementById('petitions-close').addEventListener('click', function() {
      Petitions.hide();
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
    var petitions = GameState.petitions || [];
    if (petitions.length === 0) {
      this.listEl.innerHTML = '<div style="color:var(--text-dim);font-size:12px;padding:8px;">No pending petitions.</div>';
      return;
    }

    var html = '';
    for (var i = 0; i < petitions.length; i++) {
      var p = petitions[i];
      if (p.status && p.status !== 'pending') continue;

      html += '<div class="petition-item">';
      html += '<div class="petition-type">' + _esc(p.building_type || p.type || 'Request') + '</div>';
      html += '<div class="petition-desc">' + _esc(p.description || p.reason || '') + '</div>';
      html += '<div class="petition-actions">';
      html += '<button class="btn-primary btn-small" data-petition-approve="' + i + '">Approve</button>';
      html += '<button class="btn-secondary btn-small" data-petition-dismiss="' + i + '">Dismiss</button>';
      html += '</div>';
      html += '</div>';
    }

    this.listEl.innerHTML = html;

    // Attach handlers
    var approveButtons = this.listEl.querySelectorAll('[data-petition-approve]');
    for (var a = 0; a < approveButtons.length; a++) {
      approveButtons[a].addEventListener('click', function() {
        var idx = parseInt(this.getAttribute('data-petition-approve'), 10);
        fetch('/petitions/' + idx + '/approve', { method: 'POST' })
          .then(function(r) { return r.json(); })
          .then(function() { Petitions.render(); });
      });
    }

    var dismissButtons = this.listEl.querySelectorAll('[data-petition-dismiss]');
    for (var d = 0; d < dismissButtons.length; d++) {
      dismissButtons[d].addEventListener('click', function() {
        var idx = parseInt(this.getAttribute('data-petition-dismiss'), 10);
        fetch('/petitions/' + idx + '/dismiss', { method: 'POST' })
          .then(function(r) { return r.json(); })
          .then(function() { Petitions.render(); });
      });
    }
  },
};
