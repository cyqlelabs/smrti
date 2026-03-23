/**
 * Petition panel — shows community requests from Space_Culture.
 */
(function() {
  'use strict';

  TOWN.updatePetitionBadge = function() {
    var badge = document.getElementById('petition-badge');
    if (!badge) return;
    fetch('/petitions')
      .then(function(r) { return r.json(); })
      .then(function(petitions) {
        var pending = petitions.filter(function(p) { return p.status === 'pending'; });
        badge.textContent = pending.length || '';
        badge.style.display = pending.length > 0 ? 'inline-block' : 'none';
      })
      .catch(function() {});
  };

  TOWN.showPetitionPanel = function() {
    var panel = document.getElementById('petition-panel');
    if (!panel) return;
    panel.style.display = 'block';

    fetch('/petitions')
      .then(function(r) { return r.json(); })
      .then(function(petitions) {
        var pending = petitions.filter(function(p) { return p.status === 'pending'; });
        var html = '<h3>Community Petitions</h3>';
        if (pending.length === 0) {
          html += '<p class="empty">No pending petitions.</p>';
        } else {
          pending.forEach(function(p) {
            var urgencyClass = p.urgency > 0.8 ? 'urgent' : (p.urgency > 0.5 ? 'moderate' : 'low');
            html += '<div class="petition-card ' + urgencyClass + '">';
            html += '<div class="petition-icon">' + _buildingIcon(p.building_type) + '</div>';
            html += '<div class="petition-info">';
            html += '<strong>' + p.building_type + '</strong>';
            html += '<div class="petition-urgency">Urgency: ' + Math.round(p.urgency * 100) + '%</div>';
            if (p.evidence.length > 0) {
              html += '<div class="petition-evidence">"' + TOWN.escapeHtml(p.evidence[0]) + '"</div>';
            }
            if (p.petitioners.length > 0) {
              html += '<div class="petition-petitioners">From: ' + p.petitioners.join(', ') + '</div>';
            }
            html += '</div>';
            html += '<div class="petition-actions">';
            html += '<button onclick="TOWN.approvePetition(' + p.idx + ')">Approve</button>';
            html += '<button onclick="TOWN.dismissPetition(' + p.idx + ')" class="dismiss">Dismiss</button>';
            html += '</div>';
            html += '</div>';
          });
        }
        panel.innerHTML = html;
      })
      .catch(function() {
        panel.innerHTML = '<p>Failed to load petitions.</p>';
      });
  };

  TOWN.hidePetitionPanel = function() {
    var panel = document.getElementById('petition-panel');
    if (panel) panel.style.display = 'none';
  };

  TOWN.approvePetition = function(idx) {
    fetch('/petitions/' + idx + '/approve', { method: 'POST' })
      .then(function(r) { return r.json(); })
      .then(function(result) {
        if (result.building_type) {
          TOWN.enterPlacementMode(result.building_type, [5, 4]);
        }
        TOWN.showPetitionPanel();
        TOWN.updatePetitionBadge();
      })
      .catch(function() {});
  };

  TOWN.dismissPetition = function(idx) {
    fetch('/petitions/' + idx + '/dismiss', { method: 'POST' })
      .then(function() {
        TOWN.showPetitionPanel();
        TOWN.updatePetitionBadge();
      })
      .catch(function() {});
  };

  function _buildingIcon(type) {
    var icons = {
      city_hall: '\uD83C\uDFDB\uFE0F', house: '\uD83C\uDFE0', farm: '\uD83C\uDF3E', market: '\uD83C\uDFEA',
      school: '\uD83C\uDFEB', workshop: '\uD83D\uDD28', clinic: '\uD83C\uDFE5', tavern: '\uD83C\uDF7A',
      church: '\u26EA', library: '\uD83D\uDCDA'
    };
    return icons[type] || '\uD83C\uDFD7\uFE0F';
  }

})();
