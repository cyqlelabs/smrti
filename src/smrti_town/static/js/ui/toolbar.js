/**
 * Building placement toolbar and grid interaction.
 */
(function() {
  'use strict';

  var _placementMode = false;
  var _selectedType = null;
  var _ghostGfx = null;

  TOWN.initToolbar = function() {
    var container = document.getElementById('toolbar');
    if (!container) return;
    TOWN.refreshToolbar();
  };

  TOWN.refreshToolbar = function() {
    var container = document.getElementById('toolbar');
    if (!container) return;
    container.innerHTML = '';

    fetch('/buildable')
      .then(function(r) { return r.json(); })
      .then(function(buildings) {
        buildings.forEach(function(b) {
          var btn = document.createElement('button');
          btn.className = 'toolbar-btn';
          btn.title = b.type + ' (' + b.grid_size[0] + 'x' + b.grid_size[1] + ')';
          btn.textContent = _buildingIcon(b.type);
          btn.onclick = function() { TOWN.enterPlacementMode(b.type, b.grid_size); };
          container.appendChild(btn);
        });
      })
      .catch(function() {});
  };

  TOWN.enterPlacementMode = function(buildingType, gridSize) {
    _placementMode = true;
    _selectedType = buildingType;
    document.body.classList.add('placement-mode');
    TOWN.addLogEntry('system', 'Click on the map to place ' + buildingType + '. ESC to cancel.');
  };

  TOWN.exitPlacementMode = function() {
    _placementMode = false;
    _selectedType = null;
    document.body.classList.remove('placement-mode');
    if (_ghostGfx && TOWN.state.scene) {
      _ghostGfx.destroy();
      _ghostGfx = null;
    }
  };

  TOWN.isPlacementMode = function() {
    return _placementMode;
  };

  TOWN.handlePlacementClick = function(worldX, worldY) {
    if (!_placementMode || !_selectedType) return false;
    /* Convert world coords to grid coords (16px cells) */
    var gridX = Math.floor(worldX / 16);
    var gridY = Math.floor(worldY / 16);

    fetch('/place-building', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: _selectedType, grid_x: gridX, grid_y: gridY })
    })
    .then(function(r) { return r.json(); })
    .then(function(result) {
      if (result.error) {
        TOWN.addLogEntry('system', '\u274C ' + result.error);
      } else {
        TOWN.addLogEntry('system', '\u2705 ' + _selectedType + ' placed!');
        TOWN.refreshToolbar();
      }
    })
    .catch(function(err) {
      TOWN.addLogEntry('system', '\u274C Placement failed: ' + err.message);
    });

    TOWN.exitPlacementMode();
    return true;
  };

  function _buildingIcon(type) {
    var icons = {
      city_hall: '\uD83C\uDFDB\uFE0F', house: '\uD83C\uDFE0', farm: '\uD83C\uDF3E', market: '\uD83C\uDFEA',
      school: '\uD83C\uDFEB', workshop: '\uD83D\uDD28', clinic: '\uD83C\uDFE5', tavern: '\uD83C\uDF7A',
      church: '\u26EA', library: '\uD83D\uDCDA'
    };
    return icons[type] || '\uD83C\uDFD7\uFE0F';
  }

  /* ESC to cancel placement */
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && _placementMode) {
      TOWN.exitPlacementMode();
    }
  });

})();
