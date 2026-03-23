/**
 * Processes incoming WebSocket messages, updates GameState,
 * and triggers UI/Phaser updates.
 */

var TickProcessor = {
  /**
   * Handle a parsed WS message.
   * @param {object} msg
   */
  handle: function(msg) {
    var type = msg.type;

    switch (type) {
      case 'state':
        this._handleState(msg.data || msg);
        break;

      case 'tick':
        this._handleTick(msg);
        break;

      case 'dialogue_patch':
        this._handleDialoguePatch(msg);
        break;

      case 'phase':
        this._handlePhase(msg);
        break;

      case 'game_phase':
        this._handleGamePhase(msg);
        break;

      case 'mayor_candidates':
        this._handleMayorCandidates(msg);
        break;

      case 'council_formed':
        this._handleCouncilFormed(msg);
        break;

      case 'council_meeting':
        this._handleCouncilMeeting(msg);
        break;

      case 'building_placed':
        this._handleBuildingPlaced(msg);
        break;

      case 'building_demolished':
        EventLog.add('Building demolished: ' + (msg.place_name || ''), 'event');
        break;

      case 'petition_approved':
        EventLog.add('Petition approved: ' + (msg.building_type || ''), 'milestone');
        Petitions.render();
        break;

      case 'petition_update':
        EventLog.add('Petition ' + (msg.status || 'updated'), 'event');
        Petitions.render();
        break;

      case 'council_result':
        if (msg.approved && msg.building_key) {
          GameState.selectedBuilding = msg.building_key;
          EventLog.add('Council approved: place ' + msg.building_key.replace(/_/g, ' '), 'milestone');
          Council.hide();
          if (typeof Toolbar !== 'undefined') Toolbar.render();
        } else if (msg.approved) {
          var desc = msg.description || msg.action_type || 'proposal';
          EventLog.add('Council approved: ' + desc, 'milestone');
          Council.hide();
        } else {
          EventLog.add('Council proposal rejected.', 'event');
          Council.hide();
        }
        break;

      case 'immigration':
        this._handleImmigration(msg);
        break;

      case 'event':
        EventLog.add(msg.description || msg.event_type || 'Event', 'event');
        break;

      case 'milestone':
        EventLog.add((msg.name || '') + ': ' + (msg.description || ''), 'milestone');
        this._celebrateOnScreen();
        break;

      case 'crisis':
        EventLog.add('CRISIS: ' + (msg.description || msg.crisis_type || ''), 'crisis');
        break;

      case 'game_over':
        GameState.phase = PHASES.GAME_OVER;
        GameState.gameOverReason = msg.reason || 'Unknown';
        this._switchScene('GameOverScene');
        break;

      case 'generating':
        GameState.phase = PHASES.GENERATING;
        GameState.generatingMessage = msg.message || 'Generating...';
        GameState.generatingHint = msg.hint || '';
        this._showGenerating();
        break;

      case 'reset':
        this._handleReset();
        break;

      case 'error':
        EventLog.add('Error: ' + (msg.message || ''), 'crisis');
        console.error('Server error:', msg.message);
        break;

      case 'paused':
        GameState.paused = true;
        Controls.updateButtons();
        break;

      case 'resumed':
        GameState.paused = false;
        Controls.updateButtons();
        break;

      case 'ack':
        this._handleAck(msg);
        break;

      case 'ping':
        // keep-alive, no action needed
        break;

      default:
        console.log('Unknown WS message type:', type, msg);
    }
  },

  _handleState: function(data) {
    GameState.tick = data.tick || data.tick_number || 0;
    if (data.calendar) GameState.calendar = data.calendar;
    if (data.agents) GameState.agents = data.agents;
    if (data.places) GameState.places = data.places;
    if (data.gridmap) GameState.grid = data.gridmap;
    if (data.economy) GameState.economy = data.economy;
    if (data.petitions) GameState.petitions = data.petitions;
    if (data.director_mode) GameState.directorMode = data.director_mode;

    // Route to correct scene/UI based on server phase
    var phase = data.phase;
    if (phase === 'gameplay') {
      GameState.phase = PHASES.GAMEPLAY;
      this._switchScene('TownScene');
    } else if (phase === 'opening_place_hall') {
      GameState.phase = PHASES.OPENING_PLACE_HALL;
      this._switchScene('OpeningScene');
    } else if (phase === 'opening_choose_mayor') {
      GameState.phase = PHASES.OPENING_CHOOSE_MAYOR;
      this._switchScene('OpeningScene');
      if (data.candidates) this._showMayorSelect({ candidates: data.candidates });
    } else if (phase === 'opening_council') {
      GameState.phase = PHASES.OPENING_COUNCIL;
      this._switchScene('OpeningScene');
      if (data.council_specs) {
        GameState.council.members = data.council_specs;
        this._showCouncilReveal();
      }
    }

    this._updateAllUI();
  },

  _handleTick: function(msg) {
    GameState.tick = msg.tick || msg.tick_number || GameState.tick + 1;
    if (msg.calendar) GameState.calendar = msg.calendar;
    if (msg.citizens) GameState.agents = msg.citizens;
    else if (msg.agents) GameState.agents = msg.agents;
    if (msg.economy) GameState.economy = msg.economy;
    if (msg.topology) GameState.places = msg.topology.places || GameState.places;
    if (msg.gridmap) GameState.grid = msg.gridmap;
    if (msg.director_mode) GameState.directorMode = msg.director_mode;

    // Process tick events
    var events = msg.events || [];
    for (var i = 0; i < events.length; i++) {
      var e = events[i];
      EventLog.add(e.description || e.text || JSON.stringify(e), 'event');
    }

    // Deaths
    var deaths = msg.deaths || [];
    for (var d = 0; d < deaths.length; d++) {
      EventLog.add(deaths[d] + ' has died.', 'crisis');
    }

    // Births
    var births = msg.births || [];
    for (var b = 0; b < births.length; b++) {
      var birth = births[b];
      EventLog.add('A child is born: ' + (birth.name || 'Unknown'), 'milestone');
    }

    // Conversations
    var convos = msg.conversations || [];
    for (var c = 0; c < convos.length; c++) {
      var conv = convos[c];
      if (conv.speaker && conv.dialogue) {
        SpeechBubbles.show(conv.speaker, conv.dialogue, 5000);
      }
    }

    this._updateAllUI();
  },

  _handleDialoguePatch: function(msg) {
    // Server sends a single patch per message: {speaker, line, target, location, tick, calendar_day, calendar_hour}
    var line = msg.line || msg.dialogue;
    if (msg.speaker && line) {
      SpeechBubbles.show(msg.speaker, line, 5000);
      // Use the calendar time from when the dialogue was submitted, not current time.
      var timeStr;
      if (msg.calendar_day !== undefined && msg.calendar_hour !== undefined) {
        timeStr = 'D' + msg.calendar_day + ' ' +
          (msg.calendar_hour < 10 ? '0' : '') + Math.floor(msg.calendar_hour) + ':00';
      }
      EventLog.add(msg.speaker + ': "' + line + '"', 'dialogue', timeStr);
    }
  },

  _handlePhase: function(msg) {
    var phase = msg.phase;
    if (phase === 'opening_choose_mayor' && msg.candidates) {
      this._handleMayorCandidates(msg);
    } else if (phase === 'opening_council') {
      // council_specs from server, council members array may be in council_specs
      if (msg.council_specs && !msg.council) {
        msg.council = msg.council_specs;
      }
      this._handleCouncilFormed(msg);
    } else {
      this._handleGamePhase(msg);
    }
  },

  _handleGamePhase: function(msg) {
    var phase = msg.phase;
    if (phase === 'opening_place_hall') {
      GameState.phase = PHASES.OPENING_PLACE_HALL;
      this._switchScene('OpeningScene');
    } else if (phase === 'opening_choose_mayor') {
      GameState.phase = PHASES.OPENING_CHOOSE_MAYOR;
    } else if (phase === 'opening_council') {
      GameState.phase = PHASES.OPENING_COUNCIL;
    } else if (phase === 'gameplay') {
      GameState.phase = PHASES.GAMEPLAY;
      this._switchScene('TownScene');
    }
  },

  _handleMayorCandidates: function(msg) {
    GameState.mayorCandidates = msg.candidates || [];
    GameState.phase = PHASES.OPENING_CHOOSE_MAYOR;
    this._showMayorSelect();
  },

  _handleCouncilFormed: function(msg) {
    GameState.council.members = msg.council || [];
    GameState.phase = PHASES.OPENING_COUNCIL;
    if (msg.gridmap) GameState.grid = msg.gridmap;
    if (msg.topology && msg.topology.places) GameState.places = msg.topology.places;
    this._showCouncilReveal();
  },

  _handleCouncilMeeting: function(msg) {
    Council.showMeeting(msg.meeting || msg);
  },

  _handleBuildingPlaced: function(msg) {
    var building = msg.building || msg;
    // Add to grid if not already there
    var found = false;
    var buildings = GameState.grid.buildings || [];
    for (var i = 0; i < buildings.length; i++) {
      if (buildings[i].grid_x === building.grid_x && buildings[i].grid_y === building.grid_y) {
        found = true;
        break;
      }
    }
    if (!found) {
      buildings.push(building);
    }
    EventLog.add('New building: ' + (building.building_key || building.type || ''), 'event');
    BuildingRenderer.sync();
  },

  _handleImmigration: function(msg) {
    var citizens = msg.citizens || [];
    for (var i = 0; i < citizens.length; i++) {
      EventLog.add('New citizen arrives: ' + (citizens[i].name || ''), 'milestone');
    }
  },

  _handleReset: function() {
    GameState.phase = PHASES.BOOT;
    GameState.agents = [];
    GameState.places = [];
    GameState.grid = { buildings: [] };
    GameState.economy = {};
    GameState.petitions = [];
    GameState.council = { members: [], pending_meeting: null };
    GameState.events = [];
    EventLog.clear();
  },

  _handleAck: function(msg) {
    if (msg.action === 'pause') {
      GameState.paused = true;
      Controls.updateButtons();
    } else if (msg.action === 'resume' || msg.action === 'start') {
      GameState.paused = false;
      Controls.updateButtons();
    }
  },

  _updateAllUI: function() {
    Topbar.update();
    AgentRenderer.sync();
    BuildingRenderer.sync();

    // Update sidebar if something is selected
    if (GameState.selectedAgent) {
      var agents = GameState.agents || [];
      for (var i = 0; i < agents.length; i++) {
        if (agents[i].name === GameState.selectedAgent) {
          Sidebar.showAgent(agents[i]);
          break;
        }
      }
    }
  },

  _switchScene: function(sceneKey) {
    if (!GameState.phaserGame) return;
    var sm = GameState.phaserGame.scene;

    // Stop all gameplay scenes
    var scenes = ['BootScene', 'OpeningScene', 'TownScene', 'GameOverScene'];
    for (var i = 0; i < scenes.length; i++) {
      if (sm.isActive(scenes[i]) && scenes[i] !== sceneKey) {
        sm.stop(scenes[i]);
      }
    }

    if (!sm.isActive(sceneKey)) {
      sm.start(sceneKey);
    }
  },

  _showGenerating: function() {
    var el = document.getElementById('ui-generating');
    document.getElementById('generating-message').textContent = GameState.generatingMessage;
    document.getElementById('generating-hint').textContent = GameState.generatingHint;
    el.classList.remove('hidden');
  },

  _showMayorSelect: function() {
    var el = document.getElementById('ui-mayor-select');
    var container = document.getElementById('mayor-candidates');
    var candidates = GameState.mayorCandidates;

    var html = '';
    for (var i = 0; i < candidates.length; i++) {
      var c = candidates[i];
      html += '<div class="mayor-card" data-candidate="' + i + '">';
      html += '<h3>' + _esc(c.name) + '</h3>';
      html += '<div class="mayor-bio">' + _esc(c.bio || '') + '</div>';
      if (c.personality) {
        html += '<div class="mayor-style">Personality: ' + _esc(c.personality) + '</div>';
      }
      if (c.governing_style) {
        html += '<div class="mayor-style">Style: ' + _esc(c.governing_style) + '</div>';
      }
      html += '</div>';
    }
    container.innerHTML = html;

    // Attach click handlers
    var cards = container.querySelectorAll('.mayor-card');
    for (var j = 0; j < cards.length; j++) {
      cards[j].addEventListener('click', function() {
        var idx = parseInt(this.getAttribute('data-candidate'), 10);
        fetch('/opening/choose-mayor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ candidate_index: idx }),
        });
        el.classList.add('hidden');
      });
    }

    el.classList.remove('hidden');
  },

  _showCouncilReveal: function() {
    var el = document.getElementById('ui-council-reveal');
    var container = document.getElementById('council-members');
    var members = GameState.council.members;

    var html = '';
    for (var i = 0; i < members.length; i++) {
      var m = members[i];
      html += '<div class="council-card">';
      html += '<h4>' + _esc(m.name) + '</h4>';
      html += '<div class="council-role">' + _esc(m.role || '') + '</div>';
      html += '<div class="council-domain">' + _esc(m.domain || '') + '</div>';
      if (m.personality) {
        html += '<div style="font-size:11px;color:var(--text-dim);margin-top:4px;">' +
          _esc(m.personality) + '</div>';
      }
      html += '</div>';
    }
    container.innerHTML = html;

    document.getElementById('council-begin').onclick = function() {
      fetch('/opening/begin', { method: 'POST' });
      el.classList.add('hidden');
    };

    el.classList.remove('hidden');
  },

  _celebrateOnScreen: function() {
    if (!GameState.phaserGame) return;
    var scene = GameState.phaserGame.scene.getScene('TownScene');
    if (scene && scene.sys.isActive()) {
      var cam = scene.cameras.main;
      Particles.celebrate(scene,
        cam.scrollX + cam.width / 2,
        cam.scrollY + cam.height / 2
      );
    }
  },
};
