/* ================================================================
   demo.js — startDemoMode() — offline demo with fake agents
   ================================================================ */
window.TOWN = window.TOWN || {};

TOWN.startDemoMode = function() {
  /* Wait 3 seconds — if no agents have appeared (from WS), launch demo */
  setTimeout(function() {
    if (Object.keys(TOWN.state.agents).length > 0) return;

    TOWN.state.demoMode = true;
    TOWN.addLogEntry('system', 'No server detected \u2014 running demo mode');

    var scene = TOWN.state.scene;
    if (!scene) return;

    /* ── Demo Agents ───────────────────────────────────────────── */
    var demoAgents = [
      {
        name: 'Alice', location: 'Cafe_Rosetta',
        action: 'TALK', action_target: 'Bob', target: 'Bob',
        life_stage: 'adult', age_years: 28, alive: true,
        drives: { hunger: 30, energy: 70, social: 20, curiosity: 45, duty: 10, romance: 15 },
        relationships: [
          { name: 'Bob', state: 'romantic', valence: 0.7 },
          { name: 'Emma', state: 'parent', valence: 0.9 },
        ],
      },
      {
        name: 'Bob', location: 'Cafe_Rosetta',
        action: 'TALK', action_target: 'Alice', target: 'Alice',
        life_stage: 'adult', age_years: 31, alive: true,
        drives: { hunger: 45, energy: 60, social: 35, curiosity: 20, duty: 55, romance: 25 },
        relationships: [
          { name: 'Alice', state: 'romantic', valence: 0.7 },
          { name: 'Sofia', state: 'friend', valence: 0.4 },
          { name: 'Emma', state: 'parent', valence: 0.8 },
        ],
      },
      {
        name: 'Sofia', location: 'Public_Library',
        action: 'READ', action_target: null,
        life_stage: 'adult', age_years: 26, alive: true,
        drives: { hunger: 20, energy: 80, social: 60, curiosity: 70, duty: 30, romance: 5 },
        relationships: [
          { name: 'Bob', state: 'friend', valence: 0.3 },
        ],
      },
      {
        name: 'Emma', location: 'Central_Park',
        action: 'PLAY', action_target: null,
        life_stage: 'child', age_years: 8, alive: true,
        drives: { hunger: 15, energy: 90, social: 10, curiosity: 80, duty: 5, romance: 0 },
        relationships: [
          { name: 'Alice', state: 'parent', valence: 0.9 },
          { name: 'Bob', state: 'parent', valence: 0.8 },
        ],
      },
      {
        name: 'Elder John', location: 'Town_Market',
        action: 'BROWSE', action_target: null,
        life_stage: 'elder', age_years: 72, alive: true,
        drives: { hunger: 55, energy: 35, social: 40, curiosity: 30, duty: 60, romance: 0 },
        relationships: [],
      },
    ];

    /* Spawn agents */
    for (var i = 0; i < demoAgents.length; i++) {
      var a = demoAgents[i];
      TOWN.state.agents[a.name] = a;
      TOWN.createAgentSprite(scene, a);
      var sp = TOWN.state.agentSprites[a.name];
      if (sp && a.drives) {
        TOWN.drawDriveBars(sp.driveBarContainer, sp.radius, a.drives);
      }
    }
    TOWN.updateOccupantCounts();

    /* ── Demo Conversations ────────────────────────────────────── */
    setTimeout(function() {
      TOWN.showSpeechBubble(scene, 'Alice', 'Did you hear about the summer festival?');
      TOWN.addLogEntry('talk', 'Alice to Bob: \u201CDid you hear about the summer festival?\u201D');
    }, 1500);

    setTimeout(function() {
      TOWN.showSpeechBubble(scene, 'Bob', 'Yes! Live music at the park this weekend.');
      TOWN.addLogEntry('talk', 'Bob to Alice: \u201CYes! Live music at the park this weekend.\u201D');
    }, 5000);

    setTimeout(function() {
      TOWN.showSpeechBubble(scene, 'Sofia', 'This chapter on ancient history is fascinating\u2026');
      TOWN.addLogEntry('talk', 'Sofia: \u201CThis chapter on ancient history is fascinating\u2026\u201D');
    }, 8000);

    /* ── Demo Movement Loop ────────────────────────────────────── */
    var demoTick = 0;
    var demoLocations = {
      'Alice':       ['Cafe_Rosetta', 'Main_Street', 'Central_Park', 'Alice_Home', 'Cafe_Rosetta'],
      'Bob':         ['Cafe_Rosetta', 'Main_Street', 'Town_Market',  'Main_Street', 'Cafe_Rosetta'],
      'Sofia':       ['Public_Library', 'Main_Street', 'Cafe_Rosetta', 'Sofia_Home', 'Public_Library'],
      'Emma':        ['Central_Park', 'Elm_Street', 'Alice_Home', 'Central_Park', 'Central_Park'],
      'Elder John':  ['Town_Market', 'Main_Street', 'Central_Park', 'Town_Market', 'Town_Market'],
    };
    var demoTods  = ['morning', 'afternoon', 'evening', 'night', 'morning'];
    var demoHours = [8, 14, 19, 22, 6];
    var demoSeasons = ['spring', 'spring', 'summer', 'summer', 'autumn'];

    var demoPhrases = [
      'Beautiful day, isn\'t it?',
      'I need to visit the market later.',
      'Have you read any good books?',
      'The park looks lovely this season.',
      'I wonder what\'s for dinner.',
      'Time flies so quickly here.',
      'Let\'s organize something this weekend!',
      'I miss the old days.',
      'The flowers are blooming early this year.',
      'Would anyone like to go for a walk?',
    ];

    TOWN.state.demoInterval = setInterval(function() {
      if (!TOWN.state.demoMode) return;

      demoTick = (demoTick + 1) % 5;

      /* Update calendar */
      TOWN.state.calendar = {
        hour: demoHours[demoTick],
        day: 1 + Math.floor(demoTick / 5),
        season: demoSeasons[demoTick],
        year: 1,
        time_of_day: demoTods[demoTick],
      };
      TOWN.state.tickNumber++;
      TOWN.updateClockUI();

      /* Move agents */
      var names = Object.keys(demoLocations);
      for (var i = 0; i < names.length; i++) {
        var name = names[i];
        var locs = demoLocations[name];
        var agent = TOWN.state.agents[name];
        if (!agent) continue;

        agent.location = locs[demoTick];

        /* Drift drives randomly */
        if (agent.drives) {
          var dKeys = Object.keys(agent.drives);
          for (var d = 0; d < dKeys.length; d++) {
            agent.drives[dKeys[d]] = Math.max(0, Math.min(100,
              agent.drives[dKeys[d]] + Math.floor(Math.random() * 20 - 8)
            ));
          }
        }

        TOWN.updateAgentSprite(scene, agent);
      }
      TOWN.updateOccupantCounts();

      /* Occasional movement log */
      var mover = names[Math.floor(Math.random() * names.length)];
      var moverAgent = TOWN.state.agents[mover];
      if (moverAgent) {
        TOWN.addLogEntry('move', mover + ' walked to ' + moverAgent.location.replace(/_/g, ' '));
      }

      /* Occasional speech bubble */
      if (Math.random() < 0.35) {
        var speakers = Object.keys(TOWN.state.agents);
        var speaker = speakers[Math.floor(Math.random() * speakers.length)];
        var phrase = demoPhrases[Math.floor(Math.random() * demoPhrases.length)];
        TOWN.showSpeechBubble(scene, speaker, phrase);
      }

      /* Update sidebar if selected */
      if (TOWN.state.selectedAgent && TOWN.state.agents[TOWN.state.selectedAgent]) {
        TOWN.renderAgentSidebar(TOWN.state.agents[TOWN.state.selectedAgent]);
      }
    }, 6000);

  }, 3000);
};
