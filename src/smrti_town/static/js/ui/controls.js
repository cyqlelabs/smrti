/**
 * Play/pause/skip controls.
 */

var Controls = {
  el: null,
  btnPause: null,
  btnPlay: null,
  btnSkip: null,

  init: function() {
    this.el = document.getElementById('ui-controls');
    this.btnPause = document.getElementById('btn-pause');
    this.btnPlay = document.getElementById('btn-play');
    this.btnSkip = document.getElementById('btn-skip');

    this.btnPause.addEventListener('click', function() {
      GameState.paused = true;
      Controls.updateButtons();
      fetch('/pause', { method: 'POST' }).catch(function() {
        GameState.paused = false;
        Controls.updateButtons();
      });
    });

    this.btnPlay.addEventListener('click', function() {
      if (!GameState.paused) return;
      GameState.paused = false;
      Controls.updateButtons();
      fetch('/resume', { method: 'POST' })
        .then(function(r) { return r.json(); })
        .then(function(data) {
          if (data.error) {
            GameState.paused = true;
            Controls.updateButtons();
            EventLog.add('Cannot resume: ' + data.error, 'crisis');
          }
        })
        .catch(function() {
          GameState.paused = true;
          Controls.updateButtons();
        });
    });

    this.btnSkip.addEventListener('click', function() {
      fetch('/skip', { method: 'POST' });
    });
  },

  show: function() {
    this.el.classList.remove('hidden');
  },

  hide: function() {
    this.el.classList.add('hidden');
  },

  updateButtons: function() {
    if (GameState.paused) {
      this.btnPause.classList.add('active');
      this.btnPlay.classList.remove('active');
    } else {
      this.btnPause.classList.remove('active');
      this.btnPlay.classList.add('active');
    }
  },
};
