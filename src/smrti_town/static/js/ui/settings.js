/**
 * Settings modal for LLM configuration.
 */

var Settings = {
  overlayEl: null,
  formEl: null,
  _currentSettings: {},

  init: function() {
    this.overlayEl = document.getElementById('ui-settings');
    this.formEl = document.getElementById('settings-form');

    document.getElementById('settings-save').addEventListener('click', function() {
      Settings.save();
    });
    document.getElementById('settings-cancel').addEventListener('click', function() {
      Settings.hide();
    });
  },

  show: function() {
    // Fetch current settings
    fetch('/settings')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        Settings._currentSettings = data;
        Settings._renderForm(data);
        Settings.overlayEl.classList.remove('hidden');
      })
      .catch(function() {
        Settings._renderForm({});
        Settings.overlayEl.classList.remove('hidden');
      });
  },

  hide: function() {
    this.overlayEl.classList.add('hidden');
  },

  _renderForm: function(data) {
    var fields = [
      { key: 'base_url', label: 'Base URL', type: 'text', placeholder: 'http://0.0.0.0:8421/v1' },
      { key: 'model', label: 'Model', type: 'text', placeholder: 'Qwen3.5-9B-Q8_0.gguf' },
      { key: 'enabled', label: 'LLM Enabled', type: 'checkbox' },
      { key: 'temperature', label: 'Temperature', type: 'number', step: '0.1' },
      { key: 'max_tokens', label: 'Max Tokens', type: 'number' },
    ];

    var html = '';
    for (var i = 0; i < fields.length; i++) {
      var f = fields[i];
      var val = data[f.key] !== undefined ? data[f.key] : '';
      html += '<div class="settings-field">';
      html += '<label for="setting-' + f.key + '">' + _esc(f.label) + '</label>';

      if (f.type === 'checkbox') {
        html += '<input type="checkbox" id="setting-' + f.key + '"' +
          (val ? ' checked' : '') + '>';
      } else {
        html += '<input type="' + f.type + '" id="setting-' + f.key +
          '" value="' + _esc(String(val)) + '"' +
          (f.placeholder ? ' placeholder="' + _esc(f.placeholder) + '"' : '') +
          (f.step ? ' step="' + f.step + '"' : '') + '>';
      }
      html += '</div>';
    }
    this.formEl.innerHTML = html;
  },

  save: function() {
    var data = {};
    var fields = ['base_url', 'model', 'enabled', 'temperature', 'max_tokens'];
    for (var i = 0; i < fields.length; i++) {
      var el = document.getElementById('setting-' + fields[i]);
      if (!el) continue;
      if (el.type === 'checkbox') {
        data[fields[i]] = el.checked;
      } else if (el.type === 'number') {
        data[fields[i]] = el.value ? parseFloat(el.value) : undefined;
      } else {
        data[fields[i]] = el.value;
      }
    }

    fetch('/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
      .then(function(r) { return r.json(); })
      .then(function() {
        Settings.hide();
      })
      .catch(function(err) {
        console.error('Failed to save settings:', err);
      });
  },
};
