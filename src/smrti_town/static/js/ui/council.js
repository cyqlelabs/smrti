/**
 * Council meeting overlay — debate transcript + proposal approve/reject/counter.
 */

var Council = {
  overlayEl: null,
  debateEl: null,
  proposalEl: null,
  actionsEl: null,

  init: function() {
    this.overlayEl = document.getElementById('ui-council-overlay');
    this.debateEl = document.getElementById('council-debate');
    this.proposalEl = document.getElementById('council-proposal');
    this.actionsEl = document.getElementById('council-actions');
  },

  /**
   * Show a council meeting.
   * @param {object} meeting - {meeting_id, debate: [{role, name, argument}], proposal: {action_type, building_key, description, cost}, status}
   */
  showMeeting: function(meeting) {
    GameState.council.pending_meeting = meeting;

    // Render debate
    var debateHtml = '';
    var debate = meeting.debate || [];
    for (var i = 0; i < debate.length; i++) {
      var d = debate[i];
      debateHtml += '<div class="debate-entry">';
      debateHtml += '<span class="debate-role">' + _esc(d.role) + '</span>';
      debateHtml += '<span class="debate-name">' + _esc(d.name) + '</span>';
      debateHtml += '<div class="debate-argument">' + _esc(d.argument) + '</div>';
      debateHtml += '</div>';
    }
    this.debateEl.innerHTML = debateHtml;

    // Render proposal
    var proposal = meeting.proposal || {};
    var proposalHtml = '';
    proposalHtml += '<div class="proposal-type">' + _esc(proposal.action_type || 'Build') + '</div>';
    proposalHtml += '<div class="proposal-desc">' + _esc(proposal.description || '') + '</div>';
    if (proposal.building_key) {
      var def = BUILDINGS[proposal.building_key];
      proposalHtml += '<div class="proposal-desc" style="color:var(--text-dim);margin-top:2px;">' +
        _esc(proposal.building_key) + '</div>';
    }
    if (proposal.cost !== undefined) {
      proposalHtml += '<div class="proposal-cost">Cost: ' + proposal.cost + 'g</div>';
    }
    this.proposalEl.innerHTML = proposalHtml;

    // Render action buttons
    var meetingId = meeting.meeting_id;
    var actionsHtml = '';
    if (meeting.status === 'pending' || !meeting.status) {
      actionsHtml += '<button class="btn-primary" id="council-approve">Approve</button>';
      actionsHtml += '<button class="btn-danger" id="council-reject">Reject</button>';
      if (proposal.building_key) {
        actionsHtml += '<button class="btn-secondary" id="council-counter">Counter</button>';
      }
    } else {
      actionsHtml += '<div style="color:var(--text-dim);font-size:12px;">Decision: ' +
        _esc(meeting.status) + '</div>';
    }
    this.actionsEl.innerHTML = actionsHtml;

    // Attach handlers
    var approveBtn = document.getElementById('council-approve');
    var rejectBtn = document.getElementById('council-reject');
    var counterBtn = document.getElementById('council-counter');

    if (approveBtn) {
      approveBtn.addEventListener('click', function() {
        fetch('/council/approve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ meeting_id: meetingId }),
        }).then(function() { Council.hide(); });
      });
    }

    if (rejectBtn) {
      rejectBtn.addEventListener('click', function() {
        fetch('/council/reject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ meeting_id: meetingId }),
        }).then(function() { Council.hide(); });
      });
    }

    if (counterBtn) {
      counterBtn.addEventListener('click', function() {
        // Simple counter: prompt for building type
        var counterType = prompt('Counter-propose building type (e.g., school, clinic):');
        if (counterType) {
          fetch('/council/counter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ meeting_id: meetingId, building_key: counterType }),
          }).then(function() { Council.hide(); });
        }
      });
    }

    this.overlayEl.classList.remove('hidden');
  },

  hide: function() {
    this.overlayEl.classList.add('hidden');
    GameState.council.pending_meeting = null;
  },
};
