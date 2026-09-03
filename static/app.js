// app.js – client‑side controller for OmniRAG UI
// This script wires up all interactive elements defined in static/index.html.
// It uses the Fetch API to talk to the FastAPI backend (same origin).

/** Utility helpers **/
function $(selector) { return document.querySelector(selector); }
function $$(selector) { return Array.from(document.querySelectorAll(selector)); }

// ---------- Tab Switching ----------
function switchTab(tabName) {
  // Update tab button active state
  $$('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.id === `tab-${tabName}`);
    btn.setAttribute('aria-selected', btn.id === `tab-${tabName}`);
  });
  // Show corresponding panel
  $$('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `panel-${tabName}`);
  });
}

// ---------- Drag & Drop for File Upload ----------
function handleDragOver(event) {
  event.preventDefault();
  event.currentTarget.classList.add('dragover');
}
function handleDragLeave(event) {
  event.currentTarget.classList.remove('dragover');
}
function handleDrop(event) {
  event.preventDefault();
  event.currentTarget.classList.remove('dragover');
  const files = event.dataTransfer.files;
  if (files.length) handleFileSelected({ target: { files } });
}
function handleFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;
  // preview UI
  $('#preview-filename').textContent = file.name;
  $('#preview-filesize').textContent = `${(file.size / 1024).toFixed(1)} KB`;
  const ext = file.name.split('.').pop().toLowerCase();
  const badge = $('#preview-badge');
  badge.textContent = ext.toUpperCase();
  $('#file-preview-area').classList.remove('hidden');
  // store file globally for upload
  window._selectedFile = file;
}

async function uploadSelectedFile() {
  const file = window._selectedFile;
  if (!file) {
    alert('No file selected');
    return;
  }
  // show progress UI
  $('#upload-progress-area').classList.remove('hidden');
  $('#upload-progress-bar').style.width = '0%';
  $('#upload-status-text').textContent = 'Uploading...';

  const formData = new FormData();
  formData.append('file', file);
  try {
    const resp = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    if (!resp.ok) throw new Error(`Upload failed (${resp.status})`);
    const data = await resp.json();
    console.log('Upload response', data);
    // reset UI
    $('#upload-status-text').textContent = 'Processing OCR & Vector Index...';
    // simulate progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += 20;
      $('#upload-progress-bar').style.width = `${progress}%`;
      if (progress >= 100) {
        clearInterval(interval);
        $('#upload-status-text').textContent = 'Done! Refreshing document list...';
        // Refresh document list after short delay
        setTimeout(() => {
          listDocuments();
          $('#file-preview-area').classList.add('hidden');
          $('#upload-progress-area').classList.add('hidden');
          window._selectedFile = null;
        }, 800);
      }
    }, 300);
  } catch (err) {
    alert(err.message);
    $('#upload-progress-area').classList.add('hidden');
  }
}

// ---------- Document List & Re‑index ----------
async function listDocuments() {
  try {
    const resp = await fetch('/api/documents');
    if (!resp.ok) throw new Error('Failed to fetch documents');
    const docs = await resp.json();
    const listEl = $('#document-list');
    listEl.innerHTML = '';
    if (docs.length === 0) {
      listEl.innerHTML = '<div class="empty-state">No documents indexed.</div>';
    } else {
      docs.forEach(fn => {
        const div = document.createElement('div');
        div.className = 'doc-item';
        div.innerHTML = `
          <div class="doc-details">
            <span class="doc-name">${fn}</span>
          </div>
          <button class="btn btn-danger btn-sm" onclick="deleteDocument('${fn}')">Delete</button>
        `;
        listEl.appendChild(div);
      });
    }
    // update badge count
    $('#doc-count-badge').textContent = `${docs.length} Document${docs.length !== 1 ? 's' : ''}`;
  } catch (err) {
    console.error(err);
  }
}

async function deleteDocument(filename) {
  if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;
  try {
    const resp = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    });
    if (!resp.ok) throw new Error('Delete failed');
    await resp.json();
    listDocuments();
  } catch (err) {
    alert(err.message);
  }
}

async function reindexAllDocuments() {
  $('#reindex-btn').disabled = true;
  try {
    const resp = await fetch('/api/reindex', { method: 'POST' });
    if (!resp.ok) throw new Error('Re‑index failed');
    await resp.json();
    // give a tiny visual feedback
    $('#reindex-btn').textContent = 'Re‑indexed ✅';
    setTimeout(() => {
      $('#reindex-btn').textContent = 'Rebuild Index';
      $('#reindex-btn').disabled = false;
    }, 1500);
    listDocuments();
  } catch (err) {
    alert(err.message);
    $('#reindex-btn').disabled = false;
  }
}

// ---------- Query Handling ----------
async function handleQuerySubmit(event) {
  event.preventDefault();
  const input = $('#query-input');
  const question = input.value.trim();
  if (!question) return;
  // disable UI while waiting
  $('#query-submit-btn').disabled = true;
  const includeContexts = $('#include-contexts-toggle').checked;
  const ocrMode = $('#ocr-mode-select').value;
  // Build query string
  const url = new URL('/query', location.origin);
  url.searchParams.append('question', question);
  url.searchParams.append('include_contexts', includeContexts);
  // (OCR mode currently controlled server‑side via config; we just send as header for future use)
  try {
    const resp = await fetch(url, { headers: { 'X-OCR-Mode': ocrMode } });
    if (!resp.ok) throw new Error('Query failed');
    const data = await resp.json();
    // render bot message
    const messagesEl = $('#chat-messages');
    const botMsg = document.createElement('div');
    botMsg.className = 'message message-bot';
    botMsg.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-bubble"><p>${data.answer}</p></div>
    `;
    messagesEl.appendChild(botMsg);
    // if contexts requested, render collapsible drawer
    if (includeContexts && data.contexts && data.contexts.length) {
      const drawer = document.createElement('div');
      drawer.className = 'context-drawer';
      drawer.innerHTML = `
        <div class="context-header" onclick="this.parentElement.querySelector('.context-body').classList.toggle('hidden')">
          <span>Retrieved Context (${data.contexts.length})</span>
          <span>▾</span>
        </div>
        <div class="context-body hidden">
          ${data.contexts.map(c => `<div class="context-chunk">${c}</div>`).join('')}
        </div>
      `;
      messagesEl.appendChild(drawer);
    }
    // scroll to bottom
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (err) {
    alert(err.message);
  } finally {
    $('#query-submit-btn').disabled = false;
    input.value = '';
  }
}

function useSuggestion(text) {
  $('#query-input').value = text;
  $('#chat-form').dispatchEvent(new Event('submit'));
}

// ---------- Evaluation Benchmark ----------
async function runBenchmarkEvaluation() {
  const limit = parseInt($('#eval-limit-select').value, 10);
  $('#run-eval-btn').disabled = true;
  $('#run-eval-btn').textContent = 'Running...';
  try {
    const resp = await fetch('/eval/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit }),
    });
    if (!resp.ok) throw new Error('Eval failed');
    const report = await resp.json();
    // Populate metric cards
    $('#metric-faithfulness').textContent = `${report.faithfulness?.toFixed(1) ?? '--'}%`;
    $('#metric-answer-relevance').textContent = `${report.answer_relevance?.toFixed(1) ?? '--'}%`;
    $('#metric-context-relevance').textContent = `${report.context_relevance?.toFixed(1) ?? '--'}%`;
    $('#metric-similarity').textContent = `${report.semantic_similarity?.toFixed(1) ?? '--'}%`;
    // Show detailed results (optional – simple list)
    const container = $('#eval-results-container');
    container.innerHTML = '';
    if (report.samples && report.samples.length) {
      report.samples.forEach((s, idx) => {
        const div = document.createElement('div');
        div.className = 'eval-sample-result';
        div.innerHTML = `
          <div class="eval-sample-question">Q${idx + 1}: ${s.question}</div>
          <div class="eval-q-and-a">
            <div><strong>Answer:</strong> ${s.answer}</div>
            <div><strong>Ground Truth:</strong> ${s.ground_truth}</div>
          </div>
        `;
        container.appendChild(div);
      });
    } else {
      container.innerHTML = '<div class="empty-state">No results returned.</div>';
    }
    $('#eval-duration-badge').textContent = `Done (${(report.duration_seconds ?? 0).toFixed(1)}s)`;
  } catch (err) {
    alert(err.message);
  } finally {
    $('#run-eval-btn').disabled = false;
    $('#run-eval-btn').textContent = 'Run Evaluation Benchmark';
  }
}

// ---------- Initialization ----------
window.addEventListener('load', () => {
  listDocuments();
  // optional: set system status
  $('#system-status-text').textContent = 'System Active';
});

// Export functions for inline HTML event handlers
window.switchTab = switchTab;
window.handleDragOver = handleDragOver;
window.handleDragLeave = handleDragLeave;
window.handleDrop = handleDrop;
window.handleFileSelected = handleFileSelected;
window.uploadSelectedFile = uploadSelectedFile;
window.reindexAllDocuments = reindexAllDocuments;
window.deleteDocument = deleteDocument;
window.runBenchmarkEvaluation = runBenchmarkEvaluation;
window.handleQuerySubmit = handleQuerySubmit;
window.useSuggestion = useSuggestion;
