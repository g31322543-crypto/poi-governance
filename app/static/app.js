'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

const TONE = { auto_merge: 'blue', auto_create: 'green', auto_reject: 'red', escalate: 'amber' };
const VERD_TONE = { approve: 'green', reject: 'red', publish: 'green', hide: 'red', merge: 'blue', new: 'green', uncertain: 'amber' };

function badge(text, tone) { return `<span class="badge ${tone}">${esc(text)}</span>`; }

function confBar(p, tone) {
  const pct = Math.max(0, Math.min(100, Math.round((p ?? 0) * 100)));
  return `<div class="confbar"><div class="confbar-fill ${tone}" style="width:${pct}%"></div></div><span class="confval">${pct}%</span>`;
}

// ---- status ---------------------------------------------------------------
async function refreshStatus() {
  try {
    const s = await getJSON('/api/status');
    const dbTxt = s.db_size ? `${s.db_size} places` : (s.db_loaded ? 'loading…' : '—');
    const backend = s.dedup_backend === 'llm' ? 'DeepSeek (LLM)' : 'rules (offline)';
    const agentTxt = s.agent_loaded ? 'ready' : 'warming…';
    const storeTxt = s.store ? `audit ${s.store.audit} · places ${s.store.places}` : '—';
    $('#status-chips').innerHTML = [
      chip('model', 'Laya · ' + esc(s.laya_subfolder), 'blue'),
      chip('dedup', backend, s.dedup_backend === 'llm' ? 'blue' : 'amber'),
      chip('db', dbTxt, s.db_loaded ? 'green' : 'gray'),
      chip('agent', agentTxt, s.agent_loaded ? 'green' : 'amber'),
      chip('store', storeTxt, s.store ? 'green' : 'gray'),
      chip('gate', 'auto ≥ ' + Math.round(s.confidence_auto * 100) + '%', 'gray'),
    ].join('');
  } catch (e) {
    $('#status-chips').innerHTML = chip('error', 'server unreachable', 'red');
  }
}
function chip(label, text, tone) {
  return `<span class="chip"><b>${esc(label)}</b><i class="${tone || ''}">${esc(text)}</i></span>`;
}

// ---- examples -------------------------------------------------------------
async function loadExamples() {
  try {
    const ex = await getJSON('/api/examples');
    const box = $('#examples');
    box.innerHTML = ex.map(e =>
      `<button type="button" class="example" data-kind="${esc(e.kind)}">${esc(e.label)}</button>`).join('');
    $$('.example', box).forEach((btn, i) => btn.addEventListener('click', () => fillForm(ex[i])));
  } catch (e) { /* DB may not be warm yet; retry on next status tick */ }
}

function fillForm(ex) {
  const f = $('#submit-form').elements;
  const s = ex.submission;
  f.namedItem('name').value = s.name || '';
  f.namedItem('category').value = s.category || '';
  f.namedItem('address').value = s.address || '';
  f.namedItem('phone').value = s.phone || '';
  f.namedItem('brand').value = s.brand || '';
  f.namedItem('description').value = s.description || '';
  f.namedItem('lat').value = (s.lat != null) ? s.lat : '';
  f.namedItem('lng').value = (s.lng != null) ? s.lng : '';
}

// ---- live submit ----------------------------------------------------------
async function onSubmit(e) {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {
    name: (fd.get('name') || '').trim(),
    category: (fd.get('category') || '').trim(),
    address: (fd.get('address') || '').trim(),
    phone: (fd.get('phone') || '').trim(),
    brand: (fd.get('brand') || '').trim(),
    description: (fd.get('description') || '').trim(),
    lat: (fd.get('lat') || '') === '' ? null : parseFloat(fd.get('lat')),
    lng: (fd.get('lng') || '') === '' ? null : parseFloat(fd.get('lng')),
  };
  if (!payload.name) return;
  setBusy('#submit-btn', '#submit-spinner', true, 'running chain (model loads on first run)…');
  try {
    renderLive(await postJSON('/api/submit', payload));
  } catch (err) {
    showError('#live-result', err);
  } finally {
    setBusy('#submit-btn', '#submit-spinner', false);
  }
}

function setBusy(btnSel, spinSel, on, label) {
  const btn = $(btnSel), spin = $(spinSel);
  btn.disabled = on;
  spin.hidden = !on;
  spin.textContent = on ? (label || '…') : '';
}
function showError(sel, err) {
  const el = $(sel);
  el.hidden = false;
  el.innerHTML = `<div class="error">Request failed: ${esc(err.message)}</div>`;
}

function fmtVal(v) {
  if (v === true) return 'yes';
  if (v === false) return 'no';
  if (v == null) return '—';
  if (typeof v === 'number') return String(Math.round(v * 1000) / 1000);
  return String(v);
}

function signalList(signals, title) {
  if (!signals || !signals.length) return '';
  const bars = signals.map(s => {
    const w = Math.max(0, Math.min(100, Math.round((s.weight ?? 0) * 100)));
    const val = fmtVal(s.value) + (s.unit ? ' ' + esc(s.unit) : '');
    return `<span class="sig">
      <span class="sig-label">${esc(s.label)} <i>${val}</i></span>
      <span class="sig-bar"><span class="sig-fill" style="width:${w}%"></span></span>
    </span>`;
  }).join('');
  return `<div class="signals">${title ? `<span class="signals-title">${esc(title)}</span>` : ''}${bars}</div>`;
}

function signalSummary(signals) {
  if (!signals || !signals.length) return '—';
  return signals.slice(0, 3).map(s => `${s.key} ${fmtVal(s.value)}${s.unit || ''}`).join(' · ');
}

function renderLive(res) {
  const m = res.moderation, r = res.resolve, route = res.route;
  const cands = r.candidates || [];
  const candTable = cands.length
    ? `<table class="cands">
        <thead><tr><th>Candidate (existing DB)</th><th>Dist</th><th>Signals</th><th>Judge</th><th>Conf</th><th>Reason</th></tr></thead>
        <tbody>${cands.map(c => {
          const j = c.judge || {};
          const same = j.same;
          const st = same === true ? 'blue' : same === false ? 'gray' : 'amber';
          const sLabel = same === true ? 'same' : same === false ? 'different' : '?';
          const dist = c.distance_km == null ? '—' : c.distance_km.toFixed(3) + ' km';
          return `<tr>
            <td>${esc((c.poi && c.poi.name) || '(missing)')}</td>
            <td>${dist}</td>
            <td class="reason">${signalSummary(c.signals)}</td>
            <td>${badge(sLabel, st)}</td>
            <td>${j.confidence == null ? '—' : Math.round(j.confidence * 100) + '%'}</td>
            <td class="reason">${esc(j.reason || '')}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>`
    : `<div class="empty">No candidate in the blocking shortlist — nothing nearby or same-brand to compare.</div>`;

  const matched = r.matched ? esc((r.matched.name || r.matched.id || '?')) : null;

  $('#live-result').hidden = false;
  $('#live-result').innerHTML = `
    <div class="pipeline">
      <div class="step">
        <div class="step-head"><span class="num">1</span><h3>Content gate <span class="engine">Laya · System 1</span></h3></div>
        <div class="step-body">
          ${badge(m.verdict, VERD_TONE[m.verdict] || 'gray')} ${confBar(m.confidence, VERD_TONE[m.verdict] || 'gray')}
          <div class="meta">${m.reason ? 'flag: ' + esc(m.reason) : 'no violation flag'} · route ${esc(m.route)}</div>
          ${signalList(m.signals, 'violation signals (p)')}
        </div>
      </div>
      <div class="step">
        <div class="step-head"><span class="num">2</span><h3>Dedup <span class="engine">${r.backend === 'llm' ? 'DeepSeek · System 2' : 'rules · offline'}</span></h3></div>
        <div class="step-body">
          ${badge(r.verdict, VERD_TONE[r.verdict] || 'gray')}
          ${r.verdict === 'merge' ? `<div class="meta">merge with <b>${matched}</b></div>` : ''}
          ${r.verdict !== 'merge' ? `<div class="meta">${esc(r.reason || '')}</div>` : ''}
          ${r.verdict === 'merge' ? `<div class="attr"><div class="attr-title">why same — evidence</div>${signalList(r.signals)}</div>` : ''}
          ${candTable}
        </div>
      </div>
      <div class="step">
        <div class="step-head"><span class="num">3</span><h3>Route <span class="engine">confidence-gated</span></h3></div>
        <div class="step-body">
          ${badge(route.action, TONE[route.action] || 'gray')}
          <div class="meta">${esc(route.why || '')}</div>
        </div>
      </div>
    </div>`;
}

// ---- persistent store (audit log) ----------------------------------------
async function loadAudit() {
  setBusy('#audit-btn', '#audit-spinner', true);
  try {
    renderAudit(await getJSON('/api/audit?limit=50'));
  } catch (err) {
    showError('#audit-result', err);
  } finally {
    setBusy('#audit-btn', '#audit-spinner', false);
  }
}

function renderAudit(rows) {
  const el = $('#audit-result');
  el.hidden = false;
  if (!rows.length) {
    el.innerHTML = '<div class="empty">No decisions persisted yet. Submit a listing above to write the first audit row.</div>';
    return;
  }
  el.innerHTML = `
    <table class="audit">
      <thead><tr><th>time</th><th>name</th><th>moderation</th><th>dedup</th><th>action</th><th>reason</th></tr></thead>
      <tbody>${rows.map(r => `<tr>
        <td class="dim">${esc(r.ts)}</td>
        <td class="name">${esc(r.name || '?')}</td>
        <td>${badge(r.mod_verdict, VERD_TONE[r.mod_verdict] || 'gray')}</td>
        <td>${badge(r.dedup_verdict, VERD_TONE[r.dedup_verdict] || 'gray')}${r.matched_place ? ' <span class="dim">→ ' + esc(r.matched_place) + '</span>' : ''}</td>
        <td>${badge(r.action, TONE[r.action] || 'gray')}</td>
        <td class="reason">${esc(r.reason || '')}</td>
      </tr>`).join('')}</tbody>
    </table>`;
}

// ---- boot -----------------------------------------------------------------
$('#submit-form').addEventListener('submit', onSubmit);
$('#audit-btn').addEventListener('click', loadAudit);
refreshStatus();
loadExamples();
loadAudit();
setInterval(refreshStatus, 5000);
setInterval(loadExamples, 8000);
