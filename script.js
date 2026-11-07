// ---------- pure helpers (no DOM) ----------
function parseCSV(text) {
  const lines = text.replace(/\r/g, '').split('\n').filter(l => l.trim() !== '');
  if (lines.length < 2) return { cols: [], rows: [] };
  const head = lines[0];
  const delim = [',', ';', '\t'].map(d => [d, head.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
  const split = line => {
    const out = []; let cur = '', q = false;
    for (const ch of line) {
      if (ch === '"') q = !q;
      else if (ch === delim && !q) { out.push(cur.trim()); cur = ''; }
      else cur += ch;
    }
    out.push(cur.trim()); return out;
  };
  const cols = split(head);
  const rows = lines.slice(1).map(split).filter(r => r.length === cols.length);
  return { cols, rows };
}
function numericColumns(cols, rows) {
  return cols.filter((c, j) => rows.filter(r => r[j] !== '' && !isNaN(+r[j])).length >= 0.9 * rows.length);
}
function category(m, top) {
  const p = m / top * 100;
  return p >= 75 ? 'Excellent' : p >= 60 ? 'Good' : p >= 50 ? 'Average' : 'Needs Improvement';
}
function rng(seed) { let s = seed; return () => (s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296; }
function solve(A, b) { // Gaussian elimination with partial pivoting
  const n = b.length;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    for (let r = i + 1; r < n; r++) {
      const f = A[r][i] / A[i][i];
      for (let c = i; c < n; c++) A[r][c] -= f * A[i][c];
      b[r] -= f * b[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c];
    x[i] = s / A[i][i];
  }
  return x;
}
function fit(X, y) { // standardised ridge regression
  const n = X.length, k = X[0].length;
  const mean = [], sd = [];
  for (let j = 0; j < k; j++) {
    const m = X.reduce((s, r) => s + r[j], 0) / n;
    const v = X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / n;
    mean.push(m); sd.push(Math.sqrt(v) || 1);
  }
  const ym = y.reduce((s, v) => s + v, 0) / n;
  const Z = X.map(r => r.map((v, j) => (v - mean[j]) / sd[j]));
  const A = Array.from({ length: k }, () => new Array(k).fill(0)), b = new Array(k).fill(0);
  for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) {
    b[a] += Z[i][a] * (y[i] - ym);
    for (let c = 0; c < k; c++) A[a][c] += Z[i][a] * Z[i][c];
  }
  for (let a = 0; a < k; a++) A[a][a] += 1e-6 * n;
  return { w: solve(A, b), mean, sd, ym };
}
function predict(m, x) { return m.ym + x.reduce((s, v, j) => s + m.w[j] * (v - m.mean[j]) / m.sd[j], 0); }
function metrics(m, X, y) {
  const p = X.map(r => predict(m, r)), n = y.length, ym = y.reduce((s, v) => s + v, 0) / n;
  const mae = p.reduce((s, v, i) => s + Math.abs(v - y[i]), 0) / n;
  const mse = p.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / n;
  const ssTot = y.reduce((s, v) => s + (v - ym) ** 2, 0);
  return { mae, rmse: Math.sqrt(mse), r2: 1 - (mse * n) / ssTot };
}
if (typeof module !== 'undefined') module.exports = { parseCSV, numericColumns, fit, predict, metrics, category };

// ---------- page logic ----------
if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  let data = null, model = null, feats = [], tgt = '', ranges = {};
  $('file').addEventListener('change', e => {
    const f = e.target.files[0]; if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {
      data = parseCSV(rd.result);
      const nums = numericColumns(data.cols, data.rows);
      if (data.rows.length < 30 || nums.length < 2) { $('info').textContent = 'Need 30+ rows and at least 2 numeric columns.'; return; }
      data.nums = nums;
      $('info').textContent = `${f.name}: ${data.rows.length} rows, ${data.cols.length} columns.`;
      $('target').innerHTML = nums.map(c => `<option>${c}</option>`).join('');
      const hint = nums.find(c => ['final_marks', 'g3', 'final', 'marks', 'score'].includes(c.toLowerCase()));
      $('target').value = hint || nums[nums.length - 1];
      $('top').value = 100;
      drawChips(); $('setup').classList.remove('hidden');
    };
    rd.readAsText(f);
  });
  function drawChips() {
    const t = $('target').value;
    $('features').innerHTML = data.nums.filter(c => c !== t)
      .map(c => `<label><input type="checkbox" value="${c}" checked> ${c}</label>`).join('');
  }
  $('target').addEventListener('change', drawChips);
  $('train').addEventListener('click', () => {
    tgt = $('target').value;
    feats = [...document.querySelectorAll('#features input:checked')].map(c => c.value);
    if (!feats.length) { alert('Select at least one feature.'); return; }
    const ti = data.cols.indexOf(tgt), fi = feats.map(f => data.cols.indexOf(f));
    const rows = data.rows.filter(r => [ti, ...fi].every(i => r[i] !== '' && !isNaN(+r[i])));
    const rand = rng(42), idx = rows.map((_, i) => i).sort(() => rand() - 0.5);
    const cut = Math.floor(rows.length * 0.8);
    const pick = ids => ({ X: ids.map(i => fi.map(j => +rows[i][j])), y: ids.map(i => +rows[i][ti]) });
    const tr = pick(idx.slice(0, cut)), te = pick(idx.slice(cut));
    model = fit(tr.X, tr.y);
    const m = metrics(model, te.X, te.y);
    $('metrics').innerHTML = [['MAE', m.mae.toFixed(3)], ['RMSE', m.rmse.toFixed(3)], ['R²', m.r2.toFixed(3)]]
      .map(a => `<div class="metric"><b>${a[1]}</b><span>${a[0]}</span></div>`).join('');
    const imp = feats.map((f, j) => [f, Math.abs(model.w[j])]).sort((a, b) => b[1] - a[1]);
    const mx = imp[0][1] || 1;
    $('bars').innerHTML = imp.map(a => `<div class="row"><span class="name">${a[0]}</span><div class="track"><div style="width:${a[1] / mx * 100}%"></div></div></div>`).join('');
    $('split').textContent = `Training rows: ${tr.y.length} | Test rows: ${te.y.length} | Model: Linear Regression`;
    ranges = {};
    $('form').innerHTML = feats.map((f, j) => {
      const v = rows.map(r => +r[fi[j]]).sort((a, b) => a - b);
      const lo = v[0], hi = v[v.length - 1], med = v[Math.floor(v.length / 2)];
      ranges[f] = [lo, hi];
      return `<label>${f} (${lo}–${hi})<input type="number" id="f${j}" value="${med}" min="${lo}" max="${hi}" step="any"></label>`;
    }).join('');
    ['results', 'predict'].forEach(i => $(i).classList.remove('hidden'));
    $('out').classList.add('hidden');
  });
  $('go').addEventListener('click', () => {
    const top = +$('top').value || 100;
    const x = feats.map((f, j) => +$('f' + j).value);
    const p = Math.min(top, Math.max(0, predict(model, x)));
    $('pred').textContent = `${p.toFixed(1)} / ${top}`;
    $('cat').textContent = category(p, top);
    $('out').classList.remove('hidden');
    $('fill').style.width = Math.min(100, p / top * 100) + '%';
  });
}
