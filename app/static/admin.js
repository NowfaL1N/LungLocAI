const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (path, opts = {}) => {
  const r = await fetch("api/admin/" + path, { credentials: "same-origin", ...opts });
  if (r.status === 401) { showLogin(); throw new Error("login"); }
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body.detail || body || r.status);
  return body;
};
const LABELS = [["malignant", "Malignant"], ["benign", "Benign"], ["not_nodule", "Not a nodule"], ["unsure", "Unsure"]];
let current = null;

function showLogin(msg) { $("#dash").classList.add("hidden"); $("#logout").classList.add("hidden"); $("#loginView").classList.remove("hidden"); if (msg) $("#loginErr").textContent = msg; }
$("#loginBtn").onclick = async () => {
  $("#loginErr").textContent = "Checking…";
  const r = await fetch("api/admin/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: $("#pw").value }) });
  if (!r.ok) { $("#loginErr").textContent = (await r.json()).detail || "Login failed"; return; }
  $("#pw").value = ""; $("#loginErr").textContent = ""; load();
};
$("#pw").addEventListener("keydown", e => { if (e.key === "Enter") $("#loginBtn").click(); });
$("#logout").onclick = async () => { await fetch("api/admin/logout", { method: "POST" }); showLogin(); };

async function load() {
  let s;
  try { s = await api("status"); } catch (e) { if (e.message !== "login") showLogin(e.message); return; }
  $("#loginView").classList.add("hidden"); $("#dash").classList.remove("hidden"); $("#logout").classList.remove("hidden");
  const st = s.store, lc = s.label_counts || {};
  const models = Object.entries(s.models).map(([k, m]) => `${k.replace("_model", "")}: AUC ${m.metrics?.auc ?? "–"}${m.demo ? " (demo)" : ""}`).join("<br>");
  $("#stats").innerHTML = [
    [st.configured ? `<span class="pill ${st.private === false ? "bad" : "ok"}">${st.private === false ? "PUBLIC!" : "private"}</span> ${esc(st.repo || st.backend)}` : '<span class="pill bad">not configured</span>', "storage"],
    [s.n_cases ?? "–", "stored cases"],
    [`${s.n_labels ?? 0}`, `labels · ${lc.malignant || 0} malignant · ${lc.benign || 0} benign · ${lc.not_nodule || 0} not nodule`],
    [s.base_table ? '<span class="pill ok">uploaded</span>' : '<span class="pill warn">none yet</span>', "base training table"],
    [`<span style="font-size:14px">${models || "none"}</span>`, "current models"],
  ].map(([v, k]) => `<div class="stat"><div class="v" style="font-size:20px">${v}</div><div class="k">${k}</div></div>`).join("");
  $("#storeErr").innerHTML = st.error ? `<div class="notice warn">Storage problem: ${esc(st.error)}</div>` : "";
  if (!st.configured) return;
  const cases = await api("cases");
  $("#cases").innerHTML = "<thead><tr><th>Stored</th><th>Case</th><th>Scan</th><th>Nodules</th><th>Suspicious</th><th>Max cancer prob.</th><th>Labelled</th></tr></thead><tbody>" +
    (cases.length ? cases.map(c => `<tr data-id="${esc(c.case_id)}"><td>${esc(c.stored_at)}</td><td class="mono">${esc(c.case_id)}</td><td>${esc(c.scan_type)}</td><td>${esc(c.n_nodules)}</td><td>${esc(c.n_suspicious)}</td><td>${c.max_p_malignant !== "" ? Math.round(c.max_p_malignant * 100) + "%" : "–"}</td><td>${c.labelled}/${esc(c.n_nodules)}</td></tr>`).join("")
      : `<tr class="muted-row"><td colspan="7">No stored cases yet: they appear here when an uploader ticks the consent box.</td></tr>`) + "</tbody>";
  document.querySelectorAll("#cases tbody tr[data-id]").forEach(tr => tr.onclick = () => openCase(tr.dataset.id));
  retrainStatus();
}

async function openCase(id) {
  current = await api("cases/" + encodeURIComponent(id));
  $("#caseCard").classList.remove("hidden");
  $("#caseTitle").textContent = `Case ${id} · ${current.lungs?.full_body ? "full-body" : "chest"} CT · ${current.nodules.length} nodules`;
  $("#casePdf").href = `api/admin/cases/${id}/files/report.pdf`;
  $("#caseCsv").href = `api/admin/cases/${id}/files/table.csv`;
  $("#caseNodules").innerHTML = current.nodules.map(n => `
    <div class="nodule ${esc(n.ai_risk)}" id="n${n.candidate_id}">
      <img src="api/admin/cases/${id}/files/nodule_${n.candidate_id}.png" alt="">
      <div><h4>#${n.display_id} · ${esc(n.side)} lung, ${esc(n.zone).toLowerCase()} zone · ${n.diameter_mm.toFixed(1)} mm ${esc(n.nodule_type)}</h4>
        <div><span class="tag ${esc(n.ai_risk)}">AI: ${esc(n.ai_label)}</span>${n.p_malignant != null ? `<span class="tag">cancer prob ${(n.p_malignant * 100).toFixed(0)}%</span>` : ""}<span class="tag">Lung-RADS ${esc(n.lungrads)}</span></div>
        <div class="kv"><div><span>Mean density</span>${Math.round(n.mean_hu)} HU</div><div><span>Sphericity</span>${n.sphericity.toFixed(2)}</div><div><span>Margin irregularity</span>${n.margin_irregularity.toFixed(2)}</div></div>
        <div class="labels" data-cid="${n.candidate_id}">${LABELS.map(([v, t]) => `<button class="${v} ${n.owner_label === v ? "on" : ""}" data-v="${v}">${t}</button>`).join("")}
          <button data-v="" title="remove label">✕</button></div></div></div>`).join("") || "<p>No nodules reported in this case.</p>";
  document.querySelectorAll(".labels button").forEach(b => b.onclick = () => setLabel(id, b.parentElement.dataset.cid, b.dataset.v, b.parentElement));
  $("#caseCard").scrollIntoView({ behavior: "smooth" });
}

async function setLabel(caseId, cid, label, box) {
  box.querySelectorAll("button").forEach(b => b.disabled = true);
  try {
    await api("label", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ case_id: caseId, candidate_id: +cid, label }) });
    box.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === label && label !== ""));
  } catch (e) { alert("Could not save label: " + e.message); }
  box.querySelectorAll("button").forEach(b => b.disabled = false);
}

$("#selftestBtn").onclick = async () => {
  $("#selftestMsg").textContent = "Writing a test file…";
  try { const r = await api("selftest", { method: "POST" }); $("#selftestMsg").innerHTML = r.ok ? `<span class="pill ok">OK</span> wrote and read back a file in ${esc(r.repo || r.backend)} (${r.seconds}s)${r.private ? " · repository is private" : ""}` : '<span class="pill bad">read-back mismatch</span>'; }
  catch (e) { $("#selftestMsg").innerHTML = `<span class="pill bad">failed</span> ${esc(e.message)}`; }
};
$("#retrainBtn").onclick = async () => {
  if (!confirm("Retrain the XGBoost models now? The current model stays unless the new one is at least as good.")) return;
  try { await api("retrain", { method: "POST" }); } catch (e) { alert(e.message); }
  retrainStatus();
};
let rt = null;
async function retrainStatus() {
  const r = await api("retrain");
  $("#retrainState").textContent = { idle: "", running: "running…", done: "finished", error: "failed" }[r.state] || r.state;
  $("#retrainBtn").disabled = r.state === "running";
  $("#retrainLog").classList.toggle("hidden", !r.log.length);
  $("#retrainLog").textContent = r.log.join("\n");
  $("#retrainLog").scrollTop = 1e9;
  if (r.report) {
    const rows = Object.entries(r.report.tasks).map(([t, x]) => `<tr><td><b>${t}</b></td><td>${esc(x.status)}</td><td>${x.candidate_auc ?? "–"}</td><td>${x.current_auc ?? "–"}</td><td>${x.positives ?? "–"} / ${x.negatives ?? "–"}</td></tr>`).join("");
    $("#retrainReport").innerHTML = `<div class="table-wrap"><table><tr><th>Model</th><th>Result</th><th>New AUC</th><th>Old AUC</th><th>pos / neg</th></tr>${rows}</table></div>
      <p class="small muted">Data: ${r.report.data.base_rows} base rows + ${r.report.data.upload_rows} labelled rows from ${r.report.data.upload_cases} uploaded scans.</p>`;
  }
  clearTimeout(rt);
  if (r.state === "running") rt = setTimeout(retrainStatus, 2000);
  else if (r.state === "done" && r.report) { /* refresh model badges */ api("status").then(() => {}); }
}

$("#baseBtn").onclick = async () => {
  const f = $("#baseFile").files[0];
  if (!f) return;
  $("#baseMsg").textContent = "Uploading…";
  const fd = new FormData(); fd.append("file", f, f.name);
  try { const r = await api("base-table", { method: "POST", body: fd }); $("#baseMsg").textContent = `Saved (${r.features} feature columns, labels: ${r.labels.join(", ")})`; load(); }
  catch (e) { $("#baseMsg").textContent = e.message; }
};

load();
