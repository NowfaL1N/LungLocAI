const $ = (s) => document.querySelector(s);
const show = (id) => ["uploadView", "progressView", "errorView", "resultView"].forEach(v => $("#" + v).classList.toggle("hidden", v !== id));
let files = [], job = null, res = null, axis = "axial";

/* ---------------- model badge ---------------- */
fetch("api/model").then(r => r.json()).then(m => {
  const b = $("#modelBadge");
  const det = Object.values(m.details);
  if (!det.length) { b.textContent = "No XGBoost model loaded"; b.classList.add("demo"); return; }
  const demo = det.some(d => d.demo);
  const mal = m.details.malignancy_model;
  b.textContent = (demo ? "DEMO model · " : "XGBoost · ") + (mal ? `cancer AUC ${mal.metrics?.auc != null ? mal.metrics.auc.toFixed(2) : "–"}` : "nodule detection only");
  if (demo) b.classList.add("demo");
});

fetch("api/config").then(r => r.json()).then(c => {
  $("#ttl").textContent = c.job_ttl_hours;
  if (c.research_storage) $("#consentBox").classList.remove("hidden");
}).catch(() => {});

/* ---------------- upload ---------------- */
const drop = $("#drop"), input = $("#fileInput");
const setFiles = (fl) => {
  files = [...fl];
  $("#fileList").innerHTML = files.map(f => `${f.name} <span class="muted">(${(f.size / 1e6).toFixed(1)} MB)</span>`).join("<br>");
  const names = files.map(f => f.name.toLowerCase());
  const mhd = names.some(n => n.endsWith(".mhd")), raw = names.some(n => n.endsWith(".raw") || n.endsWith(".zraw"));
  $("#analyzeBtn").disabled = !files.length || (mhd && !raw);
  if (mhd && !raw) $("#fileList").innerHTML += `<br><span style="color:#a4161a">Also select the matching .raw file.</span>`;
};
input.addEventListener("change", () => setFiles(input.files));
["dragenter", "dragover"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => setFiles(ev.dataTransfer.files));

$("#analyzeBtn").onclick = async () => {
  const fd = new FormData();
  files.forEach(f => fd.append("files", f, f.name));
  fd.append("consent", $("#consent").checked ? "1" : "0");
  show("progressView"); renderSteps({ step: "load", steps: STEPS_DEFAULT, message: "Uploading…" });
  const r = await fetch("api/analyze", { method: "POST", body: fd });
  if (!r.ok) { let m = "Upload failed (" + r.status + ")"; try { m = (await r.json()).detail || m; } catch (e) {} return fail(m); }
  start((await r.json()).job_id);
};
document.querySelectorAll("[data-demo]").forEach(b => b.onclick = async () => {
  show("progressView"); renderSteps({ step: "load", steps: STEPS_DEFAULT, message: "Starting sample…" });
  const r = await fetch("api/demo?kind=" + b.dataset.demo, { method: "POST" });
  if (!r.ok) return fail((await r.json()).detail || "Sample not available");
  start((await r.json()).job_id);
});

/* ---------------- progress ---------------- */
const STEPS_DEFAULT = [["load", "Reading CT"], ["lungs", "3-D volume + lung extraction"], ["candidates", "Finding nodules"],
  ["features", "Building feature table"], ["xgboost", "XGBoost classification"], ["report", "Writing report"]];
function renderSteps(s) {
  const steps = s.steps || STEPS_DEFAULT, cur = steps.findIndex(x => x[0] === s.step);
  $("#steps").innerHTML = steps.map((x, i) =>
    `<li class="${s.state === "done" || i < cur ? "done" : i === cur ? "active" : ""}"><span class="ic"></span>${x[1]}</li>`).join("");
  $("#progressMsg").textContent = s.message || "";
}
function fail(msg) { $("#errorMsg").textContent = msg; show("errorView"); }
function start(id) { job = id; location.hash = "job=" + id; poll(); }
async function poll() {
  const r = await fetch("api/jobs/" + job);
  if (!r.ok) return fail("This result is no longer available. Results are deleted automatically a few hours after analysis — please upload the scan again.");
  const s = await r.json();
  if (s.state === "error") return fail(s.message);
  if (s.state === "done") return showResult(s.result);
  show("progressView"); renderSteps(s);
  setTimeout(poll, 900);
}

/* ---------------- results ---------------- */
const pct = (p) => p == null ? "–" : (p * 100).toFixed(0) + "%";
function showResult(r) {
  res = r; show("resultView");
  const L = r.lungs, nods = r.nodules, hasP = nods.some(n => n.p_malignant != null);
  $("#resTitle").textContent = nods.length ? `${nods.length} lung nodule${nods.length > 1 ? "s" : ""} found` : "No nodules found";
  $("#resSub").textContent = `${r.scan.source_file} · ${r.scan.format} · ${L.full_body ? "full-body scan, lungs auto-extracted" : "chest CT"} · analysed in ${r.seconds}s`;
  $("#pdfBtn").href = `api/jobs/${r.job_id}/report.pdf`;
  $("#csvBtn").href = `api/jobs/${r.job_id}/table.csv`;
  $("#demoWarn").classList.toggle("hidden", !r.models.demo);
  $("#warnings").innerHTML = r.warnings.map(w => `<div class="notice info">${w}</div>`).join("") +
    (r.consent ? `<div class="notice info">Thank you: the anonymised findings of this scan were shared privately with the LungLocAI researchers.</div>` : "");

  const sus = nods.filter(n => (n.ai_label || "").startsWith("Suspicious")).length;
  const maxP = hasP ? Math.max(...nods.map(n => n.p_malignant || 0)) : null;
  const stats = [
    [nods.length, "nodules reported"],
    hasP ? [sus, "suspicious for lung cancer", sus > 0] : [r.n_rejected, "candidates rejected"],
    hasP ? [nods.length ? pct(maxP) : "–", "highest cancer probability", maxP >= 0.7] : [nods.length ? Math.max(...nods.map(n => n.diameter_mm)).toFixed(1) + " mm" : "–", "largest nodule"],
    [r.n_candidates, "candidate regions examined"],
    [Math.round(L.lung_volume_ml) + " mL", "lung volume"],
  ];
  $("#stats").innerHTML = stats.map(([v, k, a]) => `<div class="stat ${a ? "alert" : ""}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  $("#impression").innerHTML = r.impression.map(x => `<li>${x}</li>`).join("");

  if (L.full_body) {
    $("#overviewCard").classList.remove("hidden");
    $("#overviewImg").src = `api/jobs/${r.job_id}/overview.png`;
    $("#overviewText").textContent = `The uploaded scan covers ${Math.round(L.scan_length_mm)} mm. The lungs (${Math.round(L.lung_length_mm)} mm, blue) were located automatically and the yellow region was cropped and analysed; abdominal gas and outside air were rejected.`;
  }

  // nodule cards
  $("#nodules").innerHTML = nods.length ? nods.map(n => `
    <div class="nodule ${n.ai_risk}">
      <img src="api/jobs/${r.job_id}/nodule/${n.candidate_id}.png" alt="nodule ${n.display_id}" />
      <div>
        <h4>#${n.display_id} · ${n.side} lung, ${n.zone.toLowerCase()} zone</h4>
        <div><span class="tag ${n.ai_risk}">${n.ai_label}</span><span class="tag">Lung-RADS ${n.lungrads}*</span><span class="tag prio">${n.priority}</span></div>
        ${n.p_malignant != null ? `<div class="prob"><div class="bar"><div class="pin" style="left:${n.p_malignant * 100}%"></div></div>
          <div class="lbl"><span>benign</span><b>AI cancer probability ${(n.p_malignant * 100).toFixed(1)}%</b><span>malignant</span></div></div>` : ""}
        <div class="kv">
          <div><span>Diameter</span>${n.diameter_mm.toFixed(1)} mm</div>
          <div><span>Volume</span>${Math.round(n.volume_mm3)} mm³</div>
          <div><span>Type</span>${n.nodule_type}</div>
          <div><span>Mean density</span>${Math.round(n.mean_hu)} HU</div>
          <div><span>Sphericity</span>${n.sphericity.toFixed(2)}</div>
          <div><span>Margin irregularity</span>${n.margin_irregularity.toFixed(2)}</div>
          <div><span>Distance to pleura</span>${n.pleura_dist_mm.toFixed(1)} mm${n.juxtapleural ? " (juxtapleural)" : ""}</div>
          <div><span>Nodule probability</span>${pct(n.p_nodule)}</div>
          <div><span>Position (mm)</span>${n.world_x.toFixed(0)}, ${n.world_y.toFixed(0)}, ${n.world_z.toFixed(0)}</div>
        </div>
        <div class="mgmt"><b>Suggested next step:</b> ${n.management}</div>
        <button class="btn ghost" style="margin-top:10px" onclick="goTo(${n.candidate_id})">Show in CT viewer</button>
      </div>
    </div>`).join("") + `<p class="fine">*Automatic size/type category in the style of ACR Lung-RADS v2022 — must be confirmed by a radiologist.</p>`
    : `<div class="empty">No nodules passed the detector and XGBoost filters.</div>`;

  // viewer
  setAxis("axial", nods.length ? nods[0].voxel_z : Math.floor(r.shape[0] / 2));
  draw3d();
  loadTable();
  modelInfo();
}

function axisMax(a) { return { axial: res.shape[0], coronal: res.shape[1], sagittal: res.shape[2] }[a] - 1; }
function setAxis(a, idx) {
  axis = a;
  document.querySelectorAll("#axisTabs button").forEach(b => b.classList.toggle("on", b.dataset.axis === a));
  const s = $("#sliceSlider"); s.max = axisMax(a); s.value = idx ?? Math.floor(axisMax(a) / 2);
  drawSlice();
}
let pending = null;
function drawSlice() {
  const s = $("#sliceSlider");
  $("#sliceLabel").textContent = `${axis} ${s.value}/${s.max}`;
  clearTimeout(pending);
  pending = setTimeout(() => $("#sliceImg").src = `api/jobs/${res.job_id}/slice?axis=${axis}&index=${s.value}&lung=${$("#lungToggle").checked ? 1 : 0}`, 40);
}
$("#sliceSlider").addEventListener("input", drawSlice);
$("#lungToggle").addEventListener("change", drawSlice);
document.querySelectorAll("#axisTabs button").forEach(b => b.onclick = () => setAxis(b.dataset.axis));
window.goTo = (cid) => {
  const n = res.nodules.find(x => x.candidate_id === cid);
  const idx = { axial: n.voxel_z, coronal: n.voxel_y, sagittal: n.voxel_x }[axis];
  setAxis(axis, idx);
  $("#sliceImg").scrollIntoView({ behavior: "smooth", block: "center" });
};

const RISK = { High: "#e63946", Intermediate: "#f4a261", Low: "#2a9d8f", "n/a": "#5aa0ff" };
async function draw3d() {
  const sc = await (await fetch(`api/jobs/${res.job_id}/scene`)).json();
  const traces = [];
  if (sc.lung) traces.push({ type: "mesh3d", ...sc.lung, color: "#8fb8ff", opacity: 0.16, name: "Lungs", hoverinfo: "skip", flatshading: true });
  sc.nodules.forEach(n => traces.push({ type: "mesh3d", x: n.x, y: n.y, z: n.z, i: n.i, j: n.j, k: n.k, color: RISK[n.risk] || RISK["n/a"],
    opacity: 1, name: "#" + n.id, hovertemplate: `#${n.id} · ${n.d.toFixed(1)} mm<br>${n.label}${n.p != null ? "<br>cancer prob " + (n.p * 100).toFixed(0) + "%" : ""}<extra></extra>` }));
  // labels
  traces.push({ type: "scatter3d", mode: "text", x: sc.nodules.map(n => Math.max(...n.x) + 6), y: sc.nodules.map(n => n.y[0]),
    z: sc.nodules.map(n => Math.max(...n.z) + 4), text: sc.nodules.map(n => "#" + n.id), textfont: { color: "#fff", size: 13 }, hoverinfo: "skip" });
  const ax = { showgrid: false, zeroline: false, showticklabels: false, title: { text: "" }, showbackground: false, showspikes: false };
  const wait = () => window.Plotly ? Plotly.newPlot("plot3d", traces, {
    paper_bgcolor: "#0b0f16", margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
    scene: { xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: "data",
      camera: { eye: { x: 0, y: -1.9, z: 0.25 }, up: { x: 0, y: 0, z: 1 } } }
  }, { displaylogo: false, responsive: true }) : setTimeout(wait, 200);
  wait();
}

async function loadTable() {
  const txt = await (await fetch(`api/jobs/${res.job_id}/table.csv`)).text();
  const rows = txt.trim().split("\n").map(l => l.split(","));
  const head = rows[0];
  const want = ["display_id", "candidate_id", "side", "zone", "nodule_type", "diameter_mm", "volume_mm3", "sphericity", "elongation",
    "mean_hu", "std_hu", "solid_fraction", "calcified_fraction", "edge_contrast_hu", "margin_irregularity", "pleura_dist_mm",
    "p_nodule", "p_malignant", "ai_label"];
  const cols = want.map(w => head.indexOf(w)).filter(i => i >= 0);
  const fmt = (v) => isNaN(+v) || v === "" ? v : (+v % 1 ? (+v).toFixed(2) : v);
  $("#tableCount").textContent = `(${rows.length - 1} candidate rows × ${res.feature_columns.length} features; grey = rejected as non-nodule)`;
  const isNod = head.indexOf("is_nodule");
  $("#featTable").innerHTML = "<tr>" + cols.map(i => `<th>${head[i]}</th>`).join("") + "</tr>" +
    rows.slice(1).map(r => `<tr class="${r[isNod] === "0" ? "rejected" : ""}">` + cols.map(i => `<td>${fmt(r[i])}</td>`).join("") + "</tr>").join("");
}

function modelInfo() {
  const d = res.models.details;
  $("#modelInfo").innerHTML = Object.keys(d).length ? `<div class="model-grid">` + Object.entries(d).map(([k, m]) => `
    <div class="card"><h3>${k}${m.demo ? " <span class='tag'>DEMO</span>" : ""}</h3>
      <div class="small">Task: <b>${m.task}</b> · label <code>${m.label}</code> · threshold ${m.threshold}</div>
      <div class="small">Trained on ${m.trained_on} — ${m.n_rows} rows, ${m.n_scans} scans (${m.created})</div>
      <div class="small">${m.metrics.cv}: AUC <b>${m.metrics.auc}</b> · sensitivity ${m.metrics.sensitivity} · specificity ${m.metrics.specificity} · F1 ${m.metrics.f1}</div>
      <div class="small muted">Top features: ${(m.top_features || []).slice(0, 6).map(x => x[0]).join(", ")}</div></div>`).join("") + "</div>"
    : "<p>No model loaded.</p>";
}

/* resume a job from the URL */
const m = location.hash.match(/job=(\w+)/);
if (m) start(m[1]);
