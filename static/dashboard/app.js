// ---------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "research") loadResearch();
    if (btn.dataset.tab === "training") loadJobs();
  });
});

// ---------------------------------------------------------------------
// Playground
// ---------------------------------------------------------------------
const pgFile = document.getElementById("pg-file");
const pgCanvas = document.getElementById("pg-canvas");
const pgPreviewWrap = document.getElementById("pg-preview-wrap");
let pgImage = null;

pgFile.addEventListener("change", () => {
  const file = pgFile.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    pgImage = img;
    drawCanvas([]);
    pgPreviewWrap.hidden = false;
  };
  img.src = url;
});

function drawCanvas(regions) {
  if (!pgImage) return;
  const ctx = pgCanvas.getContext("2d");
  pgCanvas.width = pgImage.naturalWidth;
  pgCanvas.height = pgImage.naturalHeight;
  ctx.drawImage(pgImage, 0, 0);
  ctx.lineWidth = Math.max(2, pgImage.naturalWidth / 400);
  ctx.strokeStyle = "#5b8cff";
  ctx.font = `${Math.max(14, pgImage.naturalWidth / 60)}px sans-serif`;
  regions.forEach((region, index) => {
    const points = region.polygon;
    if (!points || points.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "#5b8cff";
    ctx.fillText(String(index + 1), points[0].x + 2, Math.max(12, points[0].y - 4));
  });
}

function engineBase() {
  const value = document.getElementById("pg-engine-url").value.trim();
  return value ? value.replace(/\/$/, "") : "";
}

document.getElementById("playground-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = pgFile.files[0];
  if (!file) return;

  const submitBtn = document.getElementById("pg-submit");
  const statusEl = document.getElementById("pg-status");
  submitBtn.disabled = true;
  statusEl.textContent = "요청 중... (모델 첫 로드 시 오래 걸릴 수 있습니다)";

  const params = new URLSearchParams({
    source_language: document.getElementById("pg-source").value,
    target_language: document.getElementById("pg-target").value,
    quality: document.getElementById("pg-quality").value,
    ocr_mode: document.getElementById("pg-mode").value,
    inpaint: document.getElementById("pg-inpaint").checked ? "true" : "false",
    inpaint_method: document.getElementById("pg-inpaint-method").value,
    skip_translate: document.getElementById("pg-skip-translate").checked ? "true" : "false",
  });

  const formData = new FormData();
  formData.append("file", file);

  const started = performance.now();
  try {
    const response = await fetch(`${engineBase()}/v1/images/translate?${params}`, {
      method: "POST",
      body: formData,
    });
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    if (!response.ok) {
      const detail = await response.text();
      statusEl.textContent = `실패 (${response.status}): ${detail}`;
      return;
    }
    const data = await response.json();
    statusEl.textContent = `완료 (${elapsed}s), source_language=${data.source_language}, regions=${data.regions.length}`;
    drawCanvas(data.regions);
    renderResults(data);
  } catch (error) {
    statusEl.textContent = `오류: ${error}. Engine base URL을 확인하세요 (CORS/네트워크).`;
  } finally {
    submitBtn.disabled = false;
  }
});

function renderResults(data) {
  const container = document.getElementById("pg-results");
  container.innerHTML = "";

  if (data.inpainted_image) {
    const img = document.createElement("img");
    img.src = data.inpainted_image;
    img.className = "result-image";
    container.appendChild(img);
  }

  if (!data.regions.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "검출된 텍스트 영역이 없습니다.";
    container.appendChild(empty);
    return;
  }

  data.regions.forEach((region, index) => {
    const card = document.createElement("div");
    card.className = "region-card";
    card.innerHTML = `
      <span class="conf">#${index + 1} conf=${region.confidence.toFixed(2)}</span>
      <div class="src">${escapeHtml(region.text || "(빈 텍스트)")}</div>
      <div class="dst">${escapeHtml(region.translated_text || "")}</div>
    `;
    container.appendChild(card);
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------------------------------------------------------------------
// Research
// ---------------------------------------------------------------------
let chartModes = null;
let chartCurve = null;

async function loadResearch() {
  const response = await fetch("/api/dashboard/summary");
  const data = await response.json();

  renderModesChart(data.manga109s_cascade);
  renderCurveChart(data.detector_curve);
  document.getElementById("train-report").textContent = data.detector_train_report
    ? JSON.stringify(data.detector_train_report, null, 2)
    : "아직 학습 리포트가 없습니다 (train_text_detector.py 실행 전).";
  document.getElementById("router-summary").textContent = JSON.stringify(
    { ocr_router: data.ocr_router, eval_summary: data.eval_summary },
    null,
    2
  );
  renderCatalog(data.catalog);
}

function renderModesChart(cascade) {
  const wrap = document.getElementById("modes-table");
  if (!cascade || !cascade.modes) {
    wrap.textContent = "아직 벤치마크 결과가 없습니다 (benchmark_manga109s.py 실행 전).";
    if (chartModes) { chartModes.destroy(); chartModes = null; }
    return;
  }
  const modes = Object.keys(cascade.modes);
  const recall = modes.map((m) => cascade.modes[m].recall);
  const precision = modes.map((m) => cascade.modes[m].precision);
  const cer = modes.map((m) => cascade.modes[m].cer ?? 0);

  if (chartModes) chartModes.destroy();
  chartModes = new Chart(document.getElementById("chart-modes"), {
    type: "bar",
    data: {
      labels: modes,
      datasets: [
        { label: "recall@0.5", data: recall, backgroundColor: "#5b8cff" },
        { label: "precision", data: precision, backgroundColor: "#33c17a" },
        { label: "CER", data: cer, backgroundColor: "#e0a53a" },
      ],
    },
    options: {
      scales: { y: { beginAtZero: true, max: 1 } },
      plugins: { legend: { labels: { color: "#e6e9ee" } } },
    },
  });

  let rows = "<table><tr><th>mode</th><th>recall</th><th>precision</th><th>CER</th><th>latency(s)</th></tr>";
  modes.forEach((m) => {
    const row = cascade.modes[m];
    rows += `<tr><td>${m}</td><td>${row.recall.toFixed(3)}</td><td>${row.precision.toFixed(3)}</td><td>${(row.cer ?? 0).toFixed(3)}</td><td>${row.latency_sec.toFixed(1)}</td></tr>`;
  });
  rows += "</table>";
  wrap.innerHTML = rows;
}

function renderCurveChart(rows) {
  if (chartCurve) { chartCurve.destroy(); chartCurve = null; }
  if (!rows || !rows.length) {
    return;
  }
  const epochs = rows.map((r) => r.epoch);
  const precision = rows.map((r) => Number(r["metrics/precision(B)"]));
  const recall = rows.map((r) => Number(r["metrics/recall(B)"]));
  const map50 = rows.map((r) => Number(r["metrics/mAP50(B)"]));

  chartCurve = new Chart(document.getElementById("chart-curve"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        { label: "precision", data: precision, borderColor: "#5b8cff", fill: false },
        { label: "recall", data: recall, borderColor: "#33c17a", fill: false },
        { label: "mAP50", data: map50, borderColor: "#e0a53a", fill: false },
      ],
    },
    options: {
      scales: { y: { beginAtZero: true, max: 1 } },
      plugins: { legend: { labels: { color: "#e6e9ee" } } },
    },
  });
}

function renderCatalog(catalog) {
  const wrap = document.getElementById("catalog-table");
  if (!catalog) {
    wrap.textContent = "catalog.json 없음";
    return;
  }
  let rows = "<table><tr><th>id</th><th>language</th><th>license</th><th>access</th></tr>";
  catalog.forEach((item) => {
    rows += `<tr><td>${item.id}</td><td>${(item.language || []).join(", ")}</td><td>${item.license}</td><td>${item.access}</td></tr>`;
  });
  rows += "</table>";
  wrap.innerHTML = rows;
}

document.getElementById("research-refresh").addEventListener("click", loadResearch);

// ---------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------
let selectedJobId = null;
let jobPollTimer = null;

document.getElementById("train-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const script = document.getElementById("train-script").value;
  const argsText = document.getElementById("train-args").value.trim();
  const args = argsText ? argsText.split(/\s+/) : [];
  const statusEl = document.getElementById("train-status");

  try {
    const response = await fetch("/api/dashboard/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ script, args }),
    });
    const data = await response.json();
    if (!response.ok) {
      statusEl.textContent = `거부됨: ${data.detail}`;
      return;
    }
    statusEl.textContent = `시작됨: job_id=${data.job_id}`;
    selectedJobId = data.job_id;
    loadJobs();
    pollSelectedJob();
  } catch (error) {
    statusEl.textContent = `오류: ${error}`;
  }
});

async function loadJobs() {
  const response = await fetch("/api/dashboard/train");
  const jobs = await response.json();
  const wrap = document.getElementById("jobs-list");
  if (!jobs.length) {
    wrap.innerHTML = '<p class="hint">아직 실행한 작업이 없습니다.</p>';
    return;
  }
  wrap.innerHTML = "";
  jobs.forEach((job) => {
    const row = document.createElement("div");
    row.className = "job-row";
    const started = new Date(job.started_at * 1000).toLocaleTimeString();
    row.innerHTML = `
      <span>${job.script} <span class="hint">${job.id} · ${started}</span></span>
      <span class="badge ${job.status}">${job.status}</span>
    `;
    row.addEventListener("click", () => {
      selectedJobId = job.id;
      pollSelectedJob();
    });
    wrap.appendChild(row);
  });
  if (jobs.some((j) => j.status === "running")) {
    clearTimeout(jobPollTimer);
    jobPollTimer = setTimeout(loadJobs, 4000);
  }
}

async function pollSelectedJob() {
  if (!selectedJobId) return;
  const response = await fetch(`/api/dashboard/train/${selectedJobId}`);
  if (!response.ok) return;
  const job = await response.json();
  const logEl = document.getElementById("job-log");
  logEl.textContent = job.log || "(로그 없음)";
  logEl.scrollTop = logEl.scrollHeight;
  if (job.status === "running") {
    setTimeout(pollSelectedJob, 2500);
  }
}
