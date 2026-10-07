/* ── DOM references ─────────────────────────────────────────────────────────── */
const dropZone      = document.getElementById('dropZone');
const imageInput    = document.getElementById('imageInput');
const dropLabel     = document.getElementById('dropLabel');
const confSlider    = document.getElementById('conf');
const iouSlider     = document.getElementById('iou');
const imgszSelect   = document.getElementById('imgsz');
const confVal       = document.getElementById('confVal');
const iouVal        = document.getElementById('iouVal');
const predictBtn    = document.getElementById('predictBtn');
const resultsCard   = document.getElementById('resultsCard');
const uploadCard    = document.getElementById('uploadCard');
const annotatedImg  = document.getElementById('annotatedImg');
const downloadLink  = document.getElementById('downloadLink');
const statsBox      = document.getElementById('statsBox');
const detectionList = document.getElementById('detectionList');
const loadingOverlay= document.getElementById('loadingOverlay');
const errorBanner   = document.getElementById('errorBanner');
const resetBtn      = document.getElementById('resetBtn');

let selectedFile = null;

/* ── Slider labels ──────────────────────────────────────────────────────────── */
confSlider.addEventListener('input', () => { confVal.textContent = confSlider.value; });
iouSlider.addEventListener('input',  () => { iouVal.textContent  = iouSlider.value; });

/* ── File selection via click ───────────────────────────────────────────────── */
imageInput.addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

/* ── Drag & drop ────────────────────────────────────────────────────────────── */
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

function setFile(file) {
  selectedFile = file;
  dropLabel.textContent = `✔ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  predictBtn.disabled = false;
  hideError();
}

/* ── Predict ────────────────────────────────────────────────────────────────── */
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  showLoading(true);
  hideError();

  const form = new FormData();
  form.append('image', selectedFile);
  form.append('conf',  confSlider.value);
  form.append('iou',   iouSlider.value);
  form.append('imgsz', imgszSelect.value);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok || !data.success) {
      showError(data.error || `Server error (${res.status})`);
      return;
    }

    renderResults(data);
  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    showLoading(false);
  }
});

/* ── Render results ─────────────────────────────────────────────────────────── */
function renderResults(data) {
  // Annotated image
  const src = `data:image/jpeg;base64,${data.annotated_b64}`;
  annotatedImg.src = src;
  downloadLink.href = src;

  // Summary pills
  statsBox.innerHTML = '';
  if (data.total_detections === 0) {
    statsBox.innerHTML = '<span style="color:var(--muted)">No objects detected. Try lowering the confidence threshold.</span>';
  } else {
    const pill = (label, count) => `<span class="stat-pill">${label}: ${count}</span>`;
    statsBox.innerHTML = pill('Total', data.total_detections);
    for (const [cls, cnt] of Object.entries(data.summary)) {
      statsBox.innerHTML += pill(cls, cnt);
    }
  }

  // Detection list
  detectionList.innerHTML = '';
  if (data.detections.length === 0) {
    detectionList.innerHTML = '<p style="color:var(--muted);font-size:.85rem">—</p>';
  } else {
    data.detections
      .sort((a, b) => b.confidence - a.confidence)
      .forEach(d => {
        const item = document.createElement('div');
        item.className = 'det-item';
        item.innerHTML = `
          <span class="det-class">${d.class_name}</span>
          <span class="det-conf">${(d.confidence * 100).toFixed(1)}%</span>
        `;
        item.title = `BBox: (${d.bbox.x1.toFixed(0)}, ${d.bbox.y1.toFixed(0)}) → (${d.bbox.x2.toFixed(0)}, ${d.bbox.y2.toFixed(0)})`;
        detectionList.appendChild(item);
      });
  }

  // Show results, hide upload
  uploadCard.classList.add('hidden');
  resultsCard.classList.remove('hidden');
}

/* ── Reset ──────────────────────────────────────────────────────────────────── */
resetBtn.addEventListener('click', () => {
  selectedFile = null;
  imageInput.value = '';
  dropLabel.textContent = 'Drop image here';
  predictBtn.disabled = true;
  resultsCard.classList.add('hidden');
  uploadCard.classList.remove('hidden');
  hideError();
});

/* ── Helpers ────────────────────────────────────────────────────────────────── */
function showLoading(on) {
  loadingOverlay.classList.toggle('hidden', !on);
}
function showError(msg) {
  errorBanner.textContent = `⚠ ${msg}`;
  errorBanner.classList.remove('hidden');
}
function hideError() {
  errorBanner.classList.add('hidden');
}
