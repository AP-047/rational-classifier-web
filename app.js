// app.js

// 1️⃣ Canvas drawing setup
const canvas = document.getElementById('drawCanvas');
const ctx    = canvas.getContext('2d');

// Fill initial background black
ctx.fillStyle   = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// 2️⃣ Set stroke to white
ctx.strokeStyle = 'white';
ctx.lineWidth   = 20;
ctx.lineCap     = 'round';

let drawing = false;

function startDraw(e) {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(
    e.offsetX || e.touches[0].clientX - canvas.getBoundingClientRect().left,
    e.offsetY || e.touches[0].clientY - canvas.getBoundingClientRect().top
  );
}

function draw(e) {
  if (!drawing) return;
  ctx.lineTo(
    e.offsetX || e.touches[0].clientX - canvas.getBoundingClientRect().left,
    e.offsetY || e.touches[0].clientY - canvas.getBoundingClientRect().top
  );
  ctx.stroke();
}

function endDraw() {
  drawing = false;
}

// Mouse events
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseout', endDraw);

// Touch events for mobile
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); });
canvas.addEventListener('touchmove',  (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener('touchend',   (e) => { e.preventDefault(); endDraw(); });

// Clear button
document.getElementById('clearBtn').addEventListener('click', () => {
  // Fill entire canvas black
  ctx.fillStyle   = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // Reset stroke to white
  ctx.strokeStyle = 'white';
  document.getElementById('prediction').innerText = 'Draw a digit and click Predict!';
});



// --------------------------------------------------------------------------------

// 2️⃣ Load artifacts (PCA + classifier JSON)
let pca, rfcModels;

async function loadArtifacts() {
  pca = await fetch('model/trained_pca.json').then(res => res.json());
  rfcModels = await fetch('model/rfc.json').then(res => res.json());
}

function getCanvasImage28() {
  // 1) Capture the full-size drawing in temp canvas
  const temp = document.createElement('canvas');
  temp.width = temp.height = 280;
  const tctx = temp.getContext('2d');
  tctx.drawImage(canvas, 0, 0);

  // 2) Prepare a 28×28 offscreen canvas
  const small = document.createElement('canvas');
  small.width = small.height = 28;
  const sctx  = small.getContext('2d');

  // a) Fill background black
  sctx.fillStyle = 'black';
  sctx.fillRect(0, 0, 28, 28);

  // b) Enable antialiasing and draw scaled-down image
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(temp, 0, 0, 28, 28);

  // c) Extract raw pixels and invert: white strokes → high values
  let imgData = sctx.getImageData(0, 0, 28, 28).data;
  const flatRaw = new Uint8ClampedArray(784);
  for (let i = 0, j = 0; i < imgData.length; i += 4, j++) {
    // Since strokes are white (255) on black (0), no invert needed:
    flatRaw[j] = imgData[i];  
  }

  // d) Compute weighted centroid of bright pixels
  let sumX = 0, sumY = 0, total = 0;
  for (let idx = 0; idx < 784; idx++) {
    const v = flatRaw[idx];
    if (v > 10) {  // threshold noise
      const x = idx % 28, y = Math.floor(idx / 28);
      sumX += x * v;
      sumY += y * v;
      total += v;
    }
  }
  if (total === 0) {
    return new Float32Array(784);  // blank
  }
  const cx = sumX / total, cy = sumY / total;
  const shiftX = Math.round(14 - cx), shiftY = Math.round(14 - cy);

  // 3) Redraw centered on black background
  sctx.fillStyle = 'black';
  sctx.fillRect(0, 0, 28, 28);
  sctx.setTransform(1, 0, 0, 1, shiftX, shiftY);
  sctx.drawImage(temp, 0, 0, 28, 28);
  sctx.setTransform(1, 0, 0, 1, 0, 0);

  // 4) Extract final pixels (white-on-black) as Float32Array
  imgData = sctx.getImageData(0, 0, 28, 28).data;
  const flat = new Float32Array(784);
  for (let i = 0, j = 0; i < imgData.length; i += 4, j++) {
    flat[j] = imgData[i];
  }

  return flat;
}


// 4️⃣ Apply PCA transform (with mean centering)
function applyPCA(flatPixels) {
  const nComp = pca.components.length;
  const out   = new Float32Array(nComp);
  for (let i = 0; i < nComp; i++) {
    let sum = 0;
    for (let j = 0; j < 784; j++) {
      sum += pca.components[i][j] * (flatPixels[j] - pca.mean[j]);
    }
    out[i] = sum;
  }
  return out;
}



// 5️⃣ Predict function (builds G/H and scores)
function predictDigit(pcaVec) {
  let best = { digit: null, score: -Infinity };

  // Helper: generate multi-indices
  function genMulti(n, d) {
    const out = [];
    function recurse(dim, rem, idx=[]) {
      if (dim === n - 1) out.push([...idx, rem]);
      else {
        for (let k = 0; k <= rem; k++) {
          recurse(dim + 1, k, [...idx, rem - k]);
        }
      }
    }
    for (let sum = 0; sum <= d; sum++) recurse(0, sum);
    return out;
  }

  const sample = rfcModels['digit_0'];
  const n      = sample.n_components,
        dn     = sample.degree_n,
        dd     = sample.degree_d;
  const miN = genMulti(n, dn),
        miD = genMulti(n, dd);

  // Build G and H for this sample
  const G = miN.map(idx =>
    idx.reduce((prod,k,i) => prod * Math.pow(pcaVec[i], k), 1)
  );
  const H = miD.map(idx =>
    idx.reduce((prod,k,i) => prod * Math.pow(pcaVec[i], k), 1)
  );

  // Normalize G and H
  const normG = Math.hypot(...G), normH = Math.hypot(...H);
  for (let i = 0; i < G.length; i++) G[i] /= normG;
  for (let i = 0; i < H.length; i++) H[i] /= normH;

  // Score each digit
  for (const key in rfcModels) {
    const digit = Number(key.split('_')[1]);
    const m     = rfcModels[key];
    const num   = m.alpha.reduce((s,v,i) => s + v*G[i], 0);
    const den   = m.beta .reduce((s,v,i) => s + v*H[i], 0);
    const score = den === 0 ? -Infinity : num/den;
    if (score > best.score) best = { digit, score };
  }

  return best.digit;
}

// 6️⃣ Wire up predict button
document.getElementById('predictBtn').addEventListener('click', async () => {
  document.getElementById('prediction').innerText = 'Predicting...';
  if (!pca) await loadArtifacts();

  const flat = getCanvasImage28();      // original call
  const pvec = applyPCA(flat);
  const digit = predictDigit(pvec);

  document.getElementById('prediction').innerText =
    digit !== null ? `Predicted digit: ${digit}` : 'Prediction failed';
});



// Load artifacts on startup
loadArtifacts();