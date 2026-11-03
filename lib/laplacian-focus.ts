/**
 * Client-side Laplacian Variance Blur & Focus Meter
 * Evaluates image sharpness so the shutter button only activates when text is crisp.
 */

export interface SharpnessResult {
  variance: number;
  isSharp: boolean;
  qualityLabel: "Sharp" | "Acceptable" | "Blurry";
}

export function computeLaplacianVarianceFromCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  threshold: number = 110.0
): SharpnessResult {
  // Sample center 128x128 patch
  const patchW = Math.min(width, 128);
  const patchH = Math.min(height, 128);
  const startX = Math.floor((width - patchW) / 2);
  const startY = Math.floor((height - patchH) / 2);

  const imgData = ctx.getImageData(startX, startY, patchW, patchH);
  const data = imgData.data;

  // Convert to grayscale
  const gray = new Float32Array(patchW * patchH);
  for (let i = 0; i < data.length; i += 4) {
    gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // 3x3 Discrete Laplacian Operator:
  // [ 0,  1,  0 ]
  // [ 1, -4,  1 ]
  // [ 0,  1,  0 ]
  let sum = 0;
  let sumSq = 0;
  let count = 0;

  for (let y = 1; y < patchH - 1; y++) {
    for (let x = 1; x < patchW - 1; x++) {
      const idx = y * patchW + x;
      const lap =
        gray[(y - 1) * patchW + x] +
        gray[(y + 1) * patchW + x] +
        gray[y * patchW + (x - 1)] +
        gray[y * patchW + (x + 1)] -
        4 * gray[idx];

      sum += lap;
      sumSq += lap * lap;
      count++;
    }
  }

  if (count === 0) return { variance: 0, isSharp: false, qualityLabel: "Blurry" };

  const mean = sum / count;
  const variance = sumSq / count - mean * mean;

  let qualityLabel: "Sharp" | "Acceptable" | "Blurry" = "Blurry";
  if (variance >= threshold) {
    qualityLabel = "Sharp";
  } else if (variance >= threshold * 0.75) {
    qualityLabel = "Acceptable";
  }

  return {
    variance,
    isSharp: variance >= threshold,
    qualityLabel,
  };
}
