/** Minimal dense linear algebra on flat Float32Arrays (row-major). */

/** out[m×n] = a[m×k] @ b[k×n] (overwrites out). */
export function matmul(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
  out: Float32Array,
): void {
  out.fill(0, 0, m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const av = a[i * k + p];
      if (av === 0) continue;
      const bRow = p * n;
      const oRow = i * n;
      for (let j = 0; j < n; j++) out[oRow + j] += av * b[bRow + j];
    }
  }
}

/** out[k×n] += aᵀ @ b, where a is [m×k] and b is [m×n]. (Weight gradients.) */
export function matmulATB(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
  out: Float32Array,
): void {
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const bRow = i * n;
    for (let p = 0; p < k; p++) {
      const av = a[aRow + p];
      if (av === 0) continue;
      const oRow = p * n;
      for (let j = 0; j < n; j++) out[oRow + j] += av * b[bRow + j];
    }
  }
}

/** out[m×k] += a @ bᵀ, where a is [m×n] and b is [k×n]. (Input gradients.) */
export function matmulABT(
  a: Float32Array,
  b: Float32Array,
  m: number,
  n: number,
  k: number,
  out: Float32Array,
): void {
  for (let i = 0; i < m; i++) {
    const aRow = i * n;
    const oRow = i * k;
    for (let p = 0; p < k; p++) {
      const bRow = p * n;
      let sum = 0;
      for (let j = 0; j < n; j++) sum += a[aRow + j] * b[bRow + j];
      out[oRow + p] += sum;
    }
  }
}
