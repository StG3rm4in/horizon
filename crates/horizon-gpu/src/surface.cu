// Horizon's surface on the GPU: inputs per block (biome, temperature), operation by operation.
// Needs base.cu (GRAD, floor_i32) and the PERM / OFF tables; the indices of the temperature noises
// are given by #define: T_HEIGHT, T_INFO, T_FREEZE0..2 (and T_FREEZE_FREQ_k, T_FREEZE_AMP_k).

// ---------- Biome zoom (BiomeZoom::quart_at_block) ----------
__device__ __forceinline__ long long sf_g(long long m) {
  unsigned long long u = (unsigned long long)m;
  return (long long)(u * (u * 6364136223846793005ull + 1442695040888963407ull));
}
__device__ __forceinline__ double sf_fiddle(long long s) {
  return ((double)((s >> 24) & 1023) / 1024.0 - 0.5) * 0.9;
}
__device__ void sf_quarter(long long seed, int bx, int by, int bz, int *q) {
  int sx = bx - 2, sy = by - 2, sz = bz - 2;
  int base[3] = {sx >> 2, sy >> 2, sz >> 2};
  double fr[3] = {(double)(sx & 3) / 4.0, (double)(sy & 3) / 4.0, (double)(sz & 3) / 4.0};
  long long c0 = base[0], c1 = base[1], c2 = base[2];
  long long root = sf_g(seed);
  double best = __longlong_as_double(0x7ff0000000000000ll);
  int sel[3] = {base[0], base[1], base[2]};
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z) {
        long long m0 = (long long)((unsigned long long)root + (unsigned long long)(c0 + x));
        long long m1 = (long long)((unsigned long long)sf_g(m0) + (unsigned long long)(c1 + y));
        long long m2 = (long long)((unsigned long long)sf_g(m1) + (unsigned long long)(c2 + z));
        long long m3 = (long long)((unsigned long long)sf_g(m2) + (unsigned long long)(c0 + x));
        long long m4 = (long long)((unsigned long long)sf_g(m3) + (unsigned long long)(c1 + y));
        long long m5 = (long long)((unsigned long long)sf_g(m4) + (unsigned long long)(c2 + z));
        long long m6 = (long long)((unsigned long long)sf_g(m5) + (unsigned long long)seed);
        long long m7 = (long long)((unsigned long long)sf_g(m6) + (unsigned long long)seed);
        double dx = fr[0] - (double)x + sf_fiddle(m5);
        double dy = fr[1] - (double)y + sf_fiddle(m6);
        double dz = fr[2] - (double)z + sf_fiddle(m7);
        double d = dz * dz + dy * dy + dx * dx;
        if (best > d) {
          best = d;
          sel[0] = base[0] + x;
          sel[1] = base[1] + y;
          sel[2] = base[2] + z;
        }
      }
  q[0] = sel[0];
  q[1] = sel[1];
  q[2] = sel[2];
}
// Biome of a quart in the 3 x 3 chunks around chunk (ccx, ccz): B = sections of the chunks one
// after the other (dz then dx, from -1 to +1), each chunk as [quart y][z & 3][x & 3]; y clamped to
// the chunk.
// -1 if the quart falls outside the 3 x 3 chunks.
__device__ int sf_biome_quarter(const unsigned short *B, int ccx, int ccz, int qlow, int nqy, const int *q) {
  int dxc = (q[0] >> 2) - (ccx - 1), dzc = (q[2] >> 2) - (ccz - 1);
  if (dxc < 0 || dxc > 2 || dzc < 0 || dzc > 2) return -1;
  int y = q[1] < qlow ? qlow : (q[1] > qlow + nqy - 1 ? qlow + nqy - 1 : q[1]);
  return B[((dzc * 3 + dxc) * nqy + (y - qlow)) * 16 + (((q[2] & 3) << 2) | (q[0] & 3))];
}

// ---------- Temperature (BiomeTemperatureSettings::height_adjusted) ----------
__device__ float sf_freeze(double x, double z) {
  float v = 0.0f;
  v = __fadd_rn(v, __fmul_rn(T_FREEZE_AMP_0, sf_simplex2(PERM[T_FREEZE0], OFF[T_FREEZE0], x * T_FREEZE_FREQ_0, z * T_FREEZE_FREQ_0)));
  v = __fadd_rn(v, __fmul_rn(T_FREEZE_AMP_1, sf_simplex2(PERM[T_FREEZE1], OFF[T_FREEZE1], x * T_FREEZE_FREQ_1, z * T_FREEZE_FREQ_1)));
  v = __fadd_rn(v, __fmul_rn(T_FREEZE_AMP_2, sf_simplex2(PERM[T_FREEZE2], OFF[T_FREEZE2], x * T_FREEZE_FREQ_2, z * T_FREEZE_FREQ_2)));
  return v;
}
// base: base temperature of the biome; frozen: Frozen modifier
__device__ float sf_temperature(float base, int frozen, int x, int y, int z, int sea) {
  if (frozen) {
    double fx = (double)x, fz = (double)z;
    double freeze = (double)__fmul_rn(sf_freeze(fx * 0.05, fz * 0.05), 7.0f);
    double info = (double)sf_simplex2(PERM[T_INFO], OFF[T_INFO], fx * 0.2, fz * 0.2);
    if (freeze + info < 0.3 && (double)sf_simplex2(PERM[T_INFO], OFF[T_INFO], fx * 0.09, fz * 0.09) < 0.8) base = 0.2f;
  }
  int start = sea + 17;
  if (y > start) {
    float variation = __fmul_rn(sf_simplex2(PERM[T_HEIGHT], OFF[T_HEIGHT], (double)__fdiv_rn((float)x, 8.0f), (double)__fdiv_rn((float)z, 8.0f)), 8.0f);
    return __fsub_rn(base, __fdiv_rn(__fmul_rn(__fsub_rn(__fadd_rn(variation, (float)y), (float)start), 0.05f), 40.0f));
  }
  return base;
}

// ---------- check kernels (S2 milestone) ----------
extern "C" __global__ void ksf_quarters(const int *P, int *Q, int n, long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  sf_quarter(seed, P[3 * i], P[3 * i + 1], P[3 * i + 2], Q + 3 * i);
}
extern "C" __global__ void ksf_temperatures(const int *P, const float *BASES, const int *FROZEN, int nb, float *T, int n, int sea) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nb) return;
  int k = i / n, p = i % n;
  T[i] = sf_temperature(BASES[k], FROZEN[k], P[3 * p], P[3 * p + 1], P[3 * p + 2], sea);
}

#ifdef SF_MINY
// ---------- Column descent (material_stage::build_columns, S3 milestone) ----------
// Fill codes: & 3 = 0 solid (and 8), 1 air, 2 or 3 fluid.
#define SF_SOLID 0
#define SF_AIR 1
#define SF_FLUID 2
struct Ctx {
  int x, y, z, depth, gx, gz, above, below, water, minimum, biome, block, error;
  double secondary;
  const unsigned short *B;
  int cx, cz;
  // Veins of the region: position of the block in the region (SF_VR x SF_VH x SF_VR from SF_VLO)
  const float *VOL;
  int vx, vz;
};
__device__ __forceinline__ int sf_kind(const signed char *col, int y) {
  if (y < SF_MINY || y > SF_MINY + SF_H - 1) return SF_AIR;
  int c = col[y - SF_MINY] & 3;
  return c == 0 ? SF_SOLID : (c == 1 ? SF_AIR : SF_FLUID);
}
__device__ int sf_biome(Ctx &c) {
  if (c.biome == -2) {
    int q[3];
    sf_quarter(SF_ZOOM, c.x, c.y, c.z, q);
    c.biome = sf_biome_quarter(c.B, c.cx, c.cz, SF_BMINY >> 2, SF_BH >> 2, q);
  }
  return c.biome;
}

#ifdef SF_EXT
// ---------- Extensions (material_extensions.rs, S4 milestone) ----------
#define SF_NOTHING (-2147483647 - 1)
__device__ __forceinline__ double sf_jmin_d(double a, double b) {
  if (isnan(a)) return a;
  if (isnan(b)) return b;
  if (a == 0.0 && b == 0.0) return __longlong_as_double(__double_as_longlong(a) | __double_as_longlong(b));
  return a <= b ? a : b;
}
// Fill code at y (& 3: 0 default block, 1 air, 2 water, 3 lava); air outside the world.
__device__ __forceinline__ int sf_code(const signed char *col, int y) {
  if (y < SF_MINY || y > SF_MINY + SF_H - 1) return 1;
  return col[y - SF_MINY] & 3;
}
// MaterialExtensions::eroded_badlands: the top of the pillar (filled from otop up to this top), or
// SF_NOTHING.
__device__ int sf_pillar(const signed char *col, int x, int z, int otop) {
  double dx = (double)x, dz = (double)z;
  double amp = sf_jmin_d(fabs((double)BD_S(dx, 0.0, dz) * 8.25), (double)__fmul_rn(BD_P(dx * 0.2, 0.0, dz * 0.2), 15.0f));
  if (amp <= 0.0) return SF_NOTHING;
  double roof = fabs((double)BD_R(dx * 0.75, 0.0, dz * 0.75) * 1.5);
  int high = floor_sat(64.0 + sf_jmin_d(amp * amp * 2.5, ceil(roof * 50.0) + 24.0));
  if (otop > high) return SF_NOTHING;
  // The first default or water block below the top: water, no pillar
  for (int y = high; y >= SF_MINY; --y) {
    int k = sf_code(col, y);
    if (k == 0) break;
    if (k == 2) return SF_NOTHING;
  }
  return high;
}
// MaterialExtensions::frozen_ocean_with_melt, after the descent: writes packed ice and snow into
// `output` (state + 1), returns the highest y written (SF_NOTHING if none). b: biome of the column
// (at its first free y).
__device__ int sf_iceberg(const signed char *col, unsigned *output, int x, int z, int otop, int minimum, int b,
                          unsigned long long RLO, unsigned long long RHI, int *error) {
  double dx = (double)x, dz = (double)z;
  double amp = sf_jmin_d(fabs((double)IC_S(dx, 0.0, dz) * 8.25), (double)__fmul_rn(IC_P(dx * 1.28, 0.0, dz * 1.28), 15.0f));
  if (amp <= 1.8) return SF_NOTHING;
  double roof = fabs((double)IC_R(dx * 1.17, 0.0, dz * 1.17) * 1.5);
  double h = sf_jmin_d(amp * amp * 1.2, ceil(roof * 40.0) + 14.0);
  if (b >= NTB) {
    *error = 1;
    return SF_NOTHING;
  }
  // should_melt_frozen_ocean_iceberg_slightly: temperature at sea level > 0.1
  if (sf_temperature(__int_as_float((int)TB[b]), TG[b], x, SF_SEA, z, SF_SEA) > 0.1f) h -= 2.0;
  if (h <= 2.0) return SF_NOTHING;
  int ilow = (int)((double)SF_SEA - h - 7.0), ihigh = (int)(h + (double)SF_SEA);
  unsigned long long lo, hi;
  aq_at(RLO, RHI, x, 0, z, lo, hi);
  int limit = 2 + aq_next_int_bound(lo, hi, 4);
  int snow_h = (int)((unsigned)SF_SEA + 18u + (unsigned)aq_next_int_bound(lo, hi, 10));
  int n = 0, plus = SF_NOTHING;
  int start_y = (int)((unsigned)ihigh + 1u);
  if (otop > start_y) start_y = otop;
  for (int y = start_y; y >= minimum; --y) {
    int k = sf_code(col, y);
    // The draws follow Horizon's short circuits
    bool fill = k == 1 && y < ihigh && aq_next_double(lo, hi) > 0.01;
    if (!fill) fill = k == 2 && y > ilow && y < SF_SEA && aq_next_double(lo, hi) > 0.15;
    if (fill) {
      int s;
      if (n <= limit && y > snow_h) {
        n += 1;
        s = SF_SNOW;
      } else {
        s = SF_ICE;
      }
      if (y >= SF_MINY && y <= SF_MINY + SF_H - 1) {
        output[y - SF_MINY] = (unsigned)s + 1u;
        if (y > plus) plus = y;
      }
    }
  }
  return plus;
}
#endif
#endif
