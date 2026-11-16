// Shared primitives, reproduced operation by operation from horizon_core::noise.
// Every operation goes through *_rn intrinsics: no fused multiply-add.

__constant__ float GRAD[16][3] = {
    {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
    {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
    {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
    {1, 1, 0}, {0, -1, 1}, {-1, 1, 0}, {0, -1, -1},
};

// noise::wrap: vanilla's periodic reduction (Far Lands disabled)
__device__ __forceinline__ double wrapd(double v) {
    const double half = __longlong_as_double(0x416FFFFFFFFFFFFFLL);  // just below 2^24
    if (v >= -half && v < half) return v;
    return __dsub_rn(v, __dmul_rn(floor(__dadd_rn(__ddiv_rn(v, 33554432.0), 0.5)), 33554432.0));
}
// geometry::floor_i32 (Mth.floor)
__device__ __forceinline__ int floor_i32(double v) {
    int t = __double2int_rz(v);
    return v < (double)t ? t - 1 : t;
}
__device__ __forceinline__ float smooth(float v) {  // v*v*v*(v*(v*6-15)+10)
    float a = __fmul_rn(__fmul_rn(v, v), v);
    float b = __fadd_rn(__fmul_rn(v, __fsub_rn(__fmul_rn(v, 6.0f), 15.0f)), 10.0f);
    return __fmul_rn(a, b);
}
__device__ __forceinline__ float lerp1(float d, float s, float e) {
    return __fadd_rn(s, __fmul_rn(d, __fsub_rn(e, s)));
}
__device__ __forceinline__ float lerp2(float a, float b, float v0, float v1, float v2, float v3) {
    return lerp1(b, lerp1(a, v0, v1), lerp1(a, v2, v3));
}
__device__ __forceinline__ float lerp3(float f0, float f1, float f2, const float *v) {
    return lerp1(f2, lerp2(f0, f1, v[0], v[1], v[2], v[3]), lerp2(f0, f1, v[4], v[5], v[6], v[7]));
}
__device__ __forceinline__ void indices(const unsigned char *P, int k0, int k1, int k2, int *g) {
    int p = P[k0 & 255], q = P[(k0 + 1) & 255];
    int lower = P[(p + k1) & 255], upper = P[(p + k1 + 1) & 255];
    int lower_x = P[(q + k1) & 255], upper_x = P[(q + k1 + 1) & 255];
    int rows[8] = {lower, lower_x, upper, upper_x, lower, lower_x, upper, upper_x};
#pragma unroll
    for (int i = 0; i < 8; ++i) g[i] = P[(rows[i] + k2 + (i >> 2)) & 255] & 15;
}

// PerlinNoise::add_volume (volume path) for one point: coordinate * scale per axis,
// corner value = (gx*(dx-bx) + gz*(dz-bz)) + gy*(dy-by)
__device__ float perlin_vol(const unsigned char *P, const double *O,
                            int c0, double s0, int c1, double s1, int c2, double s2) {
    double h0 = __dadd_rn(wrapd(__dmul_rn((double)c0, s0)), O[0]);
    double h1 = __dadd_rn(wrapd(__dmul_rn((double)c1, s1)), O[1]);
    double h2 = __dadd_rn(wrapd(__dmul_rn((double)c2, s2)), O[2]);
    int k0 = floor_i32(h0), k1 = floor_i32(h1), k2 = floor_i32(h2);
    float d0 = __double2float_rn(__dsub_rn(h0, (double)k0));
    float d1 = __double2float_rn(__dsub_rn(h1, (double)k1));
    float d2 = __double2float_rn(__dsub_rn(h2, (double)k2));
    int g[8];
    indices(P, k0, k1, k2, g);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float xz = __fadd_rn(__fmul_rn(GRAD[g[i]][0], __fsub_rn(d0, (float)(i & 1))),
                             __fmul_rn(GRAD[g[i]][2], __fsub_rn(d2, (float)((i >> 2) & 1))));
        v[i] = __fadd_rn(xz, __fmul_rn(GRAD[g[i]][1], __fsub_rn(d1, (float)((i >> 1) & 1))));
    }
    return lerp3(smooth(d0), smooth(d1), smooth(d2), v);
}

// SmearedPerlinNoise: volume path with a Y fudge (PerlinNoise::add_volume, fudge_y_scale):
// the smoothing uses dy, the corner values use dy - compute_fudge_y(input_y, dy, scale)
__device__ float perlin_vol_smeared(const unsigned char *P, const double *O,
                                    int c0, double s0, int c1, double s1, int c2, double s2, double fudge) {
    double h0 = __dadd_rn(wrapd(__dmul_rn((double)c0, s0)), O[0]);
    double iy = __dmul_rn((double)c1, s1);
    double h1 = __dadd_rn(wrapd(iy), O[1]);
    double h2 = __dadd_rn(wrapd(__dmul_rn((double)c2, s2)), O[2]);
    int k0 = floor_i32(h0), k1 = floor_i32(h1), k2 = floor_i32(h2);
    float d0 = __double2float_rn(__dsub_rn(h0, (double)k0));
    double dyd = __dsub_rn(h1, (double)k1);
    float d2 = __double2float_rn(__dsub_rn(h2, (double)k2));
    double limited = (iy >= 0.0 && iy < dyd) ? iy : dyd;
    const double eps = __longlong_as_double(0x3e7ad7f2a0000000LL);  // f64::from(1e-7_f32)
    double f = __dmul_rn((double)floor_i32(__dadd_rn(__ddiv_rn(limited, fudge), eps)), fudge);
    float d1 = __double2float_rn(__dsub_rn(dyd, f));
    float fy = smooth(__double2float_rn(dyd));
    int g[8];
    indices(P, k0, k1, k2, g);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float xz = __fadd_rn(__fmul_rn(GRAD[g[i]][0], __fsub_rn(d0, (float)(i & 1))),
                             __fmul_rn(GRAD[g[i]][2], __fsub_rn(d2, (float)((i >> 2) & 1))));
        v[i] = __fadd_rn(xz, __fmul_rn(GRAD[g[i]][1], __fsub_rn(d1, (float)((i >> 1) & 1))));
    }
    return lerp3(smooth(d0), fy, smooth(d2), v);
}

// Volume path with the double part already computed per axis (cell k, position d, smoothing f).
// For a smeared noise, d1 is the Y position corrected by the fudge and f1 the smoothing of the raw
// position.
__device__ __forceinline__ float perlin_core(const unsigned char *P, int k0, float d0, float f0,
                                             int k1, float d1, float f1, int k2, float d2, float f2) {
    int g[8];
    indices(P, k0, k1, k2, g);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float xz = __fadd_rn(__fmul_rn(GRAD[g[i]][0], __fsub_rn(d0, (float)(i & 1))),
                             __fmul_rn(GRAD[g[i]][2], __fsub_rn(d2, (float)((i >> 2) & 1))));
        v[i] = __fadd_rn(xz, __fmul_rn(GRAD[g[i]][1], __fsub_rn(d1, (float)((i >> 1) & 1))));
    }
    return lerp3(f0, f1, f2, v);
}

// PerlinNoise::get3 (point by point path): dot product ((g0*a + g1*b) + g2*c)
__device__ float perlin_get3(const unsigned char *P, const double *O, double x, double y, double z) {
    double h0 = __dadd_rn(wrapd(x), O[0]);
    double h1 = __dadd_rn(wrapd(y), O[1]);
    double h2 = __dadd_rn(wrapd(z), O[2]);
    int k0 = floor_i32(h0), k1 = floor_i32(h1), k2 = floor_i32(h2);
    float p0 = __double2float_rn(__dsub_rn(h0, (double)k0));
    float p1 = __double2float_rn(__dsub_rn(h1, (double)k1));
    float p2 = __double2float_rn(__dsub_rn(h2, (double)k2));
    int g[8];
    indices(P, k0, k1, k2, g);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float a = __fmul_rn(GRAD[g[i]][0], __fsub_rn(p0, (float)(i & 1)));
        float b = __fmul_rn(GRAD[g[i]][1], __fsub_rn(p1, (float)((i >> 1) & 1)));
        float c = __fmul_rn(GRAD[g[i]][2], __fsub_rn(p2, (float)((i >> 2) & 1)));
        v[i] = __fadd_rn(__fadd_rn(a, b), c);
    }
    return lerp3(smooth(p0), smooth(p1), smooth(p2), v);
}

// ---------- 2D simplex noise (SimplexNoise::get2) ----------
__device__ __forceinline__ double sf_corner(int g, double p0, double p1, double p2, double rayon) {
  double att = rayon - p0 * p0 - p1 * p1 - p2 * p2;
  if (att < 0.0) return 0.0;
  double sq = att * att;
  return sq * sq * ((double)GRAD[g][0] * p0 + (double)GRAD[g][1] * p1 + (double)GRAD[g][2] * p2);
}
__device__ float sf_simplex2(const unsigned char *P, const double *O, double x, double y) {
  x = x + O[0];
  y = y + O[1];
  double r3 = sqrt(3.0);
  double bias = 0.5 * (r3 - 1.0);
  double debias = (3.0 - r3) / 6.0;
  double dec = (x + y) * bias;
  int i = floor_i32(x + dec), j = floor_i32(y + dec);
  double dec2 = (double)(i + j) * debias;
  double dx = x - ((double)i - dec2), dy = y - ((double)j - dec2);
  int a = dx > dy ? 1 : 0, b = dx > dy ? 0 : 1;
  i &= 255;
  j &= 255;
#define SF_GRAD(ea, eb) ((int)P[(i + (ea) + (int)P[(j + (eb)) & 255]) & 255] % 12)
  double first = sf_corner(SF_GRAD(0, 0), dx, dy, 0.0, 0.5);
  double second = sf_corner(SF_GRAD(a, b), dx - (double)a + debias, dy - (double)b + debias, 0.0, 0.5);
  double third = sf_corner(SF_GRAD(1, 1), dx - 1.0 + 2.0 * debias, dy - 1.0 + 2.0 * debias, 0.0, 0.5);
#undef SF_GRAD
  return (float)(70.0 * (first + second + third));
}
