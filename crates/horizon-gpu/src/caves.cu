// Carver mask of a chunk (Horizon's carver.rs, carver/canyon.rs, horizon_core::carving, 26.3).
// One block per planned element (room, cave tunnel, canyon): thread 0 traces the path and keeps the
// ellipsoids that reach the chunk; the whole block rasterizes them into a shared mask, merged into
// the global mask with atomicOr. The mask is a union: the order of the elements does not matter.
// Exact arithmetic: float and double with the _rn intrinsics, without contraction (--fmad=false),
// Horizon's sine table (sine_table) in global memory.
//
// Output (64-bit words): NW mask words, NC counters of cells tested per carver call (budget of
// 2^24), then one word of error flags (the chunk then falls back to the CPU).
#define CV_ROOM 0
#define CV_TUNNEL 1
#define CV_CANYON 2
#define CV_EV_MAX 512
#define CV_NY_MAX 512
#define CV_WORDS_MAX 1536
#define CV_STACK 8
// Flags
#define CV_E_EVENTS 1
#define CV_E_HEIGHT 2
#define CV_E_BOUND 4
#define CV_E_STACK 8
#define CV_E_MASK 16
#define CV_E_WIDTH 32
#define CV_E_PROVIDER 64

#define CV_SCALE 10430.378350470453
#define CV_PI_F 3.14159265358979323846f
#define CV_FRAC_PI_2_F 1.57079632679489661923f

// Planned element, 16 words (the same #[repr(C)] as CaveItem on the Rust side)
struct CvItem {
  unsigned long long seed;  // tunnel: seed of LegacyRandom::new; canyon: state after the width table
  double x, y, z;
  double scale;     // tunnel: vertical_scale; canyon: y_scale
  double horizontal;  // tunnel: run multiplier; room: horizontal radius
  double vertical;    // tunnel: run multiplier; room: vertical radius
  double floor_level;    // cave: floor_level
  unsigned kind, call;
  int end;
  unsigned width;  // canyon: start of its width table
  float thickness, yaw, pitch;
  unsigned provider;  // canyon, horizontal_radius_factor: 0 constant (fmin), 1 uniform [fmin, fmax)
  float fmin, fmax, vdefault, vcenter;
  unsigned padding[4];
};

struct Lr {
  unsigned long long s;
};
__device__ __forceinline__ void lr_seed(Lr &r, long long g) {
  r.s = ((unsigned long long)g ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
}
__device__ __forceinline__ int lr_bits(Lr &r, int n) {
  r.s = (r.s * 0x5DEECE66DULL + 11ULL) & ((1ULL << 48) - 1);
  return (int)(unsigned)(r.s >> (48 - n));
}
__device__ __forceinline__ float lr_float(Lr &r) { return __fmul_rn((float)lr_bits(r, 24), 1.0f / 16777216.0f); }
__device__ __forceinline__ int lr_int(Lr &r, int bound, unsigned *err) {
  if (bound <= 0) {
    *err |= CV_E_BOUND;
    return 0;
  }
  if ((bound & (bound - 1)) == 0) return (int)(((long long)bound * (long long)lr_bits(r, 31)) >> 31);
  for (;;) {
    int bits = lr_bits(r, 31);
    int v = bits % bound;
    if ((int)((unsigned)bits - (unsigned)v + (unsigned)(bound - 1)) >= 0) return v;
  }
}
__device__ __forceinline__ long long lr_long(Lr &r) {
  long long high = (long long)lr_bits(r, 32) << 32;
  return high + (long long)lr_bits(r, 32);
}

__device__ __forceinline__ float cv_sin(const float *T, double a) {
  long long i = __double2ll_rz(__dmul_rn(a, CV_SCALE));
  return T[i & 65535];
}
__device__ __forceinline__ float cv_cos(const float *T, double a) {
  long long i = __double2ll_rz(__dadd_rn(__dmul_rn(a, CV_SCALE), 16384.0));
  return T[i & 65535];
}
// can_reach: thickness rounded in float, distances in double
__device__ __forceinline__ bool cv_reached(int cx, int cz, double x, double z, int step, int end, float ep) {
  double dx = __dsub_rn(x, (double)(cx * 16 + 8));
  double dz = __dsub_rn(z, (double)(cz * 16 + 8));
  double rest = (double)(end - step);
  double r = (double)__fadd_rn(__fadd_rn(ep, 2.0f), 16.0f);
  return __dsub_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dz, dz)), __dmul_rn(rest, rest)) <= __dmul_rn(r, r);
}

struct Ev {
  double x, y, z, h, v;
};

struct Tun {
  long long seed;
  double x, y, z;
  float ep, yaw, pitch;
  int start, end;
  double scale;
};

// Keeps an ellipsoid if it touches the chunk (range test of carve_ellipsoid_with_budget)
__device__ __forceinline__ void cv_keep(Ev *E, int *n, unsigned *err, int cx, int cz, double x, double y, double z,
                                          double h, double v) {
  double reach = __dadd_rn(16.0, __dmul_rn(h, 2.0));
  if (fabs(__dsub_rn(x, (double)(cx * 16 + 8))) > reach || fabs(__dsub_rn(z, (double)(cz * 16 + 8))) > reach) return;
  if (*n >= CV_EV_MAX) {
    *err |= CV_E_EVENTS;
    return;
  }
  Ev e;
  e.x = x;
  e.y = y;
  e.z = z;
  e.h = h;
  e.v = v;
  E[(*n)++] = e;
}

// CaveRun::tunnel, with a stack for the branches (the first one traced before the second)
__device__ void cv_tunnels(const CvItem &it, const float *T, int cx, int cz, Ev *E, int *n, unsigned *err) {
  Tun pile[CV_STACK];
  int np = 1;
  pile[0].seed = (long long)it.seed;
  pile[0].x = it.x;
  pile[0].y = it.y;
  pile[0].z = it.z;
  pile[0].ep = it.thickness;
  pile[0].yaw = it.yaw;
  pile[0].pitch = it.pitch;
  pile[0].start = 0;
  pile[0].end = it.end;
  pile[0].scale = it.scale;
  while (np > 0) {
    Tun t = pile[--np];
    Lr r;
    lr_seed(r, t.seed);
    int shared = lr_int(r, t.end / 2, err) + t.end / 4;
    bool gentle = lr_int(r, 6, err) == 0;
    if (*err) return;
    float dyaw = 0.0f, dpitch = 0.0f;
    for (int step = t.start; step < t.end; ++step) {
      double rayon = __dadd_rn(1.5, (double)__fmul_rn(cv_sin(T, (double)__fdiv_rn(__fmul_rn(CV_PI_F, (float)step), (float)t.end)), t.ep));
      double vert = __dmul_rn(rayon, t.scale);
      float ct = cv_cos(T, (double)t.pitch);
      t.x = __dadd_rn(t.x, (double)__fmul_rn(cv_cos(T, (double)t.yaw), ct));
      t.y = __dadd_rn(t.y, (double)cv_sin(T, (double)t.pitch));
      t.z = __dadd_rn(t.z, (double)__fmul_rn(cv_sin(T, (double)t.yaw), ct));
      t.pitch = __fmul_rn(t.pitch, gentle ? 0.92f : 0.7f);
      t.pitch = __fadd_rn(t.pitch, __fmul_rn(dpitch, 0.1f));
      t.yaw = __fadd_rn(t.yaw, __fmul_rn(dyaw, 0.1f));
      dpitch = __fmul_rn(dpitch, 0.9f);
      dyaw = __fmul_rn(dyaw, 0.75f);
      {
        float a = lr_float(r), b = lr_float(r), c = lr_float(r);
        dpitch = __fadd_rn(dpitch, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 2.0f));
      }
      {
        float a = lr_float(r), b = lr_float(r), c = lr_float(r);
        dyaw = __fadd_rn(dyaw, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 4.0f));
      }
      if (step == shared && t.ep > 1.0f) {
        if (np + 2 > CV_STACK) {
          *err |= CV_E_STACK;
          return;
        }
        Tun f[2];
        for (int k = 0; k < 2; ++k) {
          f[k] = t;
          f[k].seed = lr_long(r);
          f[k].ep = __fadd_rn(__fmul_rn(lr_float(r), 0.5f), 0.5f);
          f[k].yaw = k == 0 ? __fsub_rn(t.yaw, CV_FRAC_PI_2_F) : __fadd_rn(t.yaw, CV_FRAC_PI_2_F);
          f[k].pitch = __fdiv_rn(t.pitch, 3.0f);
          f[k].start = step;
          f[k].scale = 1.0;
        }
        pile[np++] = f[1];
        pile[np++] = f[0];
        break;
      }
      if (lr_int(r, 4, err) == 0) continue;
      if (!cv_reached(cx, cz, t.x, t.z, step, t.end, t.ep)) break;
      cv_keep(E, n, err, cx, cz, t.x, t.y, t.z, __dmul_rn(rayon, it.horizontal), __dmul_rn(vert, it.vertical));
    }
  }
}

// CanyonCarver::trace
__device__ void cv_canyon(const CvItem &it, const float *T, int cx, int cz, Ev *E, int *n, unsigned *err) {
  Lr r;
  r.s = it.seed;
  double x = it.x, y = it.y, z = it.z;
  float yaw = it.yaw, pitch = it.pitch, dyaw = 0.0f, dpitch = 0.0f;
  int end = it.end;
  for (int step = 0; step < end; ++step) {
    double h = __dadd_rn(1.5, (double)__fmul_rn(cv_sin(T, (double)__fdiv_rn(__fmul_rn((float)step, CV_PI_F), (float)end)), it.thickness));
    double v = __dmul_rn(h, it.scale);
    float f = it.fmin;
    if (it.provider == 1) {
      f = __fadd_rn(__fmul_rn(lr_float(r), __fsub_rn(it.fmax, it.fmin)), it.fmin);
    } else if (it.provider != 0) {
      *err |= CV_E_PROVIDER;
      return;
    }
    h = __dmul_rn(h, (double)f);
    float centre = __fsub_rn(1.0f, __fmul_rn(fabsf(__fsub_rn(0.5f, __fdiv_rn((float)step, (float)end))), 2.0f));
    float mult = __fadd_rn(it.vdefault, __fmul_rn(it.vcenter, centre));
    v = __dmul_rn(__dmul_rn((double)mult, v), (double)__fadd_rn(__fmul_rn(lr_float(r), 0.25f), 0.75f));
    float ct = cv_cos(T, (double)pitch);
    float st = cv_sin(T, (double)pitch);
    x = __dadd_rn(x, (double)__fmul_rn(cv_cos(T, (double)yaw), ct));
    y = __dadd_rn(y, (double)st);
    z = __dadd_rn(z, (double)__fmul_rn(cv_sin(T, (double)yaw), ct));
    pitch = __fmul_rn(pitch, 0.7f);
    pitch = __fadd_rn(pitch, __fmul_rn(dpitch, 0.05f));
    yaw = __fadd_rn(yaw, __fmul_rn(dyaw, 0.05f));
    dpitch = __fmul_rn(dpitch, 0.8f);
    dyaw = __fmul_rn(dyaw, 0.5f);
    {
      float a = lr_float(r), b = lr_float(r), c = lr_float(r);
      dpitch = __fadd_rn(dpitch, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 2.0f));
    }
    {
      float a = lr_float(r), b = lr_float(r), c = lr_float(r);
      dyaw = __fadd_rn(dyaw, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 4.0f));
    }
    if (lr_int(r, 4, err) == 0) continue;
    if (!cv_reached(cx, cz, x, z, step, end, it.thickness)) return;
    cv_keep(E, n, err, cx, cz, x, y, z, h, v);
  }
}

// IN: elements (16 words each); W: width tables of the canyons (nwidth each); T: sine table.
// OUT: NW mask words, NC counters, 1 flag word (zero before the launch).
// Mask: bit (y - ymin) + (z + x * 16) * height; cells tested from y = ymax(e) down to ymin(e) + 1.
extern "C" __global__ void kcv_trace(const unsigned long long *IN, const float *W, const float *T,
                                     unsigned long long *OUT, int nw, int nc, int cx, int cz, int ymin,
                                     int ymax, int height, int gen_ymin, int nwidth) {
  __shared__ Ev E[CV_EV_MAX];
  __shared__ unsigned long long M[CV_WORDS_MAX];
  __shared__ double NX[16], NZ[16], NX2[16], NZ2[16];
  __shared__ double NY[CV_NY_MAX], NY2[CV_NY_MAX];
  __shared__ int NE, BOUNDS[6];
  __shared__ unsigned ERR, TOUCHED;
  __shared__ unsigned long long COUNT;
  const CvItem &it = *reinterpret_cast<const CvItem *>(IN + 16 * (size_t)blockIdx.x);
  int t = threadIdx.x;
  if (t == 0) {
    int n = 0;
    unsigned err = 0;
    if (nw > CV_WORDS_MAX) err |= CV_E_MASK;
    else if (it.kind == CV_ROOM) cv_keep(E, &n, &err, cx, cz, it.x, it.y, it.z, it.horizontal, it.vertical);
    else if (it.kind == CV_TUNNEL) cv_tunnels(it, T, cx, cz, E, &n, &err);
    else if (it.kind == CV_CANYON) cv_canyon(it, T, cx, cz, E, &n, &err);
    else err |= CV_E_PROVIDER;
    NE = err ? 0 : n;
    ERR = err;
    TOUCHED = 0;
    COUNT = 0;
  }
  __syncthreads();
  int ne = NE;
  if (ne > 0)
    for (int i = t; i < nw; i += blockDim.x) M[i] = 0;
  bool canyon = it.kind == CV_CANYON;
  double floor_level = it.floor_level;
  unsigned long long count = 0;
  unsigned err = 0, touched = 0;
  int x0 = cx * 16, z0 = cz * 16;
  for (int k = 0; k < ne; ++k) {
    Ev e = E[k];
    __syncthreads();
    if (t == 0) {
      int mx = __double2int_rd(__dsub_rn(e.x, e.h)) - x0 - 1;
      int Mx = __double2int_rd(__dadd_rn(e.x, e.h)) - x0;
      int my = __double2int_rd(__dsub_rn(e.y, e.v)) - 1;
      int My = __double2int_rd(__dadd_rn(e.y, e.v)) + 1;
      int mz = __double2int_rd(__dsub_rn(e.z, e.h)) - z0 - 1;
      int Mz = __double2int_rd(__dadd_rn(e.z, e.h)) - z0;
      BOUNDS[0] = mx < 0 ? 0 : mx;
      BOUNDS[1] = Mx > 15 ? 15 : Mx;
      BOUNDS[2] = my < ymin ? ymin : my;
      BOUNDS[3] = My > ymax ? ymax : My;
      BOUNDS[4] = mz < 0 ? 0 : mz;
      BOUNDS[5] = Mz > 15 ? 15 : Mz;
    }
    __syncthreads();
    int bx = BOUNDS[0], Bx = BOUNDS[1], by = BOUNDS[2], By = BOUNDS[3], bz = BOUNDS[4], Bz = BOUNDS[5];
    int nx = Bx - bx + 1, nz = Bz - bz + 1, ny = By - by;
    if (nx <= 0 || nz <= 0 || ny <= 0) continue;
    if (ny > CV_NY_MAX) {
      err |= CV_E_HEIGHT;
      continue;
    }
    if (t < nx) {
      double v = __ddiv_rn(__dsub_rn(__dadd_rn((double)(x0 + bx + t), 0.5), e.x), e.h);
      NX[t] = v;
      NX2[t] = __dmul_rn(v, v);
    }
    if (t >= 32 && t < 32 + nz) {
      double v = __ddiv_rn(__dsub_rn(__dadd_rn((double)(z0 + bz + t - 32), 0.5), e.z), e.h);
      NZ[t - 32] = v;
      NZ2[t - 32] = __dmul_rn(v, v);
    }
    for (int j = t; j < ny; j += blockDim.x) {
      double v = __ddiv_rn(__dsub_rn(__dsub_rn((double)(By - j), 0.5), e.y), e.v);
      NY[j] = v;
      NY2[j] = __dmul_rn(v, v);
    }
    __syncthreads();
    int total = nx * nz * ny;
    for (int c = t; c < total; c += blockDim.x) {
      int j = c % ny, col = c / ny;
      int ix = col / nz, iz = col % nz;
      if (__dadd_rn(NX2[ix], NZ2[iz]) >= 1.0) continue;
      ++count;
      int y = By - j;
      bool skipped;
      if (canyon) {
        int li = y - gen_ymin - 1;
        if (li < 0 || li >= nwidth) {
          err |= CV_E_WIDTH;
          continue;
        }
        double f = (double)W[it.width + li];
        skipped = __dadd_rn(__dmul_rn(__dadd_rn(NX2[ix], NZ2[iz]), f), __ddiv_rn(NY2[j], 6.0)) >= 1.0;
      } else {
        skipped = NY[j] <= floor_level || __dadd_rn(__dadd_rn(NX2[ix], NY2[j]), NZ2[iz]) >= 1.0;
      }
      if (!skipped) {
        int i = (y - ymin) + ((bz + iz) + (bx + ix) * 16) * height;
        atomicOr(&M[i >> 6], 1ULL << (i & 63));
        touched = 1;
      }
    }
  }
  if (count) atomicAdd(&COUNT, count);
  if (err) atomicOr(&ERR, err);
  if (touched) atomicOr(&TOUCHED, 1u);
  __syncthreads();
  if (TOUCHED)
    for (int i = t; i < nw; i += blockDim.x)
      if (M[i]) atomicOr(&OUT[i], M[i]);
  if (t == 0) {
    if (COUNT && it.call < (unsigned)nc) atomicAdd(&OUT[nw + it.call], COUNT);
    if (ERR) atomicOr(&OUT[nw + nc], (unsigned long long)ERR);
  }
}
