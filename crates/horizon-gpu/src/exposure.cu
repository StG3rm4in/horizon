// Explosion exposure rays (ExposureGpu, exposure.rs): the rays of ServerExplosion.getSeenPercent,
// Level.clip with ClipContext.Block.COLLIDER and no fluid, as Horizon's CPU traces them:
// clip_world (horizon-world world_ray.rs), traverse_blocks (horizon-core geometry/traversal.rs),
// VoxelShape::clip and Aabb::clip_boxes (geometry/shape.rs, geometry/mod.rs). Every operation is
// in double, rounded as the CPU rounds it (_rn intrinsics, no contraction: NVRTC_OPTIONS), in the
// CPU's order; the conversions saturate as Rust's `as` does.
//
// One thread per ray: RAYS[6 i ..] = from x, y, z, to x, y, z. The cells the rays may cross are a
// cube from (LX, LY, LZ), NX x NY x NZ cells, CELLS[(x * NY + y) * NZ + z] the index of the cell's
// collision shape. Shape s: SH[8 s ..] = cube (0 or 1), size x, y, z, first coordinate in SV (the
// size + 1 boundaries of x, then of y, then of z), first occupancy word in SB (cell (x, y, z) is
// bit (x * size y + y) * size z + z), first box in SV (6 doubles a box: min x, y, z, max x, y, z),
// box count (0: the empty shape).
// OUT[i]: 1 the ray meets no shape (the CPU counts it as seen), 0 it meets one or its traversal
// fails (a non-finite point, 4,096 cells: no miss for the CPU either), 2 it left the cube (the
// CPU traces that box itself).
#define KX_VISITS 4096
#define KX_EPS 1.0e-7
#define KX_MAX 1.7976931348623157e308

// floor_i32: truncate, saturating (NaN -> 0), then one down below a negative fraction
__device__ __forceinline__ int kx_floor(double v) {
  int t;
  if (isnan(v)) t = 0;
  else if (v >= 2147483647.0) t = 2147483647;
  else if (v <= -2147483648.0) t = (int)0x80000000;
  else t = __double2int_rz(v);
  if (v < (double)t) return t == (int)0x80000000 ? t : t - 1;
  return t;
}
// java_min, then clamp: below `lo` gives `lo`, else the Java minimum with `hi`
__device__ __forceinline__ double kx_clamp(double v, double lo, double hi) {
  if (v < lo) return lo;
  if (isnan(v)) return v;
  if (isnan(hi)) return hi;
  if (v == 0.0 && hi == 0.0)
    return __longlong_as_double(__double_as_longlong(v) | __double_as_longlong(hi));
  return v <= hi ? v : hi;
}
// VoxelShape::find_index on axis a of shape h
__device__ int kx_index(const unsigned *h, const double *__restrict__ SV, int a, double value) {
  int size = (int)h[1 + a];
  if (h[0]) return kx_floor(kx_clamp(__dmul_rn(value, (double)size), -1.0, (double)size));
  // Upper-bound search over the axis's size + 1 boundaries, including the outside cell
  const double *c = SV + h[4];
  for (int b = 0; b < a; ++b) c += h[1 + b] + 1;
  int start = 0, end = size + 1;
  while (start < end) {
    int middle = (start + end) / 2;
    if (value < c[middle]) end = middle;
    else start = middle + 1;
  }
  return start - 1;
}
// VoxelShape::is_full
__device__ __forceinline__ bool kx_full(const unsigned *h,
                                        const unsigned long long *__restrict__ SB, int x, int y,
                                        int z) {
  if (x < 0 || y < 0 || z < 0 || x >= (int)h[1] || y >= (int)h[2] || z >= (int)h[3]) return false;
  unsigned long long i =
      ((unsigned long long)x * h[2] + (unsigned long long)y) * h[3] + (unsigned long long)z;
  return (SB[h[5] + i / 64] >> (i % 64)) & 1ULL;
}
// Whether VoxelShape::clip(start, end, block) gives a hit. Aabb::clip_boxes keeps the nearest one,
// but before any hit its best distance is still 1, so a hit exists if and only if some box and
// axis passes the test with 1 as the best: the first such one answers.
__device__ bool kx_hits(const unsigned *h, const double *__restrict__ SV,
                        const unsigned long long *__restrict__ SB, const double *s, const double *e,
                        const int *block) {
  unsigned boxes = h[7];
  if (boxes == 0) return false;
  double d[3];
#pragma unroll
  for (int a = 0; a < 3; ++a) d[a] = __dsub_rn(e[a], s[a]);
  // Vec3::length_squared: (x x + y y) + z z
  double length =
      __dadd_rn(__dadd_rn(__dmul_rn(d[0], d[0]), __dmul_rn(d[1], d[1])), __dmul_rn(d[2], d[2]));
  if (length < KX_EPS) return false;
  // Inside a full cell of the grid, just past the start
  int cell[3];
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    double nudged = __dadd_rn(s[a], __dmul_rn(d[a], 0.001));
    cell[a] = kx_index(h, SV, a, __dsub_rn(nudged, (double)block[a]));
  }
  if (kx_full(h, SB, cell[0], cell[1], cell[2])) return true;
  const double *b = SV + h[6];
  for (unsigned k = 0; k < boxes; ++k, b += 6) {
    double lo[3], hi[3];
#pragma unroll
    for (int a = 0; a < 3; ++a) {
      lo[a] = __dadd_rn(b[a], (double)block[a]);
      hi[a] = __dadd_rn(b[3 + a], (double)block[a]);
    }
    // Aabb::clip_direction: the face each axis's direction meets
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      int j = (i + 1) % 3, l = (i + 2) % 3;
      double plane;
      if (d[i] > KX_EPS) plane = lo[i];
      else if (d[i] < -KX_EPS) plane = hi[i];
      else continue;
      double t = __ddiv_rn(__dsub_rn(plane, s[i]), d[i]);
      double u = __dadd_rn(s[j], __dmul_rn(t, d[j]));
      double v = __dadd_rn(s[l], __dmul_rn(t, d[l]));
      if (0.0 < t && t < 1.0 && __dsub_rn(lo[j], KX_EPS) < u && u < __dadd_rn(hi[j], KX_EPS) &&
          __dsub_rn(lo[l], KX_EPS) < v && v < __dadd_rn(hi[l], KX_EPS))
        return true;
    }
  }
  return false;
}
// (v.floor() as i64) as f64, saturating
__device__ __forceinline__ double kx_floor_long(double v) {
  double f = floor(v);
  long long l;
  if (isnan(f)) l = 0;
  else if (f >= 9.2233720368547758e18) l = 0x7fffffffffffffffLL;
  else if (f <= -9.2233720368547758e18) l = (long long)0x8000000000000000ULL;
  else l = __double2ll_rz(f);
  return __ll2double_rn(l);
}

extern "C" __global__ void kex_rays(const double *__restrict__ RAYS, int n,
                                    const unsigned short *__restrict__ CELLS, int LX, int LY,
                                    int LZ, int NX, int NY, int NZ, const unsigned *__restrict__ SH,
                                    const double *__restrict__ SV,
                                    const unsigned long long *__restrict__ SB, unsigned char *OUT) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double s[3], e[3];
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    s[a] = RAYS[6 * i + a];
    e[a] = RAYS[6 * i + 3 + a];
  }
  // traverse_blocks
  if (!isfinite(s[0]) || !isfinite(s[1]) || !isfinite(s[2]) || !isfinite(e[0]) || !isfinite(e[1]) ||
      !isfinite(e[2])) {
    OUT[i] = 0;
    return;
  }
  if (s[0] == e[0] && s[1] == e[1] && s[2] == e[2]) {
    OUT[i] = 1;
    return;
  }
  double origin[3], delta[3], step[3], boundary[3];
  int position[3], sign[3];
  bool finite = true;
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    // finish = end.lerp(start, -1e-7), origin = start.lerp(end, -1e-7): x + t * (y - x)
    double finish = __dadd_rn(e[a], __dmul_rn(-1.0e-7, __dsub_rn(s[a], e[a])));
    origin[a] = __dadd_rn(s[a], __dmul_rn(-1.0e-7, __dsub_rn(e[a], s[a])));
    delta[a] = __dsub_rn(finish, origin[a]);
    finite = finite && isfinite(finish) && isfinite(origin[a]) && isfinite(delta[a]);
  }
  if (!finite) {
    OUT[i] = 0;
    return;
  }
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    position[a] = kx_floor(origin[a]);
    sign[a] = delta[a] == 0.0 ? 0 : (delta[a] > 0.0 ? 1 : -1);
    step[a] = sign[a] == 0 ? KX_MAX : __ddiv_rn((double)sign[a], delta[a]);
    double fraction = __dsub_rn(origin[a], kx_floor_long(origin[a]));
    boundary[a] = __dmul_rn(step[a], sign[a] > 0 ? __dsub_rn(1.0, fraction) : fraction);
  }
  for (int visited = 0;; ++visited) {
    if (visited == KX_VISITS) {
      OUT[i] = 0;
      return;
    }
    long long cx = (long long)position[0] - LX, cy = (long long)position[1] - LY;
    long long cz = (long long)position[2] - LZ;
    if (cx < 0 || cy < 0 || cz < 0 || cx >= NX || cy >= NY || cz >= NZ) {
      OUT[i] = 2;
      return;
    }
    const unsigned *h = SH + 8 * (size_t)CELLS[(cx * NY + cy) * NZ + cz];
    if (kx_hits(h, SV, SB, s, e, position)) {
      OUT[i] = 0;
      return;
    }
    if (!(boundary[0] <= 1.0 || boundary[1] <= 1.0 || boundary[2] <= 1.0)) {
      OUT[i] = 1;
      return;
    }
    // Strict comparisons give Z priority on a three-way corner tie
    int axis = boundary[0] < boundary[1] ? (boundary[0] < boundary[2] ? 0 : 2)
                                         : (boundary[1] < boundary[2] ? 1 : 2);
    position[axis] = (int)((unsigned)position[axis] + (unsigned)sign[axis]);
    boundary[axis] = __dadd_rn(boundary[axis], step[axis]);
  }
}
