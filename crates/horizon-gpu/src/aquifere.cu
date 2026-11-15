// Aquiferes de Horizon (NoiseBasedAquifer, 26.3), operation par operation.
// Champs evalues comme sample_value : AQ_BARRIER, AQ_FLOOD, AQ_SPREAD, AQ_LAVA, AQ_EXCL, AQ_SURFACE
// (#define vers les fonctions generees). Calculs en double comme le CPU, sans contraction FMA.
//
// Code d'un bloc : 0 = densite > 0 (bloc par defaut, ecrit en suites par fill_noise), 1 = air,
// 2 = fluide par defaut, 3 = lave, 8 = bloc par defaut decide par l'aquifere (pression),
// plus 4 si la mise a jour du fluide est planifiee (should_schedule_fluid_update) ; -1 hors grille.
#define AQ_AUCUN 0
#define AQ_AIR 1
#define AQ_EAU 2
#define AQ_LAVE 3
#define AQ_PRESSION 8
#define WAY_BELOW_MIN_Y (-32512)
#define FLOWING_UPDATE_SIMILARITY (1.0 - 44.0 / 25.0)

struct Statut {
  int niveau;
  int etat;
};
// GlobalFluidPicker : lave sous min(-54, niveau de la mer), sinon le fluide par defaut au niveau de la mer
__device__ __forceinline__ Statut aq_picker(int y, int mer) {
  Statut s;
  int lim = -54 < mer ? -54 : mer;
  if (y < lim) { s.niveau = -54; s.etat = AQ_LAVE; } else { s.niveau = mer; s.etat = AQ_EAU; }
  return s;
}
// FluidStatus::at
__device__ __forceinline__ int aq_a(Statut s, int y) { return y < s.niveau ? s.etat : AQ_AIR; }
__device__ __forceinline__ bool aq_egal(Statut a, Statut b) { return a.niveau == b.niveau && a.etat == b.etat; }

// Position tiree d'une cellule (load_cell), relue par BlockPos::pack/unpack
__device__ void aq_cellule(int cx, int cy, int cz, unsigned long long LO, unsigned long long HI, int *p) {
  unsigned long long lo = (unsigned long long)aq_position_seed(cx, cy, cz) ^ LO, hi = HI;
  if ((lo | hi) == 0ull) { lo = 0x9e3779b97f4a7c15ull; hi = 0x6a09e667f3bcc909ull; }
  int x = (int)((unsigned)cx << 4) + aq_next_int_bound(lo, hi, 10);
  int y = (int)((unsigned)cy * 12u) + aq_next_int_bound(lo, hi, 9);
  int z = (int)((unsigned)cz << 4) + aq_next_int_bound(lo, hi, 10);
  long long l = (long long)((((unsigned long long)(unsigned)x & 0x3ffffffull) << 38) | (((unsigned long long)(unsigned)z & 0x3ffffffull) << 12) | ((unsigned long long)(unsigned)y & 0xfffull));
  p[0] = (int)(l >> 38);
  p[1] = (int)((l << 52) >> 52);
  p[2] = (int)((l << 26) >> 38);
}

// Niveaux de surface des colonnes alignees sur 4 : floor(surface_level(x, 0, z))
struct AqSurface {
  const int *S;
  int qx0, qz0, nqx;
};
__device__ __forceinline__ int aq_surface(const AqSurface &s, int x, int z) {
  return s.S[((z >> 2) - s.qz0) * s.nqx + ((x >> 2) - s.qx0)];
}
__device__ __forceinline__ double aq_clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

// compute_surface_level
__device__ int aq_niveau(int x, int y, int z, Statut global, int minimum, bool fluide_centre) {
  double sec, inonde;
  if ((double)AQ_EXCL(x, y, z) > 0.0) {
    sec = -1.0;
    inonde = -1.0;
  } else {
    int distance = minimum + 8 - y;
    double facteur;
    if (fluide_centre) {
      double alpha = ((double)distance - 0.0) / (64.0 - 0.0);
      facteur = alpha < 0.0 ? 1.0 : (alpha > 1.0 ? 0.0 : 1.0 + alpha * (0.0 - 1.0));
    } else {
      facteur = 0.0;
    }
    double valeur = aq_clamp((double)AQ_FLOOD(x, y, z), -1.0, 1.0);
    double alpha = (facteur - 1.0) / (0.0 - 1.0);
    double seuil_inonde = -0.3 + alpha * (0.8 - -0.3);
    double seuil_sec = -0.8 + alpha * (0.4 - -0.8);
    sec = valeur - seuil_sec;
    inonde = valeur - seuil_inonde;
  }
  if (inonde > 0.0) return global.niveau;
  if (sec > 0.0) {
    float etal = __fmul_rn(AQ_SPREAD(fdiv(x, 16), fdiv(y, 40), fdiv(z, 16)), 10.0f);
    int decalage = (int)((unsigned)floor_sat((double)etal / 3.0) * 3u);
    int base = fdiv(y, 40) * 40 + 20 + decalage;
    return minimum < base ? minimum : base;
  }
  return WAY_BELOW_MIN_Y;
}

// compute_fluid : le statut d'une cellule a sa position tiree
__device__ Statut aq_fluide(int x, int y, int z, const AqSurface &s, int mer) {
  const int DX[13] = {0, -2, -1, 0, 1, -3, -2, -1, 1, -2, -1, 0, 1};
  const int DZ[13] = {0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};
  Statut global = aq_picker(y, mer);
  int haut = y + 12, bas = y - 12;
  int minimum = 2147483647;
  bool fluide_centre = false;
  for (int i = 0; i < 13; ++i) {
    int sx = x + DX[i] * 16, sz = z + DZ[i] * 16;
    int surface = aq_surface(s, sx, sz);
    int ajuste = surface + 8;
    bool centre = DX[i] == 0 && DZ[i] == 0;
    if (centre && bas > ajuste) return global;
    bool dessus = haut > ajuste;
    if (dessus || centre) {
      Statut st = aq_picker(ajuste, mer);
      if (aq_a(st, ajuste) != AQ_AIR) {
        if (centre) fluide_centre = true;
        if (dessus) return st;
      }
    }
    minimum = surface < minimum ? surface : minimum;
  }
  int niveau = aq_niveau(x, y, z, global, minimum, fluide_centre);
  int etat = global.etat;
  if (niveau <= -10 && niveau != WAY_BELOW_MIN_Y && etat != AQ_LAVE && fabs((double)AQ_LAVA(fdiv(x, 64), fdiv(y, 40), fdiv(z, 64))) > 0.3) {
    etat = AQ_LAVE;
  }
  Statut r;
  r.niveau = niveau;
  r.etat = etat;
  return r;
}

__device__ __forceinline__ double aq_similarite(int a, int b) { return 1.0 - (double)(b - a) / 25.0; }

__device__ double aq_pression(int x, int y, int z, double *memo, Statut a, Statut b) {
  int sa = aq_a(a, y), sb = aq_a(b, y);
  if ((sa == AQ_LAVE && sb == AQ_EAU) || (sa == AQ_EAU && sb == AQ_LAVE)) return 2.0;
  int ecart = a.niveau - b.niveau;
  ecart = ecart < 0 ? -ecart : ecart;
  if (ecart == 0) return 0.0;
  double milieu = 0.5 * (double)(a.niveau + b.niveau);
  double dy = (double)y + 0.5 - milieu;
  double bord = (double)ecart / 2.0 - fabs(dy);
  double q;
  if (dy > 0.0) {
    double n = 0.0 + bord;
    q = n > 0.0 ? n / 1.5 : n / 2.5;
  } else {
    double n = 3.0 + bord;
    q = n > 0.0 ? n / 3.0 : n / 10.0;
  }
  double barriere;
  if (q < -2.0 || q > 2.0) {
    barriere = 0.0;
  } else {
    if (isnan(*memo)) *memo = (double)AQ_BARRIER(x, y, z);
    barriere = *memo;
  }
  return 2.0 * (barriere + q);
}

// Grille des cellules : positions (3 entiers) et statuts (niveau, etat) par cellule
struct AqGrille {
  const int *P;
  const int *ST;
  int cx0, cy0, cz0, ncx, ncy, ncz;
};
__device__ __forceinline__ Statut aq_statut(const AqGrille &g, int i) {
  Statut s;
  s.niveau = g.ST[2 * i];
  s.etat = g.ST[2 * i + 1];
  return s;
}

// compute_substance (closest puis resolve) ; -1 si une cellule sort de la grille
__device__ int aq_substance(int x, int y, int z, float densite_f, int saut, const AqGrille &g, int mer) {
  double densite = (double)densite_f;
  if (densite > 0.0) return AQ_AUCUN;
  Statut global = aq_picker(y, mer);
  if (y > saut) return aq_a(global, y);
  if (aq_a(global, y) == AQ_LAVE) return AQ_LAVE;
  int gx = (x - 5) >> 4, gy = fdiv(y + 1, 12), gz = (z - 5) >> 4;
  int dist[4] = {2147483647, 2147483647, 2147483647, 2147483647};
  int idx[4] = {0, 0, 0, 0};
  for (int dx = 0; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dz = 0; dz <= 1; ++dz) {
        int ix = gx + dx - g.cx0, iy = gy + dy - g.cy0, iz = gz + dz - g.cz0;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= g.ncx || iy >= g.ncy || iz >= g.ncz) return -1;
        int i = ix + g.ncx * (iy + g.ncy * iz);
        int ax = g.P[3 * i] - x, ay = g.P[3 * i + 1] - y, az = g.P[3 * i + 2] - z;
        int d = (int)((unsigned)ax * (unsigned)ax + (unsigned)az * (unsigned)az + (unsigned)ay * (unsigned)ay);
        if (d <= dist[0]) {
          dist[3] = dist[2]; dist[2] = dist[1]; dist[1] = dist[0]; dist[0] = d;
          idx[3] = idx[2]; idx[2] = idx[1]; idx[1] = idx[0]; idx[0] = i;
        } else if (d <= dist[1]) {
          dist[3] = dist[2]; dist[2] = dist[1]; dist[1] = d;
          idx[3] = idx[2]; idx[2] = idx[1]; idx[1] = i;
        } else if (d <= dist[2]) {
          dist[3] = dist[2]; dist[2] = d;
          idx[3] = idx[2]; idx[2] = i;
        } else if (d <= dist[3]) {
          dist[3] = d;
          idx[3] = i;
        }
      }
  Statut a = aq_statut(g, idx[0]);
  double s01 = aq_similarite(dist[0], dist[1]);
  int candidat = aq_a(a, y);
  if (s01 <= 0.0) {
    bool plan = s01 >= FLOWING_UPDATE_SIMILARITY ? !aq_egal(a, aq_statut(g, idx[1])) : false;
    return candidat | (plan ? 4 : 0);
  }
  if (candidat == AQ_EAU && aq_a(aq_picker(y - 1, mer), y - 1) == AQ_LAVE) return candidat | 4;
  double memo = __longlong_as_double(0x7ff8000000000000ll);
  Statut b = aq_statut(g, idx[1]);
  if (densite + s01 * aq_pression(x, y, z, &memo, a, b) > 0.0) return AQ_PRESSION;
  Statut c = aq_statut(g, idx[2]);
  double s02 = aq_similarite(dist[0], dist[2]);
  if (s02 > 0.0 && densite + s01 * s02 * aq_pression(x, y, z, &memo, a, c) > 0.0) return AQ_PRESSION;
  double s12 = aq_similarite(dist[1], dist[2]);
  if (s12 > 0.0 && densite + s01 * s12 * aq_pression(x, y, z, &memo, b, c) > 0.0) return AQ_PRESSION;
  bool plan = !aq_egal(a, b) || (s12 >= FLOWING_UPDATE_SIMILARITY && !aq_egal(b, c)) ||
              (s02 >= FLOWING_UPDATE_SIMILARITY && !aq_egal(a, c)) ||
              (s02 >= FLOWING_UPDATE_SIMILARITY && aq_similarite(dist[0], dist[3]) >= FLOWING_UPDATE_SIMILARITY &&
               !aq_egal(a, aq_statut(g, idx[3])));
  return candidat | (plan ? 4 : 0);
}

// ---------- noyaux ----------
extern "C" __global__ void kaq_surface(int *S, int qx0, int qz0, int nqx, int nqz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nqx * nqz) return;
  int qx = qx0 + i % nqx, qz = qz0 + i / nqx;
  S[i] = floor_sat((double)AQ_SURFACE((int)((unsigned)qx << 2), 0, (int)((unsigned)qz << 2)));
}
extern "C" __global__ void kaq_cellules(int *P, int *ST, const int *S, int qx0, int qz0, int nqx, int cx0, int cy0, int cz0,
                                        int ncx, int ncy, int ncz, unsigned long long LO, unsigned long long HI, int mer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncx * ncy * ncz) return;
  int cx = cx0 + i % ncx, cy = cy0 + (i / ncx) % ncy, cz = cz0 + i / (ncx * ncy);
  int p[3];
  aq_cellule(cx, cy, cz, LO, HI, p);
  P[3 * i] = p[0];
  P[3 * i + 1] = p[1];
  P[3 * i + 2] = p[2];
  AqSurface s;
  s.S = S;
  s.qx0 = qx0;
  s.qz0 = qz0;
  s.nqx = nqx;
  Statut st = aq_fluide(p[0], p[1], p[2], s, mer);
  ST[2 * i] = st.niveau;
  ST[2 * i + 1] = st.etat;
}
// skip_sampling_above_y de chaque chunk (NoiseBasedAquifer::new et max_surface_level)
extern "C" __global__ void kaq_saut(int *SAUT, const int *S, int qx0, int qz0, int nqx, int chx0, int chz0, int nchx, int nch, int miny, int h) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nch) return;
  int x0 = (chx0 + c % nchx) * 16, z0 = (chz0 + c / nchx) * 16;
  int mnx = (x0 - 5) >> 4, mnz = (z0 - 5) >> 4;
  int mxx = ((x0 + 15 - 5) >> 4) + 1, mxz = ((z0 + 15 - 5) >> 4) + 1;
  int lx = (mnx * 16) >> 2, lz = (mnz * 16) >> 2, hx = (mxx * 16 + 9) >> 2, hz = (mxz * 16 + 9) >> 2;
  int m = -2147483647 - 1;
  for (int qz = lz; qz <= hz; ++qz)
    for (int qx = lx; qx <= hx; ++qx) {
      int v = S[(qz - qz0) * nqx + (qx - qx0)];
      m = v > m ? v : m;
    }
  int max_surface = m + 8;
  int grille = fdiv(max_surface + 12, 12) + 1;
  SAUT[c] = grille * 12 + 11 - 1;
}
// Un thread par bloc : chunks a la suite (chunk = cz * nchx + cx), ordre de fill_noise dans le chunk
extern "C" __global__ void kaq_blocs(const float *DENS, signed char *CODES, const int *P, const int *ST, const int *SAUT,
                                     int chx0, int chz0, int nchx, int nch, int miny, int h, int cx0, int cy0, int cz0,
                                     int ncx, int ncy, int ncz, int mer) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long par_chunk = 256L * h;
  if (i >= par_chunk * nch) return;
  int c = (int)(i / par_chunk), r = (int)(i % par_chunk);
  int lc = r / h, ly = r % h;
  int x = (chx0 + c % nchx) * 16 + (lc & 15), z = (chz0 + c / nchx) * 16 + (lc >> 4), y = miny + ly;
  AqGrille g;
  g.P = P;
  g.ST = ST;
  g.cx0 = cx0;
  g.cy0 = cy0;
  g.cz0 = cz0;
  g.ncx = ncx;
  g.ncy = ncy;
  g.ncz = ncz;
  CODES[i] = (signed char)aq_substance(x, y, z, DENS[i], SAUT[c], g, mer);
}

// Une region du moteur : densite et codes dans la disposition de la grille de blocs
// (indice ((z - Z0) * GX + (x - X0)) * h + (y - miny)) ; SAUT par chunk de la region (cz * nchx + cx).
extern "C" __global__ void kaq_blocs_region(const float *DENS, signed char *CODES, const int *P, const int *ST, const int *SAUT,
                                            int X0, int Z0, int GX, int GZ, int nchx, int miny, int h, int cx0, int cy0, int cz0,
                                            int ncx, int ncy, int ncz, int mer) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)GX * GZ * h) return;
  long colonne = i / h;
  int iy = (int)(i % h), ix = (int)(colonne % GX), iz = (int)(colonne / GX);
  AqGrille g;
  g.P = P;
  g.ST = ST;
  g.cx0 = cx0;
  g.cy0 = cy0;
  g.cz0 = cz0;
  g.ncx = ncx;
  g.ncy = ncy;
  g.ncz = ncz;
  CODES[i] = (signed char)aq_substance(X0 + ix, miny + iy, Z0 + iz, DENS[i], SAUT[(iz >> 4) * nchx + (ix >> 4)], g, mer);
}
// Les codes d'un chunk de la region, dans l'ordre de fill_noise (colonne x + z * 16, puis Y)
extern "C" __global__ void kaq_chunk(const signed char *CODES, signed char *OUT, int lx, int lz, int GX, int h) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 256L * h) return;
  int lc = (int)(i / h), iy = (int)(i % h);
  int ix = lx * 16 + (lc & 15), iz = lz * 16 + (lc >> 4);
  OUT[i] = CODES[((long)iz * GX + ix) * h + iy];
}

// ---------- Beardifier (terrain autour des structures), Beardifier::sample_unchecked de Horizon ----------
// Pieces rigides : 8 entiers (min x y z, max x y z, ajustement, ground_level_delta) ; ajustement :
// 0 None, 1 Bury, 2 BeardThin, 3 BeardBox, 4 Encapsulate. Jonctions : 3 entiers (x, sol, z).
__device__ __forceinline__ float bd_bury(float x, float y, float z) {
  float sq = __fadd_rn(__fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y)), __fmul_rn(z, z));
  if (sq >= 36.0f) return 0.0f;
  return __fsub_rn(1.0f, __fdiv_rn(__fsqrt_rn(sq), 6.0f));
}
// fast_inverse_sqrt (Mth.fastInvSqrt en double)
__device__ __forceinline__ double bd_fisr(double v) {
  double moitie = 0.5 * v;
  long long bits = 6910469410427058090LL - (__double_as_longlong(v) >> 1);
  double e = __longlong_as_double(bits);
  return e * (1.5 - moitie * e * e);
}
__device__ __forceinline__ float bd_beard(const float *K, int x, int y, int z, int ecart_sol) {
  int ix = x + 12, iy = y + 12, iz = z + 12;
  if (ix < 0 || ix >= 24 || iy < 0 || iy >= 24 || iz < 0 || iz >= 24) return 0.0f;
  float vertical = __fadd_rn((float)ecart_sol, 0.5f);
  float fx = (float)x, fz = (float)z;
  float sq = __fadd_rn(__fadd_rn(__fmul_rn(fx, fx), __fmul_rn(vertical, vertical)), __fmul_rn(fz, fz));
  float f = __fdiv_rn(__fmul_rn(-vertical, (float)bd_fisr((double)__fdiv_rn(sq, 2.0f))), 2.0f);
  return __fmul_rn(f, K[(iz * 24 + ix) * 24 + iy]);
}
__device__ float bd_valeur(int x, int y, int z, const int *PC, int np, const int *JC, int nj, const float *K) {
  float v = 0.0f;
  for (int i = 0; i < np; ++i) {
    const int *p = PC + 8 * i;
    int dx = max(0, max(p[0] - x, x - p[3]));
    int dz = max(0, max(p[2] - z, z - p[5]));
    int sol = p[1] + p[7];
    int dy = y - sol;
    int adj = p[6];
    int dist_y = adj == 0 ? 0 : ((adj == 1 || adj == 2) ? dy : (adj == 3 ? max(0, max(sol - y, y - p[4])) : max(0, max(p[1] - y, y - p[4]))));
    float c;
    if (adj == 0) c = 0.0f;
    else if (adj == 1) c = bd_bury((float)dx, __fdiv_rn((float)dist_y, 2.0f), (float)dz);
    else if (adj == 2 || adj == 3) c = __fmul_rn(bd_beard(K, dx, dist_y, dz, dy), 0.8f);
    else c = __fmul_rn(bd_bury(__fdiv_rn((float)dx, 2.0f), __fdiv_rn((float)dist_y, 2.0f), __fdiv_rn((float)dz, 2.0f)), 0.8f);
    v = __fadd_rn(v, c);
  }
  for (int j = 0; j < nj; ++j) {
    const int *q = JC + 3 * j;
    int dx = x - q[0], dy = y - q[1], dz = z - q[2];
    v = __fadd_rn(v, __fmul_rn(bd_beard(K, dx, dy, dz, dy), 0.4f));
  }
  return v;
}
// Un chunk a structures : densite de la region + beardifier, puis la decision d'aquifere du bloc.
// Z = boite des blocs affectes (6 entiers) ; avec_zone = 0 si le beardifier est vide.
extern "C" __global__ void kbeard_chunk(const float *DENS, signed char *OUT, const int *PC, int np, const int *JC, int nj,
                                        const float *K, const int *Z, int avec_zone, const int *P, const int *ST, int saut,
                                        int lx, int lz, int X0, int Z0, int GX, int miny, int h, int cx0, int cy0, int cz0,
                                        int ncx, int ncy, int ncz, int mer) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 256L * h) return;
  int lc = (int)(i / h), iy = (int)(i % h);
  int ixr = lx * 16 + (lc & 15), izr = lz * 16 + (lc >> 4);
  int x = X0 + ixr, y = miny + iy, z = Z0 + izr;
  float d = DENS[((long)izr * GX + ixr) * h + iy];
  float b = 0.0f;
  if (avec_zone && x >= Z[0] && y >= Z[1] && z >= Z[2] && x <= Z[3] && y <= Z[4] && z <= Z[5]) b = bd_valeur(x, y, z, PC, np, JC, nj, K);
  AqGrille g;
  g.P = P;
  g.ST = ST;
  g.cx0 = cx0;
  g.cy0 = cy0;
  g.cz0 = cz0;
  g.ncx = ncx;
  g.ncy = ncy;
  g.ncz = ncz;
  OUT[i] = (signed char)aq_substance(x, y, z, __fadd_rn(d, b), saut, g, mer);
}

// ---------- grottes ----------
// Substance des cellules creusees d'un chunk (compute_substance avec une densite 0, comme
// apply_carving_mask), rangees dans l'ordre des bits du masque : CODES[rang du bit] = code.
// Un bloc de 256 threads par paquet de 32 mots ; le rang de depart d'un paquet est la somme des bits
// des mots precedents. Masque : bit (y - ymin) + (z + x * 16) * hauteur.
extern "C" __global__ void kgr_substance(const unsigned long long *M, signed char *CODES, int nw, int cap, int ymin,
                                         int hauteur, int X0, int Z0, const int *P, const int *ST, int saut,
                                         int cx0, int cy0, int cz0, int ncx, int ncy, int ncz, int mer) {
  __shared__ int BASE[8];
  __shared__ int PRE[32];
  int t = threadIdx.x, w0 = blockIdx.x * 32;
  int s = 0;
  for (int i = t; i < w0; i += blockDim.x) s += __popcll(M[i]);
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
  if ((t & 31) == 0) BASE[t >> 5] = s;
  if (t < 32) {
    int w = w0 + t;
    int c = w < nw ? __popcll(M[w]) : 0;
    int inc = c;
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xffffffff, inc, o);
      if (t >= o) inc += v;
    }
    PRE[t] = inc - c;
  }
  __syncthreads();
  int base = 0;
  for (int k = 0; k < (int)(blockDim.x >> 5); ++k) base += BASE[k];
  AqGrille g;
  g.P = P;
  g.ST = ST;
  g.cx0 = cx0;
  g.cy0 = cy0;
  g.cz0 = cz0;
  g.ncx = ncx;
  g.ncy = ncy;
  g.ncz = ncz;
  for (int b = t; b < 32 * 64; b += blockDim.x) {
    int w = w0 + (b >> 6), bit = b & 63;
    if (w >= nw) break;
    unsigned long long mot = M[w];
    if (!((mot >> bit) & 1ULL)) continue;
    int rang = base + PRE[b >> 6] + __popcll(mot & ((1ULL << bit) - 1ULL));
    if (rang >= cap) continue;
    int i = w * 64 + bit;
    int col = i / hauteur, y = ymin + i % hauteur;
    int x = X0 + ((col >> 4) & 15), z = Z0 + (col & 15);
    CODES[rang] = (signed char)aq_substance(x, y, z, 0.0f, saut, g, mer);
  }
}
