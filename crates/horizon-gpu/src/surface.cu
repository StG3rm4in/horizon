// Surface de Horizon sur GPU : entrees par bloc (biome, temperature), operation par operation.
// Necessite base.cu (GRAD, floor_i32) et les tables PERM / OFF ; les indices des bruits de
// temperature sont donnes par #define : T_HAUTEUR, T_INFO, T_GEL0..2 (et T_GEL_FREQ_k, T_GEL_AMP_k).

// ---------- Zoom des biomes (BiomeZoom::quart_at_block) ----------
__device__ __forceinline__ long long sf_g(long long m) {
  unsigned long long u = (unsigned long long)m;
  return (long long)(u * (u * 6364136223846793005ull + 1442695040888963407ull));
}
__device__ __forceinline__ double sf_fiddle(long long s) {
  return ((double)((s >> 24) & 1023) / 1024.0 - 0.5) * 0.9;
}
__device__ void sf_quart(long long graine, int bx, int by, int bz, int *q) {
  int sx = bx - 2, sy = by - 2, sz = bz - 2;
  int base[3] = {sx >> 2, sy >> 2, sz >> 2};
  double fr[3] = {(double)(sx & 3) / 4.0, (double)(sy & 3) / 4.0, (double)(sz & 3) / 4.0};
  long long c0 = base[0], c1 = base[1], c2 = base[2];
  long long racine = sf_g(graine);
  double meilleur = __longlong_as_double(0x7ff0000000000000ll);
  int sel[3] = {base[0], base[1], base[2]};
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z) {
        long long m0 = (long long)((unsigned long long)racine + (unsigned long long)(c0 + x));
        long long m1 = (long long)((unsigned long long)sf_g(m0) + (unsigned long long)(c1 + y));
        long long m2 = (long long)((unsigned long long)sf_g(m1) + (unsigned long long)(c2 + z));
        long long m3 = (long long)((unsigned long long)sf_g(m2) + (unsigned long long)(c0 + x));
        long long m4 = (long long)((unsigned long long)sf_g(m3) + (unsigned long long)(c1 + y));
        long long m5 = (long long)((unsigned long long)sf_g(m4) + (unsigned long long)(c2 + z));
        long long m6 = (long long)((unsigned long long)sf_g(m5) + (unsigned long long)graine);
        long long m7 = (long long)((unsigned long long)sf_g(m6) + (unsigned long long)graine);
        double dx = fr[0] - (double)x + sf_fiddle(m5);
        double dy = fr[1] - (double)y + sf_fiddle(m6);
        double dz = fr[2] - (double)z + sf_fiddle(m7);
        double d = dz * dz + dy * dy + dx * dx;
        if (meilleur > d) {
          meilleur = d;
          sel[0] = base[0] + x;
          sel[1] = base[1] + y;
          sel[2] = base[2] + z;
        }
      }
  q[0] = sel[0];
  q[1] = sel[1];
  q[2] = sel[2];
}
// Biome d'un quart dans les 3 x 3 chunks autour du chunk (ccx, ccz) : B = sections des chunks a la
// suite (dz puis dx, de -1 a +1), chaque chunk en [quart y][z & 3][x & 3] ; y borne au chunk.
// -1 si le quart sort des 3 x 3 chunks.
__device__ int sf_biome_quart(const unsigned short *B, int ccx, int ccz, int qbas, int nqy, const int *q) {
  int dxc = (q[0] >> 2) - (ccx - 1), dzc = (q[2] >> 2) - (ccz - 1);
  if (dxc < 0 || dxc > 2 || dzc < 0 || dzc > 2) return -1;
  int y = q[1] < qbas ? qbas : (q[1] > qbas + nqy - 1 ? qbas + nqy - 1 : q[1]);
  return B[((dzc * 3 + dxc) * nqy + (y - qbas)) * 16 + (((q[2] & 3) << 2) | (q[0] & 3))];
}

// ---------- Bruit simplex 2D (SimplexNoise::get2) ----------
__device__ __forceinline__ double sf_coin(int g, double p0, double p1, double p2, double rayon) {
  double att = rayon - p0 * p0 - p1 * p1 - p2 * p2;
  if (att < 0.0) return 0.0;
  double sq = att * att;
  return sq * sq * ((double)GRAD[g][0] * p0 + (double)GRAD[g][1] * p1 + (double)GRAD[g][2] * p2);
}
__device__ float sf_simplex2(const unsigned char *P, const double *O, double x, double y) {
  x = x + O[0];
  y = y + O[1];
  double r3 = sqrt(3.0);
  double biais = 0.5 * (r3 - 1.0);
  double debiais = (3.0 - r3) / 6.0;
  double dec = (x + y) * biais;
  int i = floor_i32(x + dec), j = floor_i32(y + dec);
  double dec2 = (double)(i + j) * debiais;
  double dx = x - ((double)i - dec2), dy = y - ((double)j - dec2);
  int a = dx > dy ? 1 : 0, b = dx > dy ? 0 : 1;
  i &= 255;
  j &= 255;
#define SF_GRAD(ea, eb) ((int)P[(i + (ea) + (int)P[(j + (eb)) & 255]) & 255] % 12)
  double premier = sf_coin(SF_GRAD(0, 0), dx, dy, 0.0, 0.5);
  double second = sf_coin(SF_GRAD(a, b), dx - (double)a + debiais, dy - (double)b + debiais, 0.0, 0.5);
  double troisieme = sf_coin(SF_GRAD(1, 1), dx - 1.0 + 2.0 * debiais, dy - 1.0 + 2.0 * debiais, 0.0, 0.5);
#undef SF_GRAD
  return (float)(70.0 * (premier + second + troisieme));
}

// ---------- Temperature (BiomeTemperatureSettings::height_adjusted) ----------
__device__ float sf_gel(double x, double z) {
  float v = 0.0f;
  v = __fadd_rn(v, __fmul_rn(T_GEL_AMP_0, sf_simplex2(PERM[T_GEL0], OFF[T_GEL0], x * T_GEL_FREQ_0, z * T_GEL_FREQ_0)));
  v = __fadd_rn(v, __fmul_rn(T_GEL_AMP_1, sf_simplex2(PERM[T_GEL1], OFF[T_GEL1], x * T_GEL_FREQ_1, z * T_GEL_FREQ_1)));
  v = __fadd_rn(v, __fmul_rn(T_GEL_AMP_2, sf_simplex2(PERM[T_GEL2], OFF[T_GEL2], x * T_GEL_FREQ_2, z * T_GEL_FREQ_2)));
  return v;
}
// base : temperature de base du biome ; gele : modificateur Frozen
__device__ float sf_temperature(float base, int gele, int x, int y, int z, int mer) {
  if (gele) {
    double fx = (double)x, fz = (double)z;
    double gel = (double)__fmul_rn(sf_gel(fx * 0.05, fz * 0.05), 7.0f);
    double info = (double)sf_simplex2(PERM[T_INFO], OFF[T_INFO], fx * 0.2, fz * 0.2);
    if (gel + info < 0.3 && (double)sf_simplex2(PERM[T_INFO], OFF[T_INFO], fx * 0.09, fz * 0.09) < 0.8) base = 0.2f;
  }
  int debut = mer + 17;
  if (y > debut) {
    float variation = __fmul_rn(sf_simplex2(PERM[T_HAUTEUR], OFF[T_HAUTEUR], (double)__fdiv_rn((float)x, 8.0f), (double)__fdiv_rn((float)z, 8.0f)), 8.0f);
    return __fsub_rn(base, __fdiv_rn(__fmul_rn(__fsub_rn(__fadd_rn(variation, (float)y), (float)debut), 0.05f), 40.0f));
  }
  return base;
}

// ---------- noyaux de verification (jalon S2) ----------
extern "C" __global__ void ksf_quarts(const int *P, int *Q, int n, long long graine) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  sf_quart(graine, P[3 * i], P[3 * i + 1], P[3 * i + 2], Q + 3 * i);
}
extern "C" __global__ void ksf_temperatures(const int *P, const float *BASES, const int *GELES, int nb, float *T, int n, int mer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nb) return;
  int k = i / n, p = i % n;
  T[i] = sf_temperature(BASES[k], GELES[k], P[3 * p], P[3 * p + 1], P[3 * p + 2], mer);
}

#ifdef SF_MINY
// ---------- Descente des colonnes (material_stage::build_columns, jalon S3) ----------
// Codes du remplissage : & 3 = 0 solide (et 8), 1 air, 2 ou 3 fluide.
#define SF_SOLIDE 0
#define SF_AIR 1
#define SF_FLUIDE 2
struct Ctx {
  int x, y, z, depth, gx, gz, above, below, water, minimum, biome, bloc, erreur;
  double secondary;
  const unsigned short *B;
  int cx, cz;
  // Filons de la region : position du bloc dans la region (SF_VR x SF_VH x SF_VR a partir de SF_VLO)
  const float *VOL;
  int vx, vz;
};
__device__ __forceinline__ int sf_genre(const signed char *col, int y) {
  if (y < SF_MINY || y > SF_MINY + SF_H - 1) return SF_AIR;
  int c = col[y - SF_MINY] & 3;
  return c == 0 ? SF_SOLIDE : (c == 1 ? SF_AIR : SF_FLUIDE);
}
__device__ int sf_biome(Ctx &c) {
  if (c.biome == -2) {
    int q[3];
    sf_quart(SF_ZOOM, c.x, c.y, c.z, q);
    c.biome = sf_biome_quart(c.B, c.cx, c.cz, SF_MINY >> 2, SF_H >> 2, q);
  }
  return c.biome;
}

#ifdef SF_EXT
// ---------- Extensions (material_extensions.rs, jalon S4) ----------
#define SF_RIEN (-2147483647 - 1)
__device__ __forceinline__ double sf_jmin_d(double a, double b) {
  if (isnan(a)) return a;
  if (isnan(b)) return b;
  if (a == 0.0 && b == 0.0) return __longlong_as_double(__double_as_longlong(a) | __double_as_longlong(b));
  return a <= b ? a : b;
}
// Code du remplissage en y (& 3 : 0 bloc par defaut, 1 air, 2 eau, 3 lave) ; air hors du monde.
__device__ __forceinline__ int sf_code(const signed char *col, int y) {
  if (y < SF_MINY || y > SF_MINY + SF_H - 1) return 1;
  return col[y - SF_MINY] & 3;
}
// MaterialExtensions::eroded_badlands : le haut du pilier (rempli de otop a ce haut), ou SF_RIEN.
__device__ int sf_pilier(const signed char *col, int x, int z, int otop) {
  double dx = (double)x, dz = (double)z;
  double amp = sf_jmin_d(fabs((double)BD_S(dx, 0.0, dz) * 8.25), (double)__fmul_rn(BD_P(dx * 0.2, 0.0, dz * 0.2), 15.0f));
  if (amp <= 0.0) return SF_RIEN;
  double toit = fabs((double)BD_R(dx * 0.75, 0.0, dz * 0.75) * 1.5);
  int haut = floor_sat(64.0 + sf_jmin_d(amp * amp * 2.5, ceil(toit * 50.0) + 24.0));
  if (otop > haut) return SF_RIEN;
  // Le premier bloc par defaut ou d'eau sous le haut : de l'eau, pas de pilier
  for (int y = haut; y >= SF_MINY; --y) {
    int k = sf_code(col, y);
    if (k == 0) break;
    if (k == 2) return SF_RIEN;
  }
  return haut;
}
// MaterialExtensions::frozen_ocean_with_melt, apres la descente : ecrit glace tassee et neige dans `sortie`
// (etat + 1), renvoie le plus haut y ecrit (SF_RIEN si aucun). b : biome de la colonne (en son premier y libre).
__device__ int sf_iceberg(const signed char *col, unsigned *sortie, int x, int z, int otop, int minimum, int b,
                          unsigned long long RLO, unsigned long long RHI, int *erreur) {
  double dx = (double)x, dz = (double)z;
  double amp = sf_jmin_d(fabs((double)IC_S(dx, 0.0, dz) * 8.25), (double)__fmul_rn(IC_P(dx * 1.28, 0.0, dz * 1.28), 15.0f));
  if (amp <= 1.8) return SF_RIEN;
  double toit = fabs((double)IC_R(dx * 1.17, 0.0, dz * 1.17) * 1.5);
  double h = sf_jmin_d(amp * amp * 1.2, ceil(toit * 40.0) + 14.0);
  if (b >= NTB) {
    *erreur = 1;
    return SF_RIEN;
  }
  // should_melt_frozen_ocean_iceberg_slightly : temperature au niveau de la mer > 0.1
  if (sf_temperature(__int_as_float((int)TB[b]), TG[b], x, SF_MER, z, SF_MER) > 0.1f) h -= 2.0;
  if (h <= 2.0) return SF_RIEN;
  int ibas = (int)((double)SF_MER - h - 7.0), ihaut = (int)(h + (double)SF_MER);
  unsigned long long lo, hi;
  aq_at(RLO, RHI, x, 0, z, lo, hi);
  int limite = 2 + aq_next_int_bound(lo, hi, 4);
  int hneige = (int)((unsigned)SF_MER + 18u + (unsigned)aq_next_int_bound(lo, hi, 10));
  int n = 0, plus = SF_RIEN;
  int depart = (int)((unsigned)ihaut + 1u);
  if (otop > depart) depart = otop;
  for (int y = depart; y >= minimum; --y) {
    int k = sf_code(col, y);
    // Les tirages suivent les courts-circuits de Horizon
    bool remplir = k == 1 && y < ihaut && aq_next_double(lo, hi) > 0.01;
    if (!remplir) remplir = k == 2 && y > ibas && y < SF_MER && aq_next_double(lo, hi) > 0.15;
    if (remplir) {
      int s;
      if (n <= limite && y > hneige) {
        n += 1;
        s = SF_NEIGE;
      } else {
        s = SF_GLACE;
      }
      if (y >= SF_MINY && y <= SF_MINY + SF_H - 1) {
        sortie[y - SF_MINY] = (unsigned)s + 1u;
        if (y > plus) plus = y;
      }
    }
  }
  return plus;
}
#endif
#endif
