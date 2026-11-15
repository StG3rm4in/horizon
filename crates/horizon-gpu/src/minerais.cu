// Blobs de minerai (OreFeature.doPlace de Horizon, identique a vanilla) : un bloc de 128 threads par
// blob. Elagage des spheres dans l'ordre du CPU, puis, pour chaque rangee (x, y) de la boite du blob,
// le masque des z dont la cellule est dans une sphere restante. L'ordre des cellules n'est pas garde :
// le CPU ne s'en sert que pour les minerais sans test d'exposition a l'air ni tirage par cellule.
//
// Entree IN (un seul envoi) : BH (nb int4), BS (nb int4), puis SP (doubles). Par blob b :
// BH[b] = {premiere sphere, nombre de spheres, premiere rangee de MK, sx | sy << 8} ;
// BS[b] = {x_start, y_start, z_start, 0}. SP : x, y, z, r de chaque sphere (avant elagage).
// Sortie OUT (un seul retour) : ER (nb int) puis MK, sx * sy masques par blob (rangee x * sy + y,
// bit z). ER[b] != 0 : cellule hors de la boite (le CPU aurait leve WorkLimit), le blob doit passer
// par le CPU.

#define MS_MAX 64   // spheres par blob (size <= 64)
#define MS_AXE 12   // valeurs par axe et par sphere : floor(c + r) - floor(c - r) + 1 <= 2r + 2, r < 4.5
#define MS_RANGEES (26 * 14)  // sx <= 26, sy <= 14 pour size <= 64

extern "C" __global__ void kmin_blobs(const unsigned *IN, unsigned *OUT, int nb)
{
    const int b = blockIdx.x;
    if (b >= nb)
        return;
    const int4 *BH = (const int4 *)IN;
    const int4 *BS = (const int4 *)(IN + 4 * (long)nb);
    const double *SP = (const double *)(IN + 8 * (long)nb);
    int *ER = (int *)OUT;
    unsigned *MK = OUT + nb;
    const int4 h = BH[b];
    const int4 s = BS[b];
    const int n = h.y, sx = h.w & 255, sy = (h.w >> 8) & 255;
    const int t = threadIdx.x;
    __shared__ double C[MS_MAX][3];
    __shared__ double R[MS_MAX];
    __shared__ int LO[MS_MAX][3];
    __shared__ int NB[MS_MAX][3];
    __shared__ double Q[MS_MAX][3][MS_AXE];
    __shared__ unsigned M[MS_RANGEES];
    __shared__ int DEBUT[MS_MAX + 1];
    __shared__ int erreur;
    if (n > MS_MAX || sx * sy > MS_RANGEES || sx > 32) {
        if (t == 0)
            ER[b] = 1;
        return;
    }
    if (t == 0)
        erreur = 0;
    for (int i = t; i < n; i += blockDim.x) {
        const double *v = SP + 4 * (long)(h.x + i);
        C[i][0] = v[0];
        C[i][1] = v[1];
        C[i][2] = v[2];
        R[i] = v[3];
    }
    for (int i = t; i < sx * sy; i += blockDim.x)
        M[i] = 0;
    __syncthreads();

    // Elagage : i1 dans l'ordre. Pour i1 fixe, la boucle i2 (croissante) retire chaque i2 plus petit
    // qui verifie dr * dr > d jusqu'au premier i2 plus grand qui le verifie, lequel retire i1 : apres
    // quoi i1 (rayon -1) ne retire plus rien. Les rayons lus par la boucle i2 sont ceux d'avant elle.
    if (t < 32) {
        for (int i1 = 0; i1 + 1 < n; i1++) {
            const double r1 = R[i1];
            if (r1 <= 0.0)
                continue;
            for (int j0 = i1 + 1; j0 < n; j0 += 32) {
                const int i2 = j0 + t;
                bool retire = false, moi = false;
                if (i2 < n) {
                    const double r2 = R[i2];
                    if (r2 > 0.0) {
                        const double dx = __dsub_rn(C[i1][0], C[i2][0]);
                        const double dy = __dsub_rn(C[i1][1], C[i2][1]);
                        const double dz = __dsub_rn(C[i1][2], C[i2][2]);
                        const double dr = __dsub_rn(r1, r2);
                        const double d = __dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz));
                        if (__dmul_rn(dr, dr) > d) {
                            if (dr > 0.0)
                                retire = true;
                            else
                                moi = true;
                        }
                    }
                }
                const unsigned bm = __ballot_sync(0xffffffffu, moi);
                const int k = bm ? __ffs(bm) - 1 : 32;
                if (retire && t < k)
                    R[i2] = -1.0;
                __syncwarp();
                if (bm) {
                    if (t == 0)
                        R[i1] = -1.0;
                    __syncwarp();
                    break;
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();

    // Carres par axe (SphereSquares::each_cell) : low = max(floor(c - r), start), high = max(floor(c + r), low)
    for (int q = t; q < 3 * n; q += blockDim.x) {
        const int i = q / 3, a = q - 3 * (q / 3);
        const double r = R[i];
        if (r < 0.0) {
            LO[i][a] = 0;
            NB[i][a] = 0;
            continue;
        }
        const double c = C[i][a];
        const int st = a == 0 ? s.x : a == 1 ? s.y : s.z;
        const int taille = a == 1 ? sy : sx;
        int lo = __double2int_rd(__dsub_rn(c, r));
        if (lo < st)
            lo = st;
        int hi = __double2int_rd(__dadd_rn(c, r));
        if (hi < lo)
            hi = lo;
        int nv = hi - lo + 1;
        if (nv > MS_AXE || hi - st >= taille) {
            erreur = 1;
            nv = 0;
        }
        LO[i][a] = lo;
        NB[i][a] = nv;
        for (int v = 0; v < nv; v++) {
            const double d = __ddiv_rn(__dsub_rn(__dadd_rn((double)(lo + v), 0.5), c), r);
            Q[i][a][v] = __dmul_rn(d, d);
        }
    }
    __syncthreads();
    if (t == 0) {
        int acc = 0;
        for (int i = 0; i < n; i++) {
            DEBUT[i] = acc;
            acc += NB[i][0] * NB[i][1];
        }
        DEBUT[n] = acc;
    }
    __syncthreads();

    // Rangees : xd * xd < 1, puis xd * xd + yd * yd < 1, puis (xd * xd + yd * yd) + zd * zd < 1
    const int total = DEBUT[n];
    for (int w = t; w < total; w += blockDim.x) {
        int i = 0;
        while (DEBUT[i + 1] <= w)
            i++;
        const int k = w - DEBUT[i];
        const int ny = NB[i][1];
        const int ix = k / ny, iy = k - ny * (k / ny);
        const double xd2 = Q[i][0][ix];
        if (!(xd2 < 1.0))
            continue;
        const double xy = __dadd_rn(xd2, Q[i][1][iy]);
        if (!(xy < 1.0))
            continue;
        const int z0 = LO[i][2] - s.z;
        unsigned bits = 0;
        for (int iz = 0; iz < NB[i][2]; iz++)
            if (__dadd_rn(xy, Q[i][2][iz]) < 1.0)
                bits |= 1u << (z0 + iz);
        if (bits)
            atomicOr(&M[(LO[i][0] - s.x + ix) * sy + (LO[i][1] - s.y + iy)], bits);
    }
    __syncthreads();
    for (int r = t; r < sx * sy; r += blockDim.x)
        MK[h.z + r] = M[r];
    if (t == 0)
        ER[b] = erreur;
}
