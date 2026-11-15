// Ore blobs (Horizon's OreFeature.doPlace, identical to vanilla): one block of 128 threads per
// blob. Sphere pruning in the CPU's order, then, for each row (x, y) of the blob's box, the mask of
// the z whose cell is in a remaining sphere. The order of the cells is not kept: the CPU only uses
// it for ores without an air exposure test or a per-cell draw.
//
// Input IN (a single upload): BH (nb int4), BS (nb int4), then SP (doubles). Per blob b: BH[b] =
// {first sphere, number of spheres, first row of MK, sx | sy << 8}; BS[b] = {x_start, y_start,
// z_start, 0}. SP: x, y, z, r of each sphere (before pruning).
// Output OUT (a single readback): ER (nb int) then MK, sx * sy masks per blob (row x * sy + y, bit
// z). ER[b] != 0: a cell outside the box (the CPU would have raised WorkLimit), the blob must go
// through the CPU.

#define MS_MAX 64   // spheres per blob (size <= 64)
#define MS_AXIS 12   // values per axis and per sphere: floor(c + r) - floor(c - r) + 1 <= 2r + 2, r < 4.5
#define MS_ROWS (26 * 14)  // sx <= 26, sy <= 14 for size <= 64

extern "C" __global__ void kore_blobs(const unsigned *IN, unsigned *OUT, int nb)
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
    __shared__ double Q[MS_MAX][3][MS_AXIS];
    __shared__ unsigned M[MS_ROWS];
    __shared__ int START[MS_MAX + 1];
    __shared__ int error;
    if (n > MS_MAX || sx * sy > MS_ROWS || sx > 32) {
        if (t == 0)
            ER[b] = 1;
        return;
    }
    if (t == 0)
        error = 0;
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

    // Pruning: i1 in order. For a fixed i1, the (increasing) i2 loop removes each smaller i2 that
    // satisfies dr * dr > d up to the first larger i2 that satisfies it, which removes i1: after
    // that, i1 (radius -1) removes nothing more. The radii the i2 loop reads are those from before
    // it.
    if (t < 32) {
        for (int i1 = 0; i1 + 1 < n; i1++) {
            const double r1 = R[i1];
            if (r1 <= 0.0)
                continue;
            for (int j0 = i1 + 1; j0 < n; j0 += 32) {
                const int i2 = j0 + t;
                bool removed = false, mine = false;
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
                                removed = true;
                            else
                                mine = true;
                        }
                    }
                }
                const unsigned bm = __ballot_sync(0xffffffffu, mine);
                const int k = bm ? __ffs(bm) - 1 : 32;
                if (removed && t < k)
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

    // Squares per axis (SphereSquares::each_cell): low = max(floor(c - r), start), high =
    // max(floor(c + r), low)
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
        const int size = a == 1 ? sy : sx;
        int lo = __double2int_rd(__dsub_rn(c, r));
        if (lo < st)
            lo = st;
        int hi = __double2int_rd(__dadd_rn(c, r));
        if (hi < lo)
            hi = lo;
        int nv = hi - lo + 1;
        if (nv > MS_AXIS || hi - st >= size) {
            error = 1;
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
            START[i] = acc;
            acc += NB[i][0] * NB[i][1];
        }
        START[n] = acc;
    }
    __syncthreads();

    // Rows: xd * xd < 1, then xd * xd + yd * yd < 1, then (xd * xd + yd * yd) + zd * zd < 1
    const int total = START[n];
    for (int w = t; w < total; w += blockDim.x) {
        int i = 0;
        while (START[i + 1] <= w)
            i++;
        const int k = w - START[i];
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
        ER[b] = error;
}
