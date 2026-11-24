// Included after base.cu for the scalar Perlin sampler.
// The upload layout mirrors encode() in geodes.rs.
enum {
    SIDE = 33, CELLS = SIDE * SIDE * SIDE,
    POINTS = 12, CRACKS = POINTS + 20 * 4,
    NOISE = CRACKS + 3 * 4, LAYER = 5 + 32, STRIDE = NOISE + 2 * LAYER,
    EMPTY = 0, FILL = 1, CRACK = 2, INNER = 3, MIDDLE = 4, OUTER = 5
};

__device__ __forceinline__ double geode_double(long long bits) {
    return __longlong_as_double(bits);
}

__device__ __forceinline__ double geode_term(
    const long long *point, double offset, double x, double y, double z, double noise
) {
    double dx = __dsub_rn(x, (double)point[0]);
    double dy = __dsub_rn(y, (double)point[1]);
    double dz = __dsub_rn(z, (double)point[2]);
    double distance = __dadd_rn(
        __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz)
    );
    return __dadd_rn(__ddiv_rn(1.0, __dsqrt_rn(__dadd_rn(distance, offset))), noise);
}

__device__ __forceinline__ unsigned char geode_class(
    const long long *job, double x, double y, double z
) {
    float value = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const long long *layer = job + NOISE + i * LAYER;
        double frequency = geode_double(layer[0]);
        float amplitude = __int_as_float((int)layer[1]);
        double offsets[3] = {
            geode_double(layer[2]), geode_double(layer[3]), geode_double(layer[4])
        };
        const unsigned char *perm = reinterpret_cast<const unsigned char *>(layer + 5);
        float sample = perlin_get3(
            perm, offsets, __dmul_rn(x, frequency), __dmul_rn(y, frequency),
            __dmul_rn(z, frequency)
        );
        value = __fadd_rn(value, __fmul_rn(amplitude, sample));
    }
    double noise = __dmul_rn((double)value, geode_double(job[6]));
    double shell = 0.0;
    for (int i = 0; i < job[3]; ++i) {
        const long long *point = job + POINTS + i * 4;
        shell = __dadd_rn(shell, geode_term(point, (double)point[3], x, y, z, noise));
    }
    double outer = geode_double(job[10]);
    if (shell < outer) return EMPTY;
    if (shell >= geode_double(job[7])) return FILL;

    double crack = 0.0;
    for (int i = 0; i < job[4]; ++i) {
        crack = __dadd_rn(
            crack, geode_term(job + CRACKS + i * 4, (double)job[5], x, y, z, noise)
        );
    }
    if (job[4] != 0 && crack >= geode_double(job[11])) return CRACK;
    if (shell >= geode_double(job[8])) return INNER;
    if (shell >= geode_double(job[9])) return MIDDLE;
    if (shell >= outer) return OUTER;
    return EMPTY;
}

extern "C" __global__ void kgeodes(const long long *input, unsigned char *output) {
    int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= CELLS) return;
    int batch = (int)blockIdx.y;
    const long long *job = input + batch * STRIDE;
    // encode() checks that the coordinates are well inside signed-int range.
    int x = (int)job[0] - 16 + cell % SIDE;
    int y = (int)job[1] - 16 + cell / SIDE % SIDE;
    int z = (int)job[2] - 16 + cell / (SIDE * SIDE);
    output[batch * CELLS + cell] = geode_class(job, (double)x, (double)y, (double)z);
}
