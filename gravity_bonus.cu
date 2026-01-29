#include <cstdio>
#include <cmath>

#define G    6.67430e-11
#define SOFT 1e9
#define DAY  86400.0

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

// ------------------------------
// SHARED-MEMORY TILED KERNEL
// ------------------------------
__global__
void computeAccelTiled(Body *b, double *ax, double *ay, double *az, int n)
{
    extern __shared__ Body tile[];   // tile of bodies loaded into shared memory

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double xi = b[i].x, yi = b[i].y, zi = b[i].z;
    double axi = 0, ayi = 0, azi = 0;

    // Loop over the bodies in chunks of blockDim.x
    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x)
    {
        int j = tileStart + threadIdx.x;

        // Load tile into shared memory
        if (j < n)
            tile[threadIdx.x] = b[j];

        __syncthreads();

        // Compute forces using shared memory tile
        int tileSize = min(blockDim.x, n - tileStart);

        for (int k = 0; k < tileSize; k++)
        {
            if (i == tileStart + k) continue;

            double dx = tile[k].x - xi;
            double dy = tile[k].y - yi;
            double dz = tile[k].z - zi;

            double dist2 = dx*dx + dy*dy + dz*dz + SOFT*SOFT;
            double invDist = rsqrt(dist2);
            double invDist3 = invDist * invDist * invDist;

            double f = G * tile[k].mass * invDist3;

            axi += f * dx;
            ayi += f * dy;
            azi += f * dz;
        }

        __syncthreads();
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

// ------------------------------------
// Update kernel (unchanged)
// ------------------------------------
__global__
void updateBodies(Body *b, double *ax, double *ay, double *az, int n, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    b[i].vx += ax[i] * dt;
    b[i].vy += ay[i] * dt;
    b[i].vz += az[i] * dt;

    b[i].x += b[i].vx * dt;
    b[i].y += b[i].vy * dt;
    b[i].z += b[i].vz * dt;
}

// ------------------------------------
// Main Simulation: Sun + Mercury + Earth + Jupiter
// ------------------------------------
int main()
{
    const int n = 4;
    Body h[n];

    // Sun
    h[0] = {0,0,0, 0,0,0, 1.989e30};

    // Mercury
    h[1] = {5.7909e10,0,0, 0,4.788e4,0, 3.3011e23};

    // Earth
    h[2] = {1.4959787e11,0,0, 0,2.979e4,0, 5.972e24};

    // Jupiter
    h[3] = {7.78547e11,0,0, 0,1.306e4,0, 1.898e27};

    Body *dB;
    double *ax, *ay, *az;

    cudaMalloc(&dB, n*sizeof(Body));
    cudaMalloc(&ax, n*sizeof(double));
    cudaMalloc(&ay, n*sizeof(double));
    cudaMalloc(&az, n*sizeof(double));

    cudaMemcpy(dB, h, n*sizeof(Body), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((n + block.x - 1) / block.x);

    size_t sharedBytes = block.x * sizeof(Body);

    double dt = DAY;
    int total_days = 400;

    for (int t = 0; t <= total_days; t++)
    {
        cudaMemcpy(h, dB, n*sizeof(Body), cudaMemcpyDeviceToHost);

        printf("t=%d days\n", t);
        printf("Sun      %.3e, %.3e\n", h[0].x, h[0].y);
        printf("Mercury  %.3e, %.3e\n", h[1].x, h[1].y);
        printf("Earth    %.3e, %.3e\n", h[2].x, h[2].y);
        printf("Jupiter  %.3e, %.3e\n\n", h[3].x, h[3].y);

        computeAccelTiled<<<grid, block, sharedBytes>>>(dB, ax, ay, az, n);
        updateBodies<<<grid, block>>>(dB, ax, ay, az, n, dt);

        cudaDeviceSynchronize();
    }

    return 0;
}
