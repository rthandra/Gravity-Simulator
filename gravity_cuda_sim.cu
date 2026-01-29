#include <cstdio>
#include <cmath>

#define G 6.67430e-11
#define SOFT 1e9
#define DAY 86400.0

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

__global__
void computeAccel(Body *b, double *ax, double *ay, double *az, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double xi = b[i].x, yi = b[i].y, zi = b[i].z;

    double axi = 0, ayi = 0, azi = 0;

    for (int j = 0; j < n; j++)
    {
        if (i == j) continue;

        double dx = b[j].x - xi;
        double dy = b[j].y - yi;
        double dz = b[j].z - zi;

        double dist2 = dx*dx + dy*dy + dz*dz + SOFT*SOFT;
        double invDist = rsqrt(dist2);
        double invDist3 = invDist * invDist * invDist;

        double f = G * b[j].mass * invDist3;

        axi += f * dx;
        ayi += f * dy;
        azi += f * dz;
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

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

int main()
{
    Body h[2];

    // Sun
    h[0].mass = 1.989e30;
    h[0].x = h[0].y = h[0].z = 0;
    h[0].vx = h[0].vy = h[0].vz = 0;

    // Earth
    h[1].mass = 5.972e24;
    h[1].x = 1.496e11;
    h[1].y = h[1].z = 0;
    h[1].vx = 0;
    h[1].vy = 2.979e4;
    h[1].vz = 0;

    const int n = 2;
    Body *dB;
    double *ax, *ay, *az;

    cudaMalloc(&dB, n * sizeof(Body));
    cudaMalloc(&ax, n * sizeof(double));
    cudaMalloc(&ay, n * sizeof(double));
    cudaMalloc(&az, n * sizeof(double));

    cudaMemcpy(dB, h, n * sizeof(Body), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    double dt = DAY;
    int total = 2000000;

    for (int t = 0; t <= total; t += 100000)
    {
        cudaMemcpy(h, dB, n * sizeof(Body), cudaMemcpyDeviceToHost);
        printf("t=%d\n", t);
        printf("earth = %.3e , %.3e\n", h[1].x, h[1].y);

        computeAccel<<<grid, block>>>(dB, ax, ay, az, n);
        updateBodies<<<grid, block>>>(dB, ax, ay, az, n, dt);
        cudaDeviceSynchronize();
    }

    return 0;
}
