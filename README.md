This project implements a gravitational N-body simulator using CUDA:

    1. Basic Sun–Earth simulation

    2. Extended multi-planet simulation (Sun, Mercury, Earth, Jupiter)

CSV logging + Python plotting

    3. version using shared-memory tiling for faster performance
The bonus file (gravity_bonus.cu) improves the kernel by using shared-memory tiling.
Instead of each thread repeatedly reading planet data from global memory, each block loads a chunk of bodies once into shared memory.
All threads reuse this tile, which reduces global memory traffic and speeds up force computation.

Profiling comparison between the simple and optimized versions

Basic Sun–Earth Version
The first version simulates only the Sun and Earth. It calculates gravitational force, updates velocity, and integrates position each day.

Multi-Planet Simulation
The next version (gravity_cuda_planets.cu) adds Mercury and Jupiter. Each planet interacts with all others, and the output shows the expected behavior:


→ The optimized version is ~22% faster, even with only 4 bodies. The improvement becomes much larger as the number of bodies increases.
