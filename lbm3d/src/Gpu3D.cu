#include "Sim3D.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

// The kernels below only decode the thread index and call updateCell() from Core3D.hpp, the
// same function the CPU back end runs.
//
// LBM3D_CUDA_MOCK builds this file as plain C++ on machines without a CUDA toolkit: kernels are
// then executed thread by thread on the host (see tests/cuda_mock.hpp). It exists to test the
// logic of this file (indexing, buffer swap, transfers), not for performance.
#ifdef LBM3D_CUDA_MOCK
#include "../tests/cuda_mock.hpp"
#else
#include <cuda_runtime.h>
#define LBM_LAUNCH(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)
#endif

namespace lbm3d
{

__global__ void initKernel(const int nx, const int ny, const int nz, const long long stride, real *f, real *fn)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;

    const long long idx = x + (long long)nx * (y + (long long)ny * z);
    // equilibrium at rest, stored as the deviation f_i - w_i = 0
    for (int i = 0; i < Q; ++i)
    {
        f[i * stride + idx] = real(0.0);
        fn[i * stride + idx] = real(0.0);
    }
}

__global__ void stepKernel(const Params p, const real uNow, const real *f, real *fn, const std::uint8_t *type,
                           real *rho, real *ux, real *uy, real *uz, const int frame, real *force)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= p.nx || y >= p.ny || z >= p.nz)
        return;

    real cellForce[3] = {0, 0, 0};
    updateCell(p, uNow, x, y, z, f, fn, type, rho, ux, uy, uz, frame != 0, cellForce);

    // only the few cells that touch an obstacle contribute, and only frames need the total
    if (frame != 0 && (cellForce[0] != 0 || cellForce[1] != 0 || cellForce[2] != 0))
    {
        atomicAdd(&force[0], cellForce[0]);
        atomicAdd(&force[1], cellForce[1]);
        atomicAdd(&force[2], cellForce[2]);
    }
}

namespace
{
void check(const cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
}

template <typename T> T *deviceAlloc(const long long count, const char *what)
{
    void *ptr = nullptr;
    check(cudaMalloc(&ptr, static_cast<size_t>(count) * sizeof(T)), what);
    return static_cast<T *>(ptr);
}
} // namespace

RunStats runGpu(const Sim &sim)
{
    const Params p = sim.p;
    const long long N = sim.N;

    // structure of arrays on the device; two lattices, swapped by pointer after every step
    real *f = deviceAlloc<real>(Q * p.stride, "cudaMalloc(f)");
    real *fn = deviceAlloc<real>(Q * p.stride, "cudaMalloc(fn)");
    real *dRho = deviceAlloc<real>(N, "cudaMalloc(rho)");
    real *dUx = deviceAlloc<real>(N, "cudaMalloc(ux)");
    real *dUy = deviceAlloc<real>(N, "cudaMalloc(uy)");
    real *dUz = deviceAlloc<real>(N, "cudaMalloc(uz)");
    real *dForce = deviceAlloc<real>(3, "cudaMalloc(force)");
    std::uint8_t *dType = deviceAlloc<std::uint8_t>(N, "cudaMalloc(type)");

    check(cudaMemcpy(dType, sim.type.data(), static_cast<size_t>(N), cudaMemcpyHostToDevice), "cudaMemcpy(type)");

    // x is the contiguous direction in memory: wide blocks along x keep the accesses coalesced
    const dim3 block(32, 4, 4);
    const dim3 grid((p.nx + block.x - 1) / block.x, (p.ny + block.y - 1) / block.y, (p.nz + block.z - 1) / block.z);

    LBM_LAUNCH(initKernel, grid, block, p.nx, p.ny, p.nz, p.stride, f, fn);
    check(cudaGetLastError(), "initKernel");
    check(cudaDeviceSynchronize(), "initKernel");

    // host side copies of the macroscopic fields, only filled at frame steps
    std::vector<real> hRho, hUx, hUy, hUz;
    if (sim.frameEvery > 0)
    {
        hRho.resize(static_cast<size_t>(N));
        hUx.resize(static_cast<size_t>(N));
        hUy.resize(static_cast<size_t>(N));
        hUz.resize(static_cast<size_t>(N));
    }

    FrameWriter writer(sim);
    double seconds = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step <= sim.maxIt; ++step)
    {
        const real uNow = sim.uAt(step);
        const bool frame = sim.isFrame(step);

        if (frame)
            check(cudaMemset(dForce, 0, 3 * sizeof(real)), "cudaMemset(force)");

        LBM_LAUNCH(stepKernel, grid, block, p, uNow, f, fn, dType, dRho, dUx, dUy, dUz, frame ? 1 : 0, dForce);

        real *tmp = f;
        f = fn;
        fn = tmp;

        if (frame)
        {
            // kernel launches are asynchronous: wait before stopping the clock
            check(cudaDeviceSynchronize(), "stepKernel");
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            real hForce[3];
            const size_t bytes = static_cast<size_t>(N) * sizeof(real);
            check(cudaMemcpy(hRho.data(), dRho, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(rho)");
            check(cudaMemcpy(hUx.data(), dUx, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(ux)");
            check(cudaMemcpy(hUy.data(), dUy, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(uy)");
            check(cudaMemcpy(hUz.data(), dUz, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(uz)");
            check(cudaMemcpy(hForce, dForce, 3 * sizeof(real), cudaMemcpyDeviceToHost), "cudaMemcpy(force)");

            const double force[3] = {static_cast<double>(hForce[0]), static_cast<double>(hForce[1]),
                                     static_cast<double>(hForce[2])};
            writer.write(step, hRho.data(), hUx.data(), hUy.data(), hUz.data(), force);

            t0 = std::chrono::steady_clock::now();
        }
    }

    check(cudaDeviceSynchronize(), "stepKernel");
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaFree(f);
    cudaFree(fn);
    cudaFree(dRho);
    cudaFree(dUx);
    cudaFree(dUy);
    cudaFree(dUz);
    cudaFree(dForce);
    cudaFree(dType);

    RunStats stats;
    stats.seconds = seconds;
    stats.mlups = static_cast<double>(sim.fluidCells) * (sim.maxIt + 1) / seconds / 1e6;
    return stats;
}

} // namespace lbm3d
