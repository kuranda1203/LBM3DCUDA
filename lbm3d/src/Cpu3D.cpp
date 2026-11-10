#include "Sim3D.hpp"

#include <chrono>
#include <memory>
#include <utility>

namespace lbm3d
{

// Serial and OpenMP back end (the serial version is this same code run with one thread).
//
// Differences from the 2D CPU code that matter for performance:
//  - structure of arrays instead of one heap allocated object per cell,
//  - streaming and collision fused in a single sweep over two lattices (A/B swap), so each
//    population is read once and written once per step,
//  - the parallel loop is over (z, y) pencils and the x loop stays contiguous in memory,
//  - arrays are first touched with the same static schedule that the time loop uses, so on
//    NUMA machines every thread works on pages that live on its own socket.
RunStats runCpu(const Sim &sim)
{
    const Params p = sim.p;
    const int nx = p.nx, ny = p.ny, nz = p.nz;
    const long long N = sim.N;
    const long long S = p.stride;
    const std::uint8_t *type = sim.type.data();

    std::unique_ptr<real[]> bufA(new real[Q * S]);
    std::unique_ptr<real[]> bufB(new real[Q * S]);
    std::unique_ptr<real[]> rho(new real[N]), ux(new real[N]), uy(new real[N]), uz(new real[N]);
    real *f = bufA.get();
    real *fn = bufB.get();

    // equilibrium at rest (rho = 1, u = 0): f_i = w_i, stored as the deviation f_i - w_i = 0
#pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
            {
                const long long idx = x + static_cast<long long>(nx) * (y + static_cast<long long>(ny) * z);
                for (int i = 0; i < Q; ++i)
                {
                    f[i * S + idx] = real(0.0);
                    fn[i * S + idx] = real(0.0);
                }
                rho[idx] = real(1.0);
                ux[idx] = uy[idx] = uz[idx] = real(0.0);
            }

    FrameWriter writer(sim);
    double seconds = 0;

    for (int step = 0; step <= sim.maxIt; ++step)
    {
        const real uNow = sim.uAt(step);
        const bool frame = sim.isFrame(step);
        double fx = 0, fy = 0, fz = 0;

        const auto t0 = std::chrono::steady_clock::now();

#pragma omp parallel for collapse(2) schedule(static) reduction(+ : fx, fy, fz)
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
            {
                real rowForce[3] = {0, 0, 0};
                for (int x = 0; x < nx; ++x)
                    updateCell(p, uNow, x, y, z, f, fn, type, rho.get(), ux.get(), uy.get(), uz.get(), frame,
                               rowForce);
                fx += rowForce[0];
                fy += rowForce[1];
                fz += rowForce[2];
            }

        std::swap(f, fn);
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        if (frame)
        {
            const double force[3] = {fx, fy, fz};
            writer.write(step, rho.get(), ux.get(), uy.get(), uz.get(), force);
        }
    }

    RunStats stats;
    stats.seconds = seconds;
    stats.mlups = static_cast<double>(sim.fluidCells) * (sim.maxIt + 1) / seconds / 1e6;
    return stats;
}

} // namespace lbm3d
