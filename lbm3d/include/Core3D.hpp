#ifndef LBM3D_CORE3D_HPP
#define LBM3D_CORE3D_HPP

// D3Q19 / TRT cell update shared by the CPU (OpenMP) and GPU (CUDA) back ends.
//
// Everything in this header is plain C++ with no dependencies, so that the very same
// function is compiled for the host and for the device: the physics is written (and
// can be validated) exactly once.

#include <cstdint>

#if defined(__CUDACC__)
#define LBM_HD __host__ __device__
#else
#define LBM_HD
#endif

namespace lbm3d
{

#ifdef LBM_DOUBLE
typedef double real;
#else
typedef float real;
#endif

enum
{
    Q = 19
};

// cell classification, computed once on the host
enum CellType : std::uint8_t
{
    CELL_BULK = 0, // fluid, all 18 neighbours are fluid: branch-free fast path
    CELL_NEAR = 1, // fluid, at least one neighbour is a wall / obstacle / open boundary
    CELL_SOLID = 2 // obstacle
};

struct Params
{
    int nx, ny, nz;
    long long stride; // distance between two populations in the arrays, >= nx * ny * nz
    int problem;   // 1 = lid driven cavity, 2 = channel flow (inlet x-, outlet x+) with obstacles
    int periodicZ; // 1 = periodic along z (quasi 2D), 0 = solid walls along z
    real omP;      // TRT relaxation rate of the symmetric part
    real omM;      // TRT relaxation rate of the anti-symmetric part
    real rhoOut;   // density imposed at the outlet (problem 2)
};

// Memory layout: structure of arrays, f[i * stride + cell], cell = x + nx * (y + ny * z).
// stride is the number of cells plus some padding. Without it, on lattices whose cell count
// is a large power of two (e.g. 128^3) the 19 populations of a cell sit at the same offset of
// 19 different pages; they then compete for the same cache and TLB sets and the CPU code
// becomes about 3x slower.
//
// The arrays do not hold the populations themselves but their deviation from the rest
// equilibrium, f_i - w_i. The algorithm is unchanged, but in single precision the round-off
// is then relative to the (small) deviation instead of to w_i ~ 0.3, which removes the slow
// drift of the total mass that plain float storage shows over long runs.
//
// Velocity set. Opposite directions are adjacent: opposite(i) = i + 1 for odd i, i - 1 for even i.
//   1..6   axis directions      (weight 1/18)
//   7..18  face diagonals       (weight 1/36)

/// One fused "pull" step for one cell: gather the incoming populations from the neighbours
/// (streaming + boundary conditions), compute the moments, collide (TRT) and store the
/// post-collision populations into fn.
///
/// f     : post-collision populations (minus w_i) of the previous step (read only)
/// fn    : post-collision populations (minus w_i) of this step (written only at this cell)
/// force : momentum exchanged with obstacles through this cell's links is ADDED to force[0..2]
LBM_HD inline void updateCell(const Params &p, const real uNow, const int x, const int y, const int z, const real *f,
                              real *fn, const std::uint8_t *type, real *rho, real *ux, real *uy, real *uz,
                              const bool writeMacro, real *force)
{
    const int cx[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
    const int cy[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
    const int cz[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
    const real w0 = real(1.0 / 3.0);
    const real w1 = real(1.0 / 18.0);
    const real w2 = real(1.0 / 36.0);

    const int nx = p.nx, ny = p.ny, nz = p.nz;
    const long long N = p.stride;
    const long long idx = x + (long long)nx * (y + (long long)ny * z);
    const std::uint8_t t = type[idx];

    if (t == CELL_SOLID)
    {
        if (writeMacro)
        {
            rho[idx] = real(1.0);
            ux[idx] = real(0.0);
            uy[idx] = real(0.0);
            uz[idx] = real(0.0);
        }
        return;
    }

    // ---------------------------------------------------------------- streaming (pull)
    real fin[Q];
    fin[0] = f[idx];

    if (t == CELL_BULK)
    {
        for (int i = 1; i < Q; ++i)
        {
            const long long off = cx[i] + (long long)nx * (cy[i] + (long long)ny * cz[i]);
            fin[i] = f[i * N + idx - off];
        }
    }
    else
    {
        bool haveLocalU = false;
        real lux = 0, luy = 0, luz = 0;

        for (int i = 1; i < Q; ++i)
        {
            const int o = (i & 1) ? i + 1 : i - 1; // opposite direction
            const real wi = (i < 7) ? w1 : w2;
            const int sx = x - cx[i];
            const int sy = y - cy[i];
            int sz = z - cz[i];
            if (p.periodicZ)
            {
                if (sz < 0)
                    sz += nz;
                else if (sz >= nz)
                    sz -= nz;
            }
            const bool outX = sx < 0 || sx >= nx;
            const bool outYZ = sy < 0 || sy >= ny || sz < 0 || sz >= nz;
            const real fo = f[o * N + idx]; // what this cell sent towards the source cell

            if (!outX && !outYZ)
            {
                const long long sidx = sx + (long long)nx * (sy + (long long)ny * sz);
                if (type[sidx] != CELL_SOLID)
                {
                    fin[i] = f[i * N + sidx];
                }
                else
                {
                    // obstacle: halfway bounce back + momentum exchange (force on the obstacle)
                    fin[i] = fo;
                    // 2 * (fo + w_i) would be the full exchange. The w_i terms are the push of
                    // the uniform reference pressure, which cancels over a closed surface;
                    // leaving them out keeps the sum well conditioned in single precision.
                    const real m = real(2.0) * fo;
                    force[0] -= m * cx[i];
                    force[1] -= m * cy[i];
                    force[2] -= m * cz[i];
                }
            }
            else if (outYZ || p.problem == 1)
            {
                // solid wall: halfway bounce back
                fin[i] = fo;
                // moving lid of the cavity: the wall beyond y = 0, sliding along +x (rho_wall = 1)
                if (p.problem == 1 && sy < 0 && !outX && sz >= 0 && sz < nz)
                    fin[i] += real(6.0) * wi * cx[i] * uNow;
            }
            else if (sx < 0)
            {
                // velocity inlet: bounce back from a wall moving with the inlet velocity,
                // evaluated where the link crosses the inlet plane
                const real Y = (y - real(0.5) * cy[i] + real(0.5)) / ny;
                real prof = real(4.0) * Y * (real(1.0) - Y);
                if (!p.periodicZ)
                {
                    const real Z = (z - real(0.5) * cz[i] + real(0.5)) / nz;
                    prof *= real(4.0) * Z * (real(1.0) - Z);
                }
                fin[i] = fo + real(6.0) * wi * cx[i] * uNow * prof;
            }
            else
            {
                // pressure outlet: anti bounce back with the velocity of this cell
                if (!haveLocalU)
                {
                    real r = real(1.0);
                    for (int k = 0; k < Q; ++k)
                    {
                        const real fk = f[k * N + idx];
                        r += fk;
                        lux += fk * cx[k];
                        luy += fk * cy[k];
                        luz += fk * cz[k];
                    }
                    lux /= r;
                    luy /= r;
                    luz /= r;
                    haveLocalU = true;
                }
                const real cu = cx[i] * lux + cy[i] * luy + cz[i] * luz;
                const real uu = lux * lux + luy * luy + luz * luz;
                fin[i] = -fo + real(2.0) * wi *
                                   ((p.rhoOut - real(1.0)) + p.rhoOut * (real(4.5) * cu * cu - real(1.5) * uu));
            }
        }
    }

    // ---------------------------------------------------------------- moments
    real dr = fin[0]; // density deviation, rho - 1
    real jx = 0, jy = 0, jz = 0;
    for (int i = 1; i < Q; ++i)
    {
        dr += fin[i];
        jx += fin[i] * cx[i];
        jy += fin[i] * cy[i];
        jz += fin[i] * cz[i];
    }
    const real r = real(1.0) + dr;
    const real inv = real(1.0) / r;
    const real vx = jx * inv, vy = jy * inv, vz = jz * inv;

    if (writeMacro)
    {
        rho[idx] = r;
        ux[idx] = vx;
        uy[idx] = vy;
        uz[idx] = vz;
    }

    // ---------------------------------------------------------------- TRT collision
    // equilibrium deviation: feq_i - w_i = w_i * (dr + rho * (3 cu + 4.5 cu^2 - 1.5 u^2))
    const real usq = real(-1.5) * (vx * vx + vy * vy + vz * vz);
    fn[idx] = fin[0] - p.omP * (fin[0] - w0 * (dr + r * usq));
    for (int i = 1; i < Q; i += 2)
    {
        const real wi = (i < 7) ? w1 : w2;
        const real cu = cx[i] * vx + cy[i] * vy + cz[i] * vz;
        const real feqP = wi * (dr + r * (usq + real(4.5) * cu * cu)); // symmetric part of the equilibrium
        const real feqM = wi * r * real(3.0) * cu;                     // anti-symmetric part
        const real fP = real(0.5) * (fin[i] + fin[i + 1]);
        const real fM = real(0.5) * (fin[i] - fin[i + 1]);
        const real dP = p.omP * (fP - feqP);
        const real dM = p.omM * (fM - feqM);
        fn[i * N + idx] = fin[i] - dP - dM;
        fn[(i + 1) * N + idx] = fin[i + 1] - dP + dM;
    }
}

} // namespace lbm3d

#endif // LBM3D_CORE3D_HPP
