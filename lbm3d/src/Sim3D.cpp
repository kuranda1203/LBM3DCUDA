#include "Sim3D.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace lbm3d
{

real Sim::uAt(const int step) const
{
    // same smooth start-up as the 2D code
    const double t = static_cast<double>(step);
    return static_cast<real>(uMax * (1.0 - std::exp(-t * t / (2.0 * sigma * sigma))));
}

Sim loadSim(const Options &opt)
{
    std::ifstream in(opt.inputFile);
    if (!in.is_open())
        throw std::runtime_error("Could not open input file " + opt.inputFile);

    Sim sim;
    sim.opt = opt;
    Params &p = sim.p;

    // Same file format as the 2D code, with three sizes on the second line:
    //   problem type / nx ny nz / reynolds steps velocity / one "x y z" line per obstacle cell
    std::string line;
    std::getline(in, line);
    p.problem = std::stoi(line);
    if (p.problem != 1 && p.problem != 2)
        throw std::runtime_error("3D solver supports problem type 1 (cavity) and 2 (channel with obstacles)");

    std::getline(in, line);
    {
        std::istringstream ss(line);
        std::vector<int> shape;
        int v;
        while (ss >> v)
            shape.push_back(v);
        if (shape.size() != 3)
            throw std::runtime_error("3D solver needs 'nx ny nz' on the second line");
        p.nx = shape[0];
        p.ny = shape[1];
        p.nz = shape[2];
        if (p.nx < 2 || p.ny < 2 || p.nz < 1)
            throw std::runtime_error("Invalid lattice size");
    }

    double uIn;
    if (!(in >> sim.reynolds >> sim.maxIt >> uIn))
        throw std::runtime_error("Could not read 'reynolds steps velocity'");
    sim.uMax = static_cast<real>(uIn);

    p.periodicZ = opt.periodicZ ? 1 : 0;
    p.rhoOut = real(1.0);
    sim.N = static_cast<long long>(p.nx) * p.ny * p.nz;

    // Pad the distance between two populations (see Core3D.hpp). Seen from population 0 of a
    // cell, population i then sits i * 320 bytes further into its 4 KiB page (L1 cache sets)
    // and, on large lattices, i * 37 pages further (TLB and L2 sets). Measured on 128^3,
    // single thread: 9 MLUPS without padding, 30 with it.
    {
        const long long bytes = static_cast<long long>(sizeof(real));
        const bool large = sim.N * bytes >= (1LL << 20);
        const long long modulus = (large ? (1LL << 20) : 4096) / bytes;
        const long long want = (large ? 37 * 4096 + 320 : 320) / bytes;
        const long long base = (sim.N + 15) / 16 * 16;
        p.stride = base + ((want - base % modulus) + modulus) % modulus;
    }

    // simulation parameters, same definitions as the 2D code
    sim.nu = uIn * p.ny / sim.reynolds;
    if (p.problem == 2)
        sim.nu *= 2.0 / 3.0;
    sim.tau = 0.5 + 3.0 * sim.nu;
    sim.sigma = 10.0 * p.ny;
    const double lambdaTrt = 1.0 / 4.0;
    const double tauMinus = lambdaTrt / (sim.tau - 0.5) + 0.5;
    p.omP = static_cast<real>(1.0 / sim.tau);
    p.omM = static_cast<real>(1.0 / tauMinus);

    // obstacles
    sim.type.assign(static_cast<size_t>(sim.N), CELL_BULK);
    int ox, oy, oz;
    long long nObstacle = 0;
    while (in >> ox >> oy >> oz)
    {
        if (p.problem != 2)
            continue; // like the 2D code, obstacles only exist in problem 2
        if (ox < 0 || ox >= p.nx || oy < 0 || oy >= p.ny || oz < 0 || oz >= p.nz)
            throw std::runtime_error("Obstacle cell outside of the lattice");
        std::uint8_t &t = sim.type[ox + static_cast<long long>(p.nx) * (oy + static_cast<long long>(p.ny) * oz)];
        if (t != CELL_SOLID)
            ++nObstacle;
        t = CELL_SOLID;
    }
    sim.fluidCells = sim.N - nObstacle;

    // classify fluid cells: bulk (fast path) or near a boundary
    const int nx = p.nx, ny = p.ny, nz = p.nz;
    std::vector<char> shadow(static_cast<size_t>(ny) * nz, 0);
#pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
            {
                const long long idx = x + static_cast<long long>(nx) * (y + static_cast<long long>(ny) * z);
                if (sim.type[idx] == CELL_SOLID)
                {
                    shadow[y + static_cast<size_t>(ny) * z] = 1; // only this thread touches plane z
                    continue;
                }
                bool near = false;
                for (int dz = -1; dz <= 1 && !near; ++dz)
                    for (int dy = -1; dy <= 1 && !near; ++dy)
                        for (int dx = -1; dx <= 1 && !near; ++dx)
                        {
                            if (std::abs(dx) + std::abs(dy) + std::abs(dz) == 0 ||
                                std::abs(dx) + std::abs(dy) + std::abs(dz) == 3)
                                continue; // D3Q19 has no cube diagonals
                            const int sx = x + dx, sy = y + dy, sz = z + dz;
                            // a periodic neighbour also makes the cell "near": the bulk fast path
                            // addresses its neighbours with fixed offsets and cannot wrap around
                            if (sx < 0 || sx >= nx || sy < 0 || sy >= ny || sz < 0 || sz >= nz)
                                near = true;
                            else if (sim.type[sx + static_cast<long long>(nx) *
                                                       (sy + static_cast<long long>(ny) * sz)] == CELL_SOLID)
                                near = true;
                        }
                // CELL_NEAR is only ever written to / read from non solid cells, CELL_SOLID never changes
                if (near)
                    sim.type[idx] = CELL_NEAR;
            }

    // reference quantities for the force coefficients
    sim.frontalArea = static_cast<double>(std::count(shadow.begin(), shadow.end(), 1));
    sim.uMean = p.periodicZ ? uIn * 2.0 / 3.0 : uIn * 4.0 / 9.0; // mean of the inlet profile

    if (opt.frames > 0)
        sim.frameEvery = std::max(1, sim.maxIt / opt.frames);

    return sim;
}

// ------------------------------------------------------------------------------------------------

namespace
{
void writeBigEndian(std::ofstream &out, const std::vector<float> &buf)
{
    std::vector<char> bytes(buf.size() * 4);
    for (size_t i = 0; i < buf.size(); ++i)
    {
        std::uint32_t v;
        std::memcpy(&v, &buf[i], 4);
        bytes[4 * i + 0] = static_cast<char>(v >> 24);
        bytes[4 * i + 1] = static_cast<char>(v >> 16);
        bytes[4 * i + 2] = static_cast<char>(v >> 8);
        bytes[4 * i + 3] = static_cast<char>(v);
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}
} // namespace

FrameWriter::FrameWriter(const Sim &sim) : sim(sim)
{
    if (sim.frameEvery > 0 && sim.opt.writeVtk)
    {
        forces.open(sim.opt.outDir + "/forces.txt");
        if (!forces.is_open())
            throw std::runtime_error("Could not write into '" + sim.opt.outDir + "' (does the folder exist?)");
        forces << "# step Fx Fy Fz Cd Cl(y) Cs(z)   [coefficients: 2 F / (uMean^2 * frontalArea), uMean = "
               << sim.uMean << ", frontalArea = " << sim.frontalArea << "]\n";
    }
}

void FrameWriter::write(const int step, const real *rho, const real *ux, const real *uy, const real *uz,
                        const double force[3])
{
    const Params &p = sim.p;
    const long long N = sim.N;

    // diagnostics
    double mass = 0, maxU = 0;
    bool finite = true;
#pragma omp parallel for schedule(static) reduction(+ : mass) reduction(max : maxU) reduction(&& : finite)
    for (long long i = 0; i < N; ++i)
    {
        if (sim.type[i] == CELL_SOLID)
            continue;
        mass += rho[i];
        const double u2 = static_cast<double>(ux[i]) * ux[i] + static_cast<double>(uy[i]) * uy[i] +
                          static_cast<double>(uz[i]) * uz[i];
        finite = finite && std::isfinite(u2);
        maxU = std::max(maxU, u2);
    }
    maxU = std::sqrt(maxU);

    std::printf("step %8d  mass/cell %.8f  max|u| %.5f", step, mass / static_cast<double>(sim.fluidCells), maxU);
    double c[3] = {0, 0, 0};
    if (p.problem == 2 && sim.frontalArea > 0)
    {
        for (int d = 0; d < 3; ++d)
            c[d] = 2.0 * force[d] / (sim.uMean * sim.uMean * sim.frontalArea);
        std::printf("  Cd %.5f  Cl %.5f  Cs %.5f", c[0], c[1], c[2]);
    }
    std::printf("\n");
    std::fflush(stdout);
    if (!finite)
        throw std::runtime_error("Simulation diverged (non finite velocity). Lower the velocity or the Reynolds "
                                 "number, or use a finer lattice.");

    if (!sim.opt.writeVtk)
        return;

    forces << step << ' ' << force[0] << ' ' << force[1] << ' ' << force[2] << ' ' << c[0] << ' ' << c[1] << ' '
           << c[2] << '\n';
    forces.flush();

    // legacy VTK, binary: opens directly in ParaView / VisIt, and is read by scripts/plotting3D.py
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%04d.vtk", frame++);
    std::ofstream out(sim.opt.outDir + name, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Could not create output file in " + sim.opt.outDir);
    out << "# vtk DataFile Version 3.0\n"
        << "lbm3d step " << step << "\n"
        << "BINARY\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << p.nx << ' ' << p.ny << ' ' << p.nz << "\n"
        << "ORIGIN 0 0 0\n"
        << "SPACING 1 1 1\n"
        << "POINT_DATA " << N << "\n"
        << "VECTORS velocity float\n";
    std::vector<float> buf;
    const long long chunk = 1 << 20;
    for (long long start = 0; start < N; start += chunk)
    {
        const long long end = std::min(N, start + chunk);
        buf.resize(static_cast<size_t>(3 * (end - start)));
        for (long long i = start; i < end; ++i)
        {
            buf[3 * (i - start) + 0] = static_cast<float>(ux[i]);
            buf[3 * (i - start) + 1] = static_cast<float>(uy[i]);
            buf[3 * (i - start) + 2] = static_cast<float>(uz[i]);
        }
        writeBigEndian(out, buf);
    }
    out << "\nSCALARS density float 1\nLOOKUP_TABLE default\n";
    for (long long start = 0; start < N; start += chunk)
    {
        const long long end = std::min(N, start + chunk);
        buf.resize(static_cast<size_t>(end - start));
        for (long long i = start; i < end; ++i)
            buf[i - start] = static_cast<float>(rho[i]);
        writeBigEndian(out, buf);
    }
    out << "\nSCALARS obstacle unsigned_char 1\nLOOKUP_TABLE default\n";
    std::vector<char> ob(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i)
        ob[i] = sim.type[i] == CELL_SOLID ? 1 : 0;
    out.write(ob.data(), static_cast<std::streamsize>(ob.size()));
    out << '\n';
}

} // namespace lbm3d
