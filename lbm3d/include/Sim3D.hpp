#ifndef LBM3D_SIM3D_HPP
#define LBM3D_SIM3D_HPP

#include "Core3D.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace lbm3d
{

struct Options
{
    std::string inputFile;
    std::string outDir = "outputs3d";
    int frames = 0;        // number of output frames (0 = benchmark, no output at all)
    bool gpu = false;
    bool periodicZ = false;
    bool writeVtk = true;  // false: only print diagnostics at frame steps
    int threads = 0;       // 0 = OpenMP default
};

struct Sim
{
    Params p;
    Options opt;
    long long N = 0;          // number of cells (the populations are p.stride apart)
    long long fluidCells = 0;
    int maxIt = 0;
    int frameEvery = 0;       // 0 = never
    real uMax = 0;            // lid velocity / peak inlet velocity
    double reynolds = 0;
    double nu = 0;
    double tau = 0;
    double sigma = 0;         // width of the start-up ramp, in steps
    double uMean = 0;         // mean inlet velocity (problem 2)
    double frontalArea = 0;   // obstacle area projected on the y-z plane (problem 2)
    std::vector<std::uint8_t> type;

    real uAt(const int step) const; // ramped lid / inlet velocity
    bool isFrame(const int step) const
    {
        return frameEvery > 0 && step % frameEvery == 0;
    }
};

struct RunStats
{
    double seconds = 0; // wall time spent in the time loop, output excluded
    double mlups = 0;   // million lattice (fluid cell) updates per second
};

Sim loadSim(const Options &opt);

/// Diagnostics + VTK frame + force log. Called by both back ends with host arrays.
class FrameWriter
{
  public:
    explicit FrameWriter(const Sim &sim);
    void write(int step, const real *rho, const real *ux, const real *uy, const real *uz, const double force[3]);

  private:
    const Sim &sim;
    std::ofstream forces;
    int frame = 0;
};

RunStats runCpu(const Sim &sim);
#ifdef LBM3D_USE_CUDA
RunStats runGpu(const Sim &sim);
#endif

} // namespace lbm3d

#endif // LBM3D_SIM3D_HPP
