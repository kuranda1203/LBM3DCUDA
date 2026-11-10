#include "Sim3D.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbm3d;

static void usage(const char *exe)
{
    std::cout << "Usage: " << exe << " <input file> <frames> [options]\n"
              << "  <frames>        number of output frames; 0 = no output (benchmark)\n"
              << "  -gpu            run on the GPU (CUDA build only)\n"
              << "  -threads=<n>    number of OpenMP threads (default: OMP_NUM_THREADS / all cores)\n"
              << "  -periodic-z     periodic instead of solid walls along z (quasi 2D, for validation)\n"
              << "  -no-vtk         print the diagnostics at every frame but do not write files\n"
              << "  -out=<dir>      output folder (default: outputs3d, must exist)\n";
}

int main(int argc, char const *argv[])
{
    if (argc < 3)
    {
        usage(argv[0]);
        return 1;
    }

    Options opt;
    opt.inputFile = argv[1];
    opt.frames = std::atoi(argv[2]);
    for (int i = 3; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-gpu")
            opt.gpu = true;
        else if (a == "-periodic-z")
            opt.periodicZ = true;
        else if (a == "-no-vtk")
            opt.writeVtk = false;
        else if (a.rfind("-threads=", 0) == 0)
            opt.threads = std::atoi(a.c_str() + 9);
        else if (a.rfind("-out=", 0) == 0)
            opt.outDir = a.substr(5);
        else
        {
            std::cout << "Unknown option " << a << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    try
    {
        const Sim sim = loadSim(opt);
        const Params &p = sim.p;

        int threads = 1;
#ifdef _OPENMP
        if (opt.threads > 0)
            omp_set_num_threads(opt.threads);
        threads = omp_get_max_threads();
#else
        if (opt.threads > 1)
            std::cout << "Warning: built without OpenMP, running on one thread\n";
#endif

        std::printf("D3Q19 TRT, %s precision\n", sizeof(real) == 8 ? "double" : "single");
        std::printf("problem %d (%s), lattice %d x %d x %d = %lld cells (%lld fluid)%s\n", p.problem,
                    p.problem == 1 ? "lid driven cavity" : "channel flow", p.nx, p.ny, p.nz, sim.N, sim.fluidCells,
                    p.periodicZ ? ", periodic along z" : "");
        std::printf("Re %.1f, u %.4f, nu %.6f, tau %.5f, steps %d\n", sim.reynolds, static_cast<double>(sim.uMax),
                    sim.nu, sim.tau, sim.maxIt);
        std::printf("memory for the populations: %.1f MB\n", 2.0 * Q * sizeof(real) * p.stride / 1048576.0);

        RunStats stats;
        if (opt.gpu)
        {
#ifdef LBM3D_USE_CUDA
            std::printf("back end: CUDA\n");
            stats = runGpu(sim);
#else
            std::cout << "This executable was built without CUDA support\n";
            return 1;
#endif
        }
        else
        {
            std::printf("back end: CPU, %d thread%s\n", threads, threads == 1 ? "" : "s");
            stats = runCpu(sim);
        }

        std::printf("time loop: %.3f s, %.2f MLUPS\n", stats.seconds, stats.mlups);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
