#ifndef LBM3D_CUDA_MOCK_HPP
#define LBM3D_CUDA_MOCK_HPP

// Minimal stand-in for the CUDA runtime, for machines without a CUDA toolkit.
// "Kernels" run on the host, one emulated thread after the other. Only used to check the logic
// of Gpu3D.cu (cmake -DLBM3D_CUDA_MOCK=ON); never use it to measure anything.

#include <cstdlib>
#include <cstring>

#define __global__
#define __device__
#define __host__

struct dim3
{
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_)
    {
    }
};

static dim3 blockIdx, blockDim, threadIdx, gridDim;

#define LBM_LAUNCH(kernel, grid, block, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        gridDim = (grid);                                                                                              \
        blockDim = (block);                                                                                            \
        for (unsigned bz_ = 0; bz_ < gridDim.z; ++bz_)                                                                 \
            for (unsigned by_ = 0; by_ < gridDim.y; ++by_)                                                             \
                for (unsigned bx_ = 0; bx_ < gridDim.x; ++bx_)                                                         \
                    for (unsigned tz_ = 0; tz_ < blockDim.z; ++tz_)                                                    \
                        for (unsigned ty_ = 0; ty_ < blockDim.y; ++ty_)                                                \
                            for (unsigned tx_ = 0; tx_ < blockDim.x; ++tx_)                                            \
                            {                                                                                          \
                                blockIdx = dim3(bx_, by_, bz_);                                                        \
                                threadIdx = dim3(tx_, ty_, tz_);                                                       \
                                kernel(__VA_ARGS__);                                                                   \
                            }                                                                                          \
    } while (0)

typedef int cudaError_t;
enum
{
    cudaSuccess = 0
};
enum cudaMemcpyKind
{
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost
};

inline cudaError_t cudaMalloc(void **ptr, size_t bytes)
{
    *ptr = std::malloc(bytes);
    return *ptr ? 0 : 1;
}
inline cudaError_t cudaFree(void *ptr)
{
    std::free(ptr);
    return 0;
}
inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind)
{
    std::memcpy(dst, src, bytes);
    return 0;
}
inline cudaError_t cudaMemset(void *dst, int value, size_t bytes)
{
    std::memset(dst, value, bytes);
    return 0;
}
inline cudaError_t cudaDeviceSynchronize()
{
    return 0;
}
inline cudaError_t cudaGetLastError()
{
    return 0;
}
inline const char *cudaGetErrorString(cudaError_t)
{
    return "mock error";
}
template <typename T> inline T atomicAdd(T *address, T value)
{
    const T old = *address;
    *address += value;
    return old;
}

#endif // LBM3D_CUDA_MOCK_HPP
