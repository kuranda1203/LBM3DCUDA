# 交接说明（给在另一台机器上接手的会话）

本仓库是 https://github.com/AndreaTorti-01/Lattice-Boltzmann-Method 的克隆，用于“并行与分布式程序设计”课程大作业。
在其基础上新增了三维求解器 `lbm3d/`（D3Q19 + TRT），原二维代码未改动。
代码托管在 https://github.com/kuranda1203/LBM3DCUDA （`origin`），原作者仓库是 `upstream`。
详细说明、验证结果和性能数据见 `lbm3d/README.md`。

## 当前状态

- 串行 / OpenMP：已在 Apple M3 上编译、验证（Ghia 方腔、Schäfer–Turek 圆柱 Cd）并测过加速比。
- CUDA（`lbm3d/src/Gpu3D.cu`）：已在 RTX 5060（CUDA 13.0）上用 nvcc 编译、与 CPU 对比并测过性能
  （128³ 1573 MLUPS，256³ 1857 MLUPS），数据在 `lbm3d/README.md`。唯一需要的修改是 CMake 里给目标补上
  `CUDA_ARCHITECTURES`。`-DLBM3D_DOUBLE=ON` 的 GPU 版、block 尺寸（32×4×4）调优还没试。
- `lbm3d/scripts/check_cavity.py` 里的 Ghia 参考值和 README 里的 Schäfer–Turek 参考范围是凭记忆录入的，尚未对照原文。

## GPU 机器（2026-11-10）

- 局域网里的一台 Ubuntu 24.04 主机（Ryzen 7 9700X，RTX 5060 8 GB，驱动 580.178，CUDA 13.0 在
  `/usr/local/cuda`，不在默认 PATH 里），从 Mac 用密钥登录（地址和用户名不写进仓库）。
- 仓库克隆在 `~/LBM3D/LBM3DCUDA`，`export PATH=/usr/local/cuda/bin:$PATH` 后 `./lbm3d/compile.sh` 构建到 `build3d/`。

## 还没做的

- 没有 MPI 版本；原二维代码的问题类型 3 没有移植；三维算例没有和文献定量对比。

## 约定

- 不要改动原二维代码（`src/`、`include/`）。
- 物理只写在 `Core3D.hpp` 一处，CPU/GPU 共用；改它之后要重跑 README 里的验证。
- 用户用中文交流；代码注释用英文。
