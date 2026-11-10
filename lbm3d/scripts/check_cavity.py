"""Compare a quasi 2D cavity run (problem 1, -periodic-z, Re = 100) with Ghia, Ghia & Shin (1982).

    ./build3d/lbm3d lbm3d/data/cavity_q2d_re100.txt 4 -periodic-z -out=lbm3d/outputs3d
    python3 lbm3d/scripts/check_cavity.py lbm3d/outputs3d

Prints u / u_lid on the vertical centre line against the reference values (Table I, Re = 100).
"""
import glob
import os
import sys

import numpy as np

from vtkio import read_frame

# distance from the wall opposite to the lid, u / u_lid
GHIA_RE100 = [
    (0.9766, 0.84123), (0.9688, 0.78871), (0.9609, 0.73722), (0.9531, 0.68717), (0.8516, 0.23151),
    (0.7344, 0.00332), (0.6172, -0.13641), (0.5000, -0.20581), (0.4531, -0.21090), (0.2813, -0.15662),
    (0.1719, -0.10150), (0.1016, -0.06434), (0.0703, -0.04775), (0.0625, -0.04192), (0.0547, -0.03717),
]


def main():
    folder = sys.argv[1]
    u_lid = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    path = sorted(glob.glob(os.path.join(folder, "frame_*.vtk")))[-1]
    step, u, _, _ = read_frame(path)
    _, nz, ny, nx = u.shape
    ux = u[0].astype(np.float64).mean(axis=0)  # average over the periodic direction
    centre = 0.5 * (ux[:, nx // 2 - 1] + ux[:, nx // 2]) if nx % 2 == 0 else ux[:, nx // 2]
    # the lid is the wall at y = -1/2; add the two wall values for the interpolation
    dist = np.concatenate(([1.0], 1.0 - (np.arange(ny) + 0.5) / ny, [0.0]))
    prof = np.concatenate(([u_lid], centre, [0.0])) / u_lid

    print(f"{os.path.basename(path)} (step {step}), lattice {nx} x {ny} x {nz}")
    print("   height     Ghia       LBM      diff")
    worst = 0.0
    for h, ref in GHIA_RE100:
        val = np.interp(h, dist[::-1], prof[::-1])
        worst = max(worst, abs(val - ref))
        print(f"  {h:7.4f}  {ref:8.5f}  {val:8.5f}  {val - ref:+8.5f}")
    print(f"largest difference: {worst:.5f} (in units of the lid velocity)")
    print(f"variation of ux along z: {np.abs(u[0] - u[0].mean(axis=0)).max():.2e}")


if __name__ == "__main__":
    main()
