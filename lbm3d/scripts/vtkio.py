"""Reader for the legacy binary VTK frames written by lbm3d."""
import numpy as np


def read_frame(path):
    """Returns (step, u, rho, obstacle); u has shape (3, nz, ny, nx), the others (nz, ny, nx)."""
    with open(path, "rb") as f:
        f.readline()
        step = int(f.readline().split()[-1])
        nx = ny = nz = None
        while True:
            line = f.readline().split()
            if line[0] == b"DIMENSIONS":
                nx, ny, nz = (int(v) for v in line[1:4])
            if line[0] == b"VECTORS":
                break
        n = nx * ny * nz
        u = np.frombuffer(f.read(12 * n), dtype=">f4").reshape(nz, ny, nx, 3).astype(np.float32)
        u = np.moveaxis(u, 3, 0)
        f.readline(), f.readline(), f.readline()  # newline, SCALARS density, LOOKUP_TABLE
        rho = np.frombuffer(f.read(4 * n), dtype=">f4").reshape(nz, ny, nx).astype(np.float32)
        f.readline(), f.readline(), f.readline()
        obstacle = np.frombuffer(f.read(n), dtype=np.uint8).reshape(nz, ny, nx).astype(bool)
    return step, u, rho, obstacle
