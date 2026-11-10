"""Plot slices of the velocity magnitude for every frame written by lbm3d.

    python3 lbm3d/scripts/plotting3D.py lbm3d/outputs3d [-vmax=0.15] [-gif]

For every frame_XXXX.vtk a frame_XXXX.png is written next to it, showing the mid planes
z = nz/2 (x-y view) and y = ny/2 (x-z view). With -gif the frames are also joined into
movie.gif. For real 3D rendering (isosurfaces, streamlines) open the .vtk files in ParaView.
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vtkio import read_frame


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    folder = sys.argv[1]
    vmax = None
    gif = False
    for a in sys.argv[2:]:
        if a.startswith("-vmax="):
            vmax = float(a[6:])
        elif a == "-gif":
            gif = True

    files = sorted(glob.glob(os.path.join(folder, "frame_*.vtk")))
    if not files:
        sys.exit(f"no frame_*.vtk in {folder}")

    pngs = []
    for path in files:
        step, u, _, obstacle = read_frame(path)
        speed = np.sqrt((u.astype(np.float64) ** 2).sum(axis=0))
        speed = np.ma.masked_where(obstacle, speed)
        nz, ny, nx = speed.shape
        top = vmax if vmax is not None else max(float(speed.max()), 1e-12)

        fig, axes = plt.subplots(2, 1, figsize=(9, 7.5 * (ny + nz) / nx + 1.6), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [ny, nz]})
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("0.6")
        # y grows downwards, as in the 2D code (the lid of the cavity is the top edge)
        im = axes[0].imshow(speed[nz // 2], origin="upper", cmap=cmap, vmin=0, vmax=top)
        axes[0].set_title(f"|u|, plane z = {nz // 2}, step {step}")
        axes[0].set_xlabel("x"), axes[0].set_ylabel("y")
        axes[1].imshow(speed[:, ny // 2, :], origin="lower", cmap=cmap, vmin=0, vmax=top)
        axes[1].set_title(f"|u|, plane y = {ny // 2}")
        axes[1].set_xlabel("x"), axes[1].set_ylabel("z")
        fig.colorbar(im, ax=axes, shrink=0.8, label="|u| (lattice units)")
        png = path[:-4] + ".png"
        fig.savefig(png, dpi=110)
        plt.close(fig)
        pngs.append(png)
        print(png)

    if gif:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in pngs]
        out = os.path.join(folder, "movie.gif")
        images[0].save(out, save_all=True, append_images=images[1:], duration=100, loop=0)
        print(out)


if __name__ == "__main__":
    main()
