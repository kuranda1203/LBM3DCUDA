"""Bird's-eye rendering of lbm3d frames: the obstacle as a shaded solid plus two translucent slices
of the velocity magnitude (one horizontal, one vertical), joined into an animated GIF.

    python3 lbm3d/scripts/render3D.py lbm3d/outputs3d -o flow.gif [--first 40] [--every 1] [--yslice 42]
            [--zslice 96] [--zslice-from 150] [--azim 35] [--elev 30] [--px 1.4] [--vmax 0.1] [--fps 12] [--still frame.png]

Needs numpy, scipy, matplotlib (colour map only) and Pillow. The camera is orthographic: it looks at the
domain from upstream, --elev degrees above the horizon and --azim degrees to the +z side.
The solid is found by marching the view rays through the obstacle mask once; the slices are intersected
analytically, so a frame costs little more than reading its .vtk file.
"""
import argparse
import glob
import os

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from vtkio import read_frame


class Scene:
    def __init__(self, obstacle, a):
        self.nz, self.ny, self.nx = obstacle.shape
        nx, ny, nz = self.nx, self.ny, self.nz
        az, el = np.radians(a.azim), np.radians(a.elev)
        # forward: from the camera into the scene (x downstream, y up, z span)
        f = np.array([np.cos(el) * np.cos(az), -np.sin(el), -np.cos(el) * np.sin(az)])
        r = np.cross(f, [0.0, 1.0, 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        corners = np.array([[x, y, z] for x in (-0.5, nx - 0.5) for y in (-0.5, ny - 0.5) for z in (-0.5, nz - 0.5)])
        centre = corners.mean(0)
        s, t, d = (corners - centre) @ r, (corners - centre) @ u, (corners - centre) @ f
        self.w = int(np.ceil((s.max() - s.min()) * a.px)) + 8
        self.h = int(np.ceil((t.max() - t.min()) * a.px)) + 8
        ss = (np.arange(self.w) - (self.w - 1) / 2) / a.px
        tt = ((self.h - 1) / 2 - np.arange(self.h)) / a.px
        S, T = np.meshgrid(ss, tt)
        depth = d.max() - d.min()
        self.o = centre + S[..., None] * r + T[..., None] * u - (depth / 2 + 2) * f  # ray origins [h, w, 3]
        self.f = f
        self.far = depth + 4

        # opaque layer: floor, then the solid in front of it
        col = np.ones((self.h, self.w, 3)) * np.array(a.background)
        zbuf = np.full((self.h, self.w), np.inf)
        tf = (-0.5 - self.o[..., 1]) / f[1]
        p = self.o + tf[..., None] * f
        on = (p[..., 0] > -0.5) & (p[..., 0] < nx - 0.5) & (p[..., 2] > -0.5) & (p[..., 2] < nz - 0.5)
        grid = (np.abs((p[..., 0] + 0.5) % 32 - 16) > 15.4) | (np.abs((p[..., 2] + 0.5) % 32 - 16) > 15.4)
        col[on] = np.where(grid[on, None], 0.62, 0.74)
        zbuf[on] = tf[on]

        solid = obstacle.astype(np.float32)
        smooth = gaussian_filter(solid, 1.2)
        gz, gy, gx = np.gradient(smooth)
        hit = np.full((self.h, self.w), np.inf)
        active = np.ones((self.h, self.w), bool)
        for k in np.arange(0.0, self.far, 0.5):
            idx = np.rint(self.o[active] + k * f).astype(np.int64)
            ok = ((idx >= 0) & (idx < [nx, ny, nz])).all(1)
            inside = np.zeros(len(idx), bool)
            inside[ok] = obstacle[idx[ok, 2], idx[ok, 1], idx[ok, 0]]
            ys, xs = np.nonzero(active)
            hit[ys[inside], xs[inside]] = k
            active[ys[inside], xs[inside]] = False
        got = np.isfinite(hit) & (hit < zbuf)
        idx = np.rint(self.o[got] + hit[got, None] * f).astype(np.int64)
        n = -np.stack([gx[idx[:, 2], idx[:, 1], idx[:, 0]], gy[idx[:, 2], idx[:, 1], idx[:, 0]],
                       gz[idx[:, 2], idx[:, 1], idx[:, 0]]], 1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
        light = np.array([-0.45, 0.75, 0.5])
        light /= np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1) + 0.15 * np.clip(n @ -f, 0, 1) ** 8
        col[got] = np.clip(shade[:, None] * np.array(a.solid), 0, 1)
        zbuf[got] = hit[got]
        self.opaque, self.zbuf = col, zbuf

        # the two slices: ray parameter, sample indices and visibility, fixed for all frames
        self.planes = []
        for axis, value in ((1, a.yslice), (2, a.zslice if a.zslice is not None else nz // 2)):
            t = (value - self.o[..., axis]) / f[axis]
            p = np.rint(self.o + t[..., None] * f).astype(np.int64)
            p[..., axis] = value
            vis = ((p >= 0) & (p < [nx, ny, nz])).all(-1) & (t < self.zbuf) & (t > 0)
            if axis == 2:
                vis &= p[..., 0] >= a.zslice_from
            pv = p[vis]
            vis[vis] = ~obstacle[pv[:, 2], pv[:, 1], pv[:, 0]]
            self.planes.append((t, p, vis))
        self.planes.sort(key=lambda pl: -np.nanmean(np.where(pl[2], pl[0], np.nan)))  # far one first
        self.cmap, self.vmax, self.alpha = colormaps[a.cmap], a.vmax, a.alpha

    def render(self, mag, label):
        img = self.opaque.copy()
        # where both slices cover a pixel the nearer one must be drawn last
        t0, t1 = self.planes[0][0], self.planes[1][0]
        both = self.planes[0][2] & self.planes[1][2]
        order = [(0, np.ones_like(both)), (1, np.ones_like(both)), (0, both & (t0 < t1))]
        for i, mask in order:
            t, p, vis = self.planes[i]
            m = vis & mask
            pv = p[m]
            rgb = self.cmap(np.clip(mag[pv[:, 2], pv[:, 1], pv[:, 0]] / self.vmax, 0, 1))[:, :3]
            img[m] = (1 - self.alpha) * img[m] + self.alpha * rgb
        out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        ImageDraw.Draw(out).text((10, 8), label, fill=(20, 20, 20))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("-o", "--output", default="flow.gif")
    ap.add_argument("--first", type=int, default=0, help="index of the first frame to use")
    ap.add_argument("--last", type=int, default=10 ** 9)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--yslice", type=int, required=True, help="height of the horizontal slice (cells)")
    ap.add_argument("--zslice", type=int, help="position of the vertical slice (default: nz/2)")
    ap.add_argument("--zslice-from", type=int, default=0,
                    help="draw the vertical slice only for x >= this, to keep the obstacle in full view")
    ap.add_argument("--azim", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--px", type=float, default=1.4, help="pixels per cell")
    ap.add_argument("--vmax", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--colors", type=int, default=128, help="GIF palette size")
    ap.add_argument("--solid", type=float, nargs=3, default=[0.93, 0.9, 0.86])
    ap.add_argument("--background", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--still", help="also save the last rendered frame as a png")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.folder, "frame_*.vtk")))[a.first:a.last + 1:a.every]
    if not files:
        raise SystemExit("no frames found")
    scene, images = None, []
    for path in files:
        step, u, _, obstacle = read_frame(path)
        if scene is None:
            scene = Scene(obstacle, a)
            print(f"image {scene.w} x {scene.h}, {len(files)} frames")
        mag = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        images.append(scene.render(mag, f"step {step}"))
        print(os.path.basename(path), flush=True)
    if a.still:
        images[-1].save(a.still)
    pal = [im.quantize(a.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
    pal[0].save(a.output, save_all=True, append_images=pal[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"{a.output}: {os.path.getsize(a.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
