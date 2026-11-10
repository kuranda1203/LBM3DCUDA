"""Turn a triangle mesh (3MF or STL) into an lbm3d input file for channel flow (problem 2).

    python3 lbm3d/scripts/voxelize.py lbm3d/models/model.3mf --info
    python3 lbm3d/scripts/voxelize.py lbm3d/models/model.3mf --rot x-90 --size 384 160 160 --height 110 \
            --re 300 --steps 20000 --u 0.08 -o lbm3d/data/model.txt --preview model.png

Only numpy is needed (matplotlib for --preview). The mesh must be closed; several closed parts may overlap.

Solver axes: the flow goes along +x, y is "up", z is the span. --rot turns the model before it is placed,
e.g. "x-90" maps a z-up model to y-up, "x-90,y90" then turns it about the vertical axis.
--move shifts one part of a multi-part 3MF first (a figure printed as separate head and body, say).
The model is scaled so that its extent along y is --height cells, centred in z, and either stands on the
bottom wall (default) or floats in the middle of the channel (--float).

A cell is solid when its centre is inside the mesh: rays along z through every (x, y) column, signed
crossings summed into a winding number, so overlapping parts are merged correctly.
"""
import argparse
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
PROD = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}"


def _transform(text):
    """3MF transform: 12 numbers, row-major 4x3, applied to row vectors."""
    if not text:
        return np.eye(3), np.zeros(3)
    m = np.array([float(t) for t in text.split()]).reshape(4, 3)
    return m[:3], m[3]


def load_3mf(path):
    """Returns a list of (vertices [n,3], triangles [m,3]), one per build item, in build coordinates."""
    z = zipfile.ZipFile(path)
    docs = {}

    def objects(model_path):
        if model_path not in docs:
            root = ET.fromstring(z.read(model_path.lstrip("/")))
            docs[model_path] = (root, {o.get("id"): o for o in root.iter(CORE + "object")})
        return docs[model_path]

    def resolve(model_path, oid):
        obj = objects(model_path)[1][oid]
        mesh = obj.find(CORE + "mesh")
        if mesh is not None:
            v = np.array([[float(e.get("x")), float(e.get("y")), float(e.get("z"))]
                          for e in mesh.find(CORE + "vertices")])
            t = np.array([[int(e.get("v1")), int(e.get("v2")), int(e.get("v3"))]
                          for e in mesh.find(CORE + "triangles")], dtype=np.int64)
            return [(v, t)]
        out = []
        for c in obj.find(CORE + "components"):
            a, b = _transform(c.get("transform"))
            for v, t in resolve(c.get(PROD + "path") or model_path, c.get("objectid")):
                out.append((v @ a + b, t))
        return out

    main = "/3D/3dmodel.model"
    parts = []
    for item in objects(main)[0].find(CORE + "build"):
        a, b = _transform(item.get("transform"))
        vs, ts, off = [], [], 0
        for v, t in resolve(item.get(PROD + "path") or main, item.get("objectid")):
            vs.append(v @ a + b)
            ts.append(t + off)
            off += len(v)
        parts.append((np.concatenate(vs), np.concatenate(ts)))
    return parts


def load_stl(path):
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else -1
    if 84 + 50 * n == len(data):  # binary
        rec = np.frombuffer(data, offset=84, count=n,
                            dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        v = rec["v"].reshape(-1, 3).astype(float)
    else:  # ascii
        v = np.array([[float(x) for x in line.split()[1:4]]
                      for line in data.decode("ascii", "replace").splitlines()
                      if line.strip().startswith("vertex")])
    return [(v, np.arange(len(v)).reshape(-1, 3))]


def load(path):
    return load_3mf(path) if path.lower().endswith(".3mf") else load_stl(path)


def rotation(spec):
    r = np.eye(3)
    for token in filter(None, (spec or "").split(",")):
        axis, angle = token[0].lower(), np.radians(float(token[1:]))
        c, s = np.cos(angle), np.sin(angle)
        m = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
             "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        r = np.array(m) @ r
    return r


def open_edges(tri):
    """Number of edges not shared by exactly two triangles, after welding identical vertices is NOT done:
    a mesh exported with duplicated vertices reports many, which is harmless for the voxelisation."""
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts != 2).sum())


def voxelize(v, tri, shape):
    """v in grid coordinates (cell centres at integers). Returns a bool array [nx, ny, nz]."""
    nx, ny, nz = shape
    # shift by a tiny irrational amount so that no ray passes exactly through a vertex or an edge
    p = v + np.array([1.2345e-5, 2.3456e-5, 0.0])
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    lo = np.ceil(np.minimum(np.minimum(a, b), c)[:, :2]).astype(np.int64)
    hi = np.floor(np.maximum(np.maximum(a, b), c)[:, :2]).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [nx - 1, ny - 1])
    w = hi - lo + 1
    keep = (w[:, 0] > 0) & (w[:, 1] > 0) & (det != 0)
    a, b, c, det, lo, w = a[keep], b[keep], c[keep], det[keep], lo[keep], w[keep]

    winding = np.zeros((nx, ny, nz + 1), dtype=np.int32)
    count = w[:, 0] * w[:, 1]
    order = np.argsort(count, kind="stable")
    start = 0
    while start < len(order):  # chunks of similar footprint keep the temporaries small
        stop = start
        cells = 0
        while stop < len(order) and cells + count[order[stop]] <= 4_000_000:
            cells += count[order[stop]]
            stop += 1
        stop = max(stop, start + 1)
        sel = order[start:stop]
        start = stop
        n = count[sel]
        owner = np.repeat(np.arange(len(sel)), n)
        local = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        t = sel[owner]
        x = lo[t, 0] + local // w[t, 1]
        y = lo[t, 1] + local % w[t, 1]
        # barycentric coordinates of the column in the projected triangle
        l1 = ((x - a[t, 0]) * (c[t, 1] - a[t, 1]) - (c[t, 0] - a[t, 0]) * (y - a[t, 1])) / det[t]
        l2 = ((b[t, 0] - a[t, 0]) * (y - a[t, 1]) - (x - a[t, 0]) * (b[t, 1] - a[t, 1])) / det[t]
        hit = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        x, y, t, l1, l2 = x[hit], y[hit], t[hit], l1[hit], l2[hit]
        zc = a[t, 2] + l1 * (b[t, 2] - a[t, 2]) + l2 * (c[t, 2] - a[t, 2])
        k = np.clip(np.ceil(zc), 0, nz).astype(np.int64)  # first cell whose centre is above the crossing
        np.add.at(winding, (x, y, k), np.where(det[t] > 0, 1, -1).astype(np.int32))
    return np.cumsum(winding, axis=2)[:, :, :nz] != 0


def preview(solid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny, nz = solid.shape

    def depth(axis, flip):
        s = np.flip(solid, axis) if flip else solid
        d = np.where(s.any(axis), s.argmax(axis), np.nan).astype(float)
        return d

    views = [("seen from upstream (looking along +x)", depth(0, False), "z", "y"),
             ("side (looking along -z)", depth(2, True).T, "x", "y"),
             ("top (looking along -y)", depth(1, True).T, "x", "z")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, img, xl, yl) in zip(axes, views):
        ax.imshow(img, origin="lower", cmap="copper_r", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"preview written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", help=".3mf or .stl")
    ap.add_argument("--info", action="store_true", help="print the parts and their bounding boxes, then exit")
    ap.add_argument("--parts", type=int, nargs="+", help="which parts of a 3MF to use (default: all)")
    ap.add_argument("--move", type=float, nargs=4, action="append", default=[], metavar=("PART", "DX", "DY", "DZ"),
                    help="shift one part (model units, before --rot), to assemble a model stored in pieces")
    ap.add_argument("--rot", default="", help='rotations in order, e.g. "x-90,y180" (degrees)')
    ap.add_argument("--size", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--height", type=float, help="extent of the model along y, in cells")
    ap.add_argument("--xpos", type=float, help="x of the model centre (default: nx/4)")
    ap.add_argument("--float", dest="floating", action="store_true", help="centre in y instead of standing on y=0")
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--u", type=float, default=0.08, help="peak inlet velocity (lattice units, keep <= 0.1)")
    ap.add_argument("-o", "--output")
    ap.add_argument("--preview", help="png with three depth-shaded views of the voxel model")
    a = ap.parse_args()

    parts = load(a.mesh)
    for i, (v, t) in enumerate(parts):
        ext = v.max(0) - v.min(0)
        print(f"part {i}: {len(v)} vertices, {len(t)} triangles, extent {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}, "
              f"min {np.round(v.min(0), 1)}, open edges {open_edges(t)}")
    if a.info:
        return
    if not (a.size and a.height and (a.output or a.preview)):
        ap.error("--size, --height and -o (or --preview) are needed")

    for i, dx, dy, dz in a.move:
        parts[int(i)] = (parts[int(i)][0] + [dx, dy, dz], parts[int(i)][1])
    use = a.parts if a.parts else range(len(parts))
    vs, ts, off = [], [], 0
    for i in use:
        vs.append(parts[i][0])
        ts.append(parts[i][1] + off)
        off += len(parts[i][0])
    v, tri = np.concatenate(vs) @ rotation(a.rot).T, np.concatenate(ts)

    nx, ny, nz = a.size
    lo, hi = v.min(0), v.max(0)
    scale = a.height / (hi[1] - lo[1])
    v = (v - (lo + hi) / 2) * scale  # centred at the origin, in cells
    half = (hi - lo) * scale / 2
    xpos = a.xpos if a.xpos is not None else nx / 4
    # cell centres are at integers and the walls half a cell outside the first and last cell
    ypos = (ny - 1) / 2 if a.floating else half[1] - 0.5
    v += [xpos, ypos, (nz - 1) / 2]
    print(f"model in cells: {2 * half[0]:.1f} x {2 * half[1]:.1f} x {2 * half[2]:.1f} "
          f"(blockage {4 * half[1] * half[2] / (ny * nz) * 100:.1f}% of the cross section by bounding box)")

    solid = voxelize(v, tri, (nx, ny, nz))
    print(f"{int(solid.sum())} solid cells, frontal area {int(solid.any(0).sum())} cells "
          f"({solid.any(0).sum() / (ny * nz) * 100:.1f}% of the cross section)")
    if solid[0].any() or solid[-1].any():
        print("WARNING: the model touches the inlet or the outlet")

    if a.preview:
        preview(solid, a.preview)
    if a.output:
        xs, ys, zs = np.nonzero(solid.transpose(2, 1, 0))[::-1]  # sorted by z, then y, then x
        with open(a.output, "w") as f:
            f.write(f"2\n{nx} {ny} {nz}\n{a.re} {a.steps} {a.u}\n")
            np.savetxt(f, np.column_stack([xs, ys, zs]), fmt="%d")
        print(f"{a.output}: {len(xs)} obstacle cells")


if __name__ == "__main__":
    main()
