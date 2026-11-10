"""Generate lbm3d input files for channel flow (problem 2) around a sphere or a cylinder.

    python3 lbm3d/scripts/make_obstacle.py sphere   --size 200 80 80 --center 50 40 40 --radius 10 \
            --re 200 --steps 10000 --u 0.1 -o lbm3d/data/sphere.txt
    python3 lbm3d/scripts/make_obstacle.py cylinder --size 440 82 2 --center 40 40 --radius 10 \
            --re 82 --steps 40000 --u 0.1 -o lbm3d/data/cylinder_re20.txt

The cylinder axis is along z. The Reynolds number is defined as in the 2D code:
Re = (2/3 u) * ny / nu, i.e. on the channel height ny and on 2/3 of the peak inlet velocity.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=["sphere", "cylinder"])
    ap.add_argument("--size", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--center", type=float, nargs="+", required=True, help="x y z (sphere) or x y (cylinder)")
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--u", type=float, required=True, help="peak inlet velocity (lattice units, keep <= 0.1-0.2)")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    nx, ny, nz = a.size
    need = 3 if a.shape == "sphere" else 2
    if len(a.center) != need:
        ap.error(f"{a.shape} needs {need} center coordinates")
    r2 = a.radius * a.radius
    count = 0
    with open(a.output, "w") as f:
        f.write(f"2\n{nx} {ny} {nz}\n{a.re} {a.steps} {a.u}\n")
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    d2 = (x - a.center[0]) ** 2 + (y - a.center[1]) ** 2
                    if a.shape == "sphere":
                        d2 += (z - a.center[2]) ** 2
                    if d2 < r2:
                        f.write(f"{x} {y} {z}\n")
                        count += 1
    print(f"{a.output}: {count} obstacle cells")


if __name__ == "__main__":
    main()
