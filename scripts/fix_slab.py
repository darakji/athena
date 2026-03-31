

import argparse
import sys
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter, CifParser

def load_structure(path: str) -> Structure:
    """
    Robust loader:
    - Try Structure.from_file
    - If CIF parse fails, try CifParser(...).get_structures()
    - If still fails, try POSCAR
    """
    # Quick sniff
    first = ""
    try:
        with open(path, "r", errors="ignore") as f:
            first = f.readline().strip()
    except Exception:
        pass

    # 1) Generic
    try:
        return Structure.from_file(path)
    except Exception as e1:
       
        if first.lower().startswith("data_") or path.lower().endswith(".cif"):
            try:
                parser = CifParser(path, occupancy_tolerance=2.0)
                structs = parser.get_structures(primitive=False)
                if not structs:
                    raise ValueError("CifParser returned zero structures")
                return structs[0]
            except Exception as e2:
              
                try:
                    return Poscar.from_file(path).structure
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to read structure from '{path}'.\n"
                        f"Structure.from_file error: {e1}\n"
                        f"CifParser error: {e2}\n"
                        f"POSCAR fallback error: {e3}\n"
                    )
        else:
          
            try:
                return Poscar.from_file(path).structure
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to read structure from '{path}'.\n"
                    f"Structure.from_file error: {e1}\n"
                    f"POSCAR fallback error: {e3}\n"
                )


def write_outputs(struct: Structure, out_prefix: str):
    cif_path = f"{out_prefix}.cif"
    poscar_path = f"{out_prefix}.vasp"
    CifWriter(struct).write_file(cif_path)
    Poscar(struct).write_file(poscar_path)
    print(f"[OK] Wrote: {cif_path}")
    print(f"[OK] Wrote: {poscar_path}")



def c_unit_and_len(lattice: Lattice):
    """
    Return unit vector along c (Cartesian) and c length.
    Works for tilted cells too (uses the c lattice vector).
    """
    c_vec = np.array(lattice.matrix[2], dtype=float)
    c_len = float(np.linalg.norm(c_vec))
    if c_len <= 0:
        raise ValueError("Invalid lattice: |c| <= 0")
    c_hat = c_vec / c_len
    return c_hat, c_len, c_vec


def project_along_c(cart_coords: np.ndarray, lattice: Lattice) -> np.ndarray:
    c_hat, _, _ = c_unit_and_len(lattice)
    return cart_coords @ c_hat  # dot product along c unit



def unwrap_by_largest_gap(struct: Structure, axis: int = 2) -> Structure:
    """
    Make slab contiguous along given fractional axis (default c = 2):
    - compute fractional coords modulo 1
    - find the largest gap in sorted fractional axis values
    - choose the cut in that largest gap (vacuum)
    - shift atoms with f < cut by +1 (ONLY in fractional axis)
    Atom count stays identical.
    """
    frac = struct.frac_coords.copy()
    f = frac[:, axis] % 1.0

    
    order = np.argsort(f)
    f_sorted = f[order]

    gaps = np.diff(f_sorted)
    wrap_gap = (f_sorted[0] + 1.0) - f_sorted[-1]
    gaps = np.concatenate([gaps, [wrap_gap]])

    k = int(np.argmax(gaps))  
    if k < len(f_sorted) - 1:
        left = f_sorted[k]
        right = f_sorted[k + 1]
        cut = 0.5 * (left + right)
    else:
       
        left = f_sorted[-1]
        right = f_sorted[0] + 1.0
        cut = (0.5 * (left + right)) % 1.0

    
    shift_mask = f < cut
    frac2 = frac.copy()
    frac2[shift_mask, axis] += 1.0

    
    new = Structure(struct.lattice, struct.species, frac2, coords_are_cartesian=False, to_unit_cell=False)
    return new


def center_and_set_vacuum(struct: Structure, vac: float, axis: int = 2) -> Structure:
    """
    After unwrapping, (optionally) recenter slab and set vacuum thickness along c.

    - Compute slab thickness along c (project cart coords)
    - New c length = thickness + vac
    - Keep a,b same; scale c-vector direction to new length
    - Translate slab so it is centered in new cell (symmetric vacuum)
    """
    if vac is None or vac <= 0:
        return struct

    lat = struct.lattice
    c_hat, c_len_old, c_vec_old = c_unit_and_len(lat)

    cart = np.array(struct.cart_coords, dtype=float)
    z = project_along_c(cart, lat)
    zmin, zmax = float(z.min()), float(z.max())
    thickness = zmax - zmin
    if thickness <= 0:
        
        return struct

    new_c_len = thickness + float(vac)
    scale = new_c_len / c_len_old
    c_vec_new = c_vec_old * scale

    
    a_vec = np.array(lat.matrix[0], dtype=float)
    b_vec = np.array(lat.matrix[1], dtype=float)
    new_lat = Lattice(np.vstack([a_vec, b_vec, c_vec_new]))

    
    slab_center_old = 0.5 * (zmin + zmax)
    target_center = 0.5 * new_c_len
    delta = (target_center - slab_center_old) * c_hat

    cart_shifted = cart + delta

    
    new_struct = Structure(new_lat, struct.species, cart_shifted, coords_are_cartesian=True, to_unit_cell=True)
    return new_struct





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CIF/POSCAR (e.g., slab.cif or Supercell-c3.vasp)")
    ap.add_argument("--out", dest="out", required=True, help="Output prefix (writes .cif and .vasp)")
    ap.add_argument("--mode", choices=["unwrap", "cut-middle"], default="unwrap",
                    help="unwrap: fix wrapped atoms (same atom count). "
                         "cut-middle: extract middle slab from 3x supercell (≈N/3 atoms).")
    ap.add_argument("--vac", type=float, default=0.0, help="Vacuum thickness (Å) to set along c after fixing/ কাটting.")
    args = ap.parse_args()

    s = load_structure(args.inp)
    print(f"[INFO] Loaded structure: {s.num_sites} atoms")

    if args.mode == "unwrap":
        s2 = unwrap_by_largest_gap(s, axis=2)
        
        s2 = center_and_set_vacuum(s2, vac=args.vac, axis=2)
        print(f"[INFO] unwrap done. Atom count: {s2.num_sites} (should be unchanged)")
        write_outputs(s2, args.out)

    elif args.mode == "cut-middle":
        s2 = cut_middle_third(s, vac=args.vac, axis=2)
        print(f"[INFO] cut-middle done. Atom count: {s2.num_sites} (should be ~1/3 of input)")
        write_outputs(s2, args.out)

    else:
        raise RuntimeError("Unknown mode")

    return 0


if __name__ == "__main__":
    sys.exit(main())


