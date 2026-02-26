import os
import shutil

# ==================================================
# Paths
# ==================================================
MD_FILE = "/home/mehuldarak/athena/li_llzo_unrelaxed_stacking/li_llzo_gap_scan_all.md"
SRC_BASE = "/home/mehuldarak/athena/li_llzo_unrelaxed_stacking"
DST_BASE = "/home/mehuldarak/athena/li_llzo_unrelaxed_bestgaps"

SUMMARY_MD = os.path.join(DST_BASE, "best_gap_summary.md")

os.makedirs(DST_BASE, exist_ok=True)

# ==================================================
# Parse MD file → best gap per interface
# ==================================================
best = {}
# best[interface] = (energy, gap, d_min)

with open(MD_FILE, "r") as f:
    for line in f:
        if not line.startswith("|"):
            continue
        if "Interface" in line or "---" in line:
            continue

        parts = [p.strip() for p in line.strip().split("|")[1:-1]]

        interface = parts[0]
        gap = float(parts[1])
        energy = float(parts[2])
        d_min = float(parts[3])

        if interface not in best or energy < best[interface][0]:
            best[interface] = (energy, gap, d_min)

# ==================================================
# Copy best-gap CIFs + write summary
# ==================================================
with open(SUMMARY_MD, "w") as log:
    log.write("# Best Gap Selection (Lowest Energy per Interface)\n\n")
    log.write("| Interface | Chosen gap (Å) | Energy (eV) | Min distance (Å) |\n")
    log.write("|-----------|----------------|-------------|------------------|\n")

    for interface, (energy, gap, d_min) in sorted(best.items()):

        gap_str = f"{gap:.2f}"

        src_cif = os.path.join(
            SRC_BASE,
            interface,
            f"Li_LLZO_gap_{gap_str}A.cif"
        )

        if not os.path.isfile(src_cif):
            print(f"⚠️ Missing CIF: {src_cif}")
            continue

        dst_cif = os.path.join(
            DST_BASE,
            f"{interface}_bestgap_{gap_str}A.cif"
        )

        shutil.copy2(src_cif, dst_cif)

        log.write(
            f"| {interface} | {gap_str} | {energy:.6f} | {d_min:.3f} |\n"
        )

        print(
            f"✔ {interface}: "
            f"gap = {gap_str} Å, "
            f"E = {energy:.4f} eV, "
            f"d_min = {d_min:.3f} Å"
        )

print("\n✔ All best-gap structures collected")
print(f"✔ Output directory: {DST_BASE}")
print(f"✔ Summary written to: {SUMMARY_MD}")