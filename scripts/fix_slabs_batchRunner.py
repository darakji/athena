import os
import subprocess

INPUT_DIR = "/home/mehuldarak/athena/Rajdeep_final_structures_burov/maybe_tests"
OUTPUT_DIR = "/home/mehuldarak/athena/Rajdeep_final_structures_burov/maybe_tests_centered"

SCRIPT = "/home/mehuldarak/athena/scripts/fix_slab.py"   # your script name

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".cif"):
        in_path = os.path.join(INPUT_DIR, fname)

        # remove .cif extension for output prefix
        base = os.path.splitext(fname)[0]
        out_prefix = os.path.join(OUTPUT_DIR, base)

        cmd = [
            "python",
            SCRIPT,
            "--in", in_path,
            "--out", out_prefix,
            "--mode", "unwrap",
            "--vac", "15"   # <-- change if needed
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

print("All files processed.")