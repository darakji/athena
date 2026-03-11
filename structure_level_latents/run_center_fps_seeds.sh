#!/bin/bash

IN_DIR="/home/mehuldarak/athena/structure_level_latents/fps_seed"
OUT_DIR="/home/mehuldarak/athena/structure_level_latents/fps_seed_centered"

VAC=20.0   # vacuum in Å (change if you want)

for cif in "${IN_DIR}"/*.cif; do
    name=$(basename "$cif" .cif)
    out="${OUT_DIR}/${name}"

    python fix_slab.py \
        --in "$cif" \
        --out "$out" \
        --mode unwrap \
        --vac ${VAC}
done