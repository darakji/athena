from dftfe.utils import build_dataset

build_dataset(
    input_dir="/home/mehuldarak/athena/dftfe_logs/selected_iteration1_superSeed_dftfe_inputs_centered",
    output_file="/home/mehuldarak/athena/fps_dftfe_and_emb_data/superseed_fps_dftfe_data_centered.extxyz",
    log_extension="*.op",      # Supports comma-separated extensions or lists
    extract_properties=['energy', 'force'],  # Optional: Exclusively extracts these properties. If omitted, extracts whatever is found.
    append=False,                            # Optional: If True, appends to the existing extxyz file. Default is False (overwrites).
    model="MACE"                             # Optional: Targeting "MACE" natively renames energy/forces -> REF_energy/REF_forces.
)