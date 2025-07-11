import os
import shutil

def copy_and_nest_h5_files(source_dir, dest_dir):
    """
    Copies each .h5 file from source_dir into its own subdirectory in dest_dir.
    Each subdirectory is named after the .h5 file (without extension).
    """
    os.makedirs(dest_dir, exist_ok=True)

    for fname in os.listdir(source_dir):
        if not fname.endswith(".h5"):
            continue

        src_path = os.path.join(source_dir, fname)
        dataset_name = os.path.splitext(fname)[0]
        dest_subdir = os.path.join(dest_dir, dataset_name)
        dest_path = os.path.join(dest_subdir, fname)

        if os.path.exists(dest_path):
            print(f"Already copied: {fname}")
            continue

        os.makedirs(dest_subdir, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"Copied {fname} → {dest_subdir}/")

    print("✅ Copy + nesting complete.")

source_dir = "/home/vincent/Senior thesis work/blechRNN-master/data/h5_files"       # ← Replace this
dest_dir = "/home/vincent/Senior thesis work/blechRNN-master/data/nested_h5_files"  # ← Replace this
copy_and_nest_h5_files(source_dir, dest_dir)