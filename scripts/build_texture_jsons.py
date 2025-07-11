
import csv, json, os, shutil
from pathlib import Path

ROOT = Path(os.getenv("DATASET_PATH", "/workspace/dataset_culture_textures"))

CSV = ROOT / "all_metadata_enriched.csv"         # adjust if needed
OUT_ROOT = Path("textures_lora")                 # where your images are

with CSV.open(newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        rel  = row["relative_path"].replace("\\", "/")
        img  = OUT_ROOT / rel                    # e.g. textures_lora/Japanese/...
        if not img.exists():
            continue

        meta = {
            "prompt": row["enriched_prompt"] or row["prompt"],
            "path":   str(img)
        }
        json_path = img.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(meta, jf, ensure_ascii=False, indent=0)
print("✅  wrote JSON side-cars")
