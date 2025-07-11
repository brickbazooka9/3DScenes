#!/usr/bin/env python
"""
visual_enrich_llava.py
----------------------
Generate rich, culturally-aware texture prompts with *local* LLaVA (Ollama).

Features
---------
• Resume: rows whose “enriched_prompt” is already present (and not an “[ERROR]”)
  are skipped automatically.

• Parallel processing (ThreadPool) – set NUM_WORKERS to taste.

• Robust: up to RETRIES per row, exponential back-off, configurable HTTP timeout.

• Final CSV keeps the original order & all original columns + “enriched_prompt”.
"""

from __future__ import annotations
import os, csv, time, base64, io
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

# --------------------------------------------------------------------------
#                            configuration
# --------------------------------------------------------------------------
ROOT = Path(os.getenv("DATASET_PATH", "/workspace/dataset_culture_textures"))
CSV_INPUT     = ROOT / "all_metadata.csv"
CSV_OUTPUT    = ROOT / "all_metadata_enriched.csv"

OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL_NAME    = "llava"

NUM_WORKERS   = 4          # threads talking to Ollama
HTTP_TIMEOUT  = 180        # seconds – long for first GPU warm-up
RETRIES       = 3          # per row
BACKOFF_BASE  = 2          # sec; actual wait = BACKOFF_BASE * attempt
SLEEP_BETWEEN = 0.1        # light idle to keep CPU cool
# --------------------------------------------------------------------------


def encode_image_b64(path: Path) -> str | None:
    """RGB-to-JPEG-to-base64; returns None on failure."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[encode] {path.name}: {exc}")
        return None


def call_llava(image_b64: str, short_label: str) -> str:
    """Hit local Ollama / LLaVA once, retrying on failures."""
    prompt = (
        "Describe this texture image in a richly detailed way, including its material, "
        "cultural origin, pattern, and overall visual aesthetic. Use the short label in quotes "
        "to guide you and finish with **one** concise line suitable as a Stable Diffusion prompt.\n\n"
        f'Short label: "{short_label}"'
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    last_err: str | Exception = "unknown"
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            js = resp.json()
            return (js.get("response") or "").strip() or "[ERROR] Empty response"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == RETRIES:
                break
            wait = BACKOFF_BASE * attempt  # 2 s, 4 s, …
            print(f"  ↻ retry {attempt}/{RETRIES} after {wait}s – {exc}")
            time.sleep(wait)

    return f"[ERROR] {last_err}"


# --------------------------------------------------------------------------
#                          CSV helpers
# --------------------------------------------------------------------------
def cleaned_reader(path: Path) -> List[Dict]:
    """Read a CSV **once**, removing BOM / stray quotes from header."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_enriched() -> dict[str, Dict]:
    """Return dict keyed by normalised relative_path → row dict."""
    if not CSV_OUTPUT.exists():
        return {}

    rows = cleaned_reader(CSV_OUTPUT)
    cache: dict[str, Dict] = {}
    for row in rows:
        key = row.get("relative_path", "").replace("\\", "/").strip()
        if key:
            cache[key] = row
    print(f"✅  Loaded {len(cache)} rows from existing enriched file")
    return cache


def needs_work(master: Dict, enriched: Dict | None) -> bool:
    """True if row absent, empty or has still an error."""
    if enriched is None:
        return True
    prompt_val = (enriched.get("enriched_prompt") or "").strip()
    return (not prompt_val) or prompt_val.startswith("[ERROR]")


# --------------------------------------------------------------------------
def enrich_worker(row: Dict) -> Dict:
    """Thread worker → always returns a row with enriched_prompt field."""
    rel = row["relative_path"].replace("\\", "/")
    img_path = ROOT / rel

    if not img_path.exists():
        row["enriched_prompt"] = "[ERROR] Image not found."
        return row

    img_b64 = encode_image_b64(img_path)
    if not img_b64:
        row["enriched_prompt"] = "[ERROR] Encoding failed."
        return row

    row["enriched_prompt"] = call_llava(img_b64, row["prompt"])
    time.sleep(SLEEP_BETWEEN)
    return row


# --------------------------------------------------------------------------
def main() -> None:
    if not CSV_INPUT.exists():
        raise FileNotFoundError(CSV_INPUT)

    master_rows = cleaned_reader(CSV_INPUT)
    enriched_cache = load_enriched()

    todo = [
        r for r in master_rows
        if needs_work(r, enriched_cache.get(r["relative_path"].replace("\\", "/")))
    ]
    print(f"🟢  {len(master_rows) - len(todo)} rows already OK, {len(todo)} need enrichment")

    # ---- parallel processing ------------------------------------------------
    if todo:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            fut_map = {pool.submit(enrich_worker, r): r for r in todo}
            for fut in tqdm(as_completed(fut_map),
                            total=len(todo),
                            desc="✨ Enriching"):
                new_row = fut.result()
                key = new_row["relative_path"].replace("\\", "/")
                enriched_cache[key] = new_row  # overwrite / add

    # ---- write merged CSV ---------------------------------------------------
    out_fields = list(master_rows[0].keys()) + ["enriched_prompt"]
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for row in master_rows:
            key = row["relative_path"].replace("\\", "/")
            merged = {**row, **enriched_cache.get(key, {})}
            merged.setdefault("enriched_prompt", "")
            writer.writerow(merged)

    print(f"🏁  Finished.  CSV saved at: {CSV_OUTPUT}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
