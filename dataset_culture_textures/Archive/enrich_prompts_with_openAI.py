#!/usr/bin/env python
"""
visual_enrich_openai.py
-----------------------
Enrich texture-metadata CSV by generating descriptive prompts with OpenAI
Vision-capable models (e.g. gpt-4o-mini).   Resumes automatically and
processes rows in parallel threads.
"""

import os, csv, time, base64, io, concurrent.futures as cf
from pathlib import Path

import openai           # pip install openai>=1.14
from PIL import Image    # pip install pillow
from tqdm import tqdm    # pip install tqdm
from dotenv import load_dotenv   # loads .env if present

# --------------------------------------------------
# ---------  configuration -------------------------
# --------------------------------------------------
BASE_DIR      = Path(r"C:/Users/Avneet Singh/OneDrive/Documents/Forward/Bath/Academics/DIssertation/Implementation/dataset_culture_textures")
CSV_INPUT     = BASE_DIR / "all_metadata.csv"
CSV_OUTPUT    = BASE_DIR / "all_metadata_enriched.csv"

OPENAI_MODEL  = "gpt-4o-mini"         # put the model ID you have access to
TEMPERATURE   = 0.4
NUM_WORKERS   = 4                     # threads: increase if network/GPU is fast
MAX_RETRIES   = 3
TIMEOUT_S     = 60                    # HTTP timeout per request

# --------------------------------------------------
load_dotenv()  # so OPENAI_API_KEY can come from .env
openai_client = openai.OpenAI(timeout=TIMEOUT_S)

# --------------------------------------------------
def b64_from_image(path: Path) -> str | None:
    """Return base-64 JPEG string for an image on disk (or None on failure)."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[encode] {path.name}: {exc}")
        return None


def build_system_prompt(short_label: str) -> str:
    return (
        "You are a culturally-aware texture analyst. "
        "Describe the material, cultural origin, pattern and overall aesthetic so the result is "
        "directly usable as a Stable-Diffusion prompt. Start with a brief identifier phrase, then "
        "1-2 richly descriptive sentences. Avoid brand names. "
        f"Original short label: {short_label!r}."
    )


def call_openai_vision(image_b64: str, short_label: str) -> str:
    """
    One API call with small retry loop.  Returns either the vision model's answer
    or '[ERROR] ...'.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": build_system_prompt(short_label)},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            }
                        ],
                    },
                ],
            )
            content = response.choices[0].message.content.strip()
            return content or "[ERROR] Empty response"
        except Exception as exc:
            if attempt == MAX_RETRIES:
                return f"[ERROR] {exc}"
            time.sleep(2 * attempt)  # simple back-off and retry


# --------------------------------------------------
def load_existing() -> dict[str, dict]:
    """
    Return a dict keyed by relative_path with *all* existing rows
    (including enriched_prompt if present) – empty dict if file missing.
    """
    if CSV_OUTPUT.exists():
        with CSV_OUTPUT.open(newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            return {row["relative_path"].replace("\\", "/"): row for row in rdr}
    return {}


def needs_processing(row: dict, existing_row: dict | None) -> bool:
    """
    Decide if the row still lacks a usable prompt and therefore
    needs to be sent to the API.
    """
    if existing_row is None:
        return True
    prompt = (existing_row.get("enriched_prompt") or "").strip()
    return (not prompt) or prompt.startswith("[ERROR]")


def worker(row: dict) -> dict:
    """
    Process a single CSV row (encode image + call model).
    Always returns a row dict with 'enriched_prompt'.
    """
    rel = row["relative_path"].replace("\\", "/")
    img_path = BASE_DIR / rel
    if not img_path.exists():
        row["enriched_prompt"] = "[ERROR] Image not found."
        return row

    img_b64 = b64_from_image(img_path)
    if not img_b64:
        row["enriched_prompt"] = "[ERROR] Encoding failed."
        return row

    row["enriched_prompt"] = call_openai_vision(img_b64, row["prompt"])
    return row


# --------------------------------------------------
def main() -> None:
    # ---------- read master CSV ------------
    if not CSV_INPUT.exists():
        raise FileNotFoundError(CSV_INPUT)

    with CSV_INPUT.open(newline="", encoding="utf-8") as inp:
        all_rows = list(csv.DictReader(inp))

    existing = load_existing()

    # ---------- filter rows to do ----------
    todo_rows = [
        row for row in all_rows
        if needs_processing(row, existing.get(row["relative_path"].replace("\\", "/")))
    ]
    print(f"🟢  {len(todo_rows)} rows need enrichment "
          f"(total {len(all_rows)}; already OK {len(all_rows)-len(todo_rows)})")

    if not todo_rows:
        print("✅  Nothing to do – all textures already have prompts.")
        return

    # ---------- parallel run ---------------
    completed: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
        futures = {exe.submit(worker, row): row for row in todo_rows}
        for fut in tqdm(cf.as_completed(futures), total=len(todo_rows), desc="🔎  Enriching"):
            new_row = fut.result()
            completed[new_row["relative_path"].replace("\\", "/")] = new_row

    # ---------- merge into final dict -------
    merged_rows = []
    for src_row in all_rows:
        key = src_row["relative_path"].replace("\\", "/")
        merged_rows.append(completed.get(key) or existing.get(key) or src_row)

    # ---------- write output ----------------
    fieldnames = list(all_rows[0].keys()) + ["enriched_prompt"]
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()
        for row in merged_rows:
            # ensure column exists
            row.setdefault("enriched_prompt", "")
            w.writerow(row)

    print(f"🏁  Done.  Updated CSV saved to: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
