import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

# --------------------------- CONFIG ---------------------------
cultures = ["Japanese", "Moroccan", "Indian", "Scandinavian"]
object_types = ["wall", "floor", "fabric", "furniture"]
base_path = "/workspace/dataset_culture_textures"
max_images = 250
delay = 3  # seconds between image downloads to avoid rate limiting
# --------------------------------------------------------------

def scrape_images(culture, object_type):
    query = f"{culture} {object_type} texture pattern interior"
    save_path = os.path.join(base_path, culture, object_type)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n🔍 Searching: '{query}'")
    with DDGS() as ddgs:
        results = list(ddgs.images(keywords=query, max_results=max_images))

    for i, result in enumerate(tqdm(results, desc=f"📥 {culture} {object_type}")):
        try:
            url = result["image"]
            response = requests.get(url, timeout=10)
            try:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                filename = f"{culture.lower()}_{object_type.lower()}_{i+1:03d}.jpg"
                filepath = os.path.join(save_path, filename)
                image.save(filepath)
                print(f"✅ Saved: {filepath}")
            except Exception as e:
                print(f"❌ Failed to save image {i+1}: {e}")
            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ Skipped image {i+1}: {e}")

# ------------------------ MAIN LOOP ------------------------

for culture in cultures:
    for obj in object_types:
        scrape_images(culture, obj)
