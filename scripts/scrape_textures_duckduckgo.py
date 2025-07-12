import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

# --------------------------- CONFIG ---------------------------
cultures = ["Japanese", "Indian", "Moroccan", "Scandinavian"]
object_types = ["wall", "floor", "fabric", "furniture"]
base_path = "/workspace/dataset_culture_textures"
max_images = 250

# ------------------------ SCRAPE ------------------------

for culture in cultures:
    for object_type in object_types:
        query = f"{culture} {object_type} texture pattern interior layout"
        save_path = os.path.join(base_path, culture, object_type)
        os.makedirs(save_path, exist_ok=True)

        print(f"\n🔍 Searching: '{query}'")
        with DDGS() as ddgs:
            results = list(ddgs.images(keywords=query, max_results=max_images))

        existing = set(os.listdir(save_path))
        downloaded = 0

        for i, result in enumerate(tqdm(results, desc=f"📥 {culture} {object_type}")):
            filename = f"{culture.lower()}_{object_type.lower()}_{i+1:03d}.jpg"
            filepath = os.path.join(save_path, filename)

            if filename in existing:
                continue  # Skip if already downloaded

            try:
                url = result["image"]
                response = requests.get(url, timeout=10)
                try:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image.save(filepath)
                    downloaded += 1
                    print(f"✅ Saved: {filepath}")
                except Exception as e:
                    print(f"❌ Failed to save image {i+1}: {e}")

                time.sleep(3)

            except Exception as e:
                print(f"⚠️ Skipped image {i+1}: {e}")

        print(f"✅ Finished: {culture} {object_type} ({downloaded} new images)")

