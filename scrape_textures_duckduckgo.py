import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

# --------------------------- CONFIG ---------------------------
# Set culture and object type to scrape
culture = "Scandinavian"         # e.g. Japanese, Moroccan, Indian, Scandinavian
object_type = "wall"        # e.g. wall, floor, fabric, furniture

# Set query
query = f"{culture} {object_type} texture pattern interior"

# Output folder
base_path = "/workspace/dataset_culture_textures"


save_path = os.path.join(base_path, culture, object_type)
os.makedirs(save_path, exist_ok=True)

# Max results to fetch
max_images = 250

# ------------------------ SEARCH & DOWNLOAD ------------------------

with DDGS() as ddgs:
    print(f"🔍 Searching: '{query}'")
    results = list(ddgs.images(keywords=query, max_results=max_images))

for i, result in enumerate(tqdm(results, desc=f"📥 Downloading {culture} {object_type} textures")):
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

        time.sleep(3)  # Be gentler to avoid rate limiting

    except Exception as e:
        print(f"⚠️ Skipped image {i+1}: {e}")
