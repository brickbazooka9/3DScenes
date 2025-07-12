import os
import requests
from ddgs import DDGS
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

# --------------------------- CONFIG ---------------------------
cultures = ["Japanese", "Indian", "Moroccan", "Scandinavian"]
object_types = ["wall", "floor", "fabric", "furniture"]

# Add variations to avoid DuckDuckGo 100-image cap
query_variations = [
    "texture pattern interior layout",
    "seamless interior design close-up",
    "pattern background texture",
]

# Output base path
base_path = "/workspace/dataset_culture_textures"
os.makedirs(base_path, exist_ok=True)

max_images = 100  # DuckDuckGo hard caps per query
sleep_time = 3

# ------------------------ MAIN LOOP ------------------------
for culture in cultures:
    for obj_type in object_types:
        save_path = os.path.join(base_path, culture, obj_type)
        os.makedirs(save_path, exist_ok=True)
        img_counter = len(os.listdir(save_path))  # continue from existing count

        for variation in query_variations:
            if img_counter >= 250:
                print(f"🛑 Skipping {culture} {obj_type}: Already has {img_counter} images")
                break

            query = f"{culture} {obj_type} {variation}"
            print(f"\n🔍 Searching: '{query}'")

            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(keywords=query, max_results=max_images))
            except Exception as e:
                print(f"❌ Failed search for {query}: {e}")
                continue

            for i, result in enumerate(tqdm(results, desc=f"📥 {culture} {obj_type}")):
                if img_counter >= 250:
                    break

                try:
                    url = result["image"]
                    response = requests.get(url, timeout=10)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    existing = os.listdir(save_path)
                    next_index = (
                        max([int(f.split('_')[-1].split('.')[0]) for f in existing if f.endswith('.jpg')] or [0]) + 1
                    )
                    filename = f"{culture.lower()}_{object_type.lower()}_{next_index:03d}.jpg"
                    filepath = os.path.join(save_path, filename)

                    if not os.path.exists(filepath):
                        image.save(filepath)
                        print(f"✅ Saved: {filepath}")
                        img_counter += 1
                    else:
                        print(f"⏩ Skipped existing file: {filename}")

                    time.sleep(sleep_time)

                except Exception as e:
                    print(f"⚠️ Skipped image {i+1}: {e}")
