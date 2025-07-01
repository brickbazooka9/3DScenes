# Implementation/prep_images.py
from pathlib import Path
from PIL import Image
res = 512

for img in Path("dataset_culture_textures").rglob("*.jpg"):
    out = Image.open(img).convert("RGB").resize((res, res), Image.LANCZOS)
    out.save(img)               # overwrite in-place
print("✅ resized to 512×512")
