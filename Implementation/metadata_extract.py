import os
import csv

# Base folder where images are stored
base_dir = "C:/Users/Avneet Singh/OneDrive/Documents/Forward/Bath/Academics/DIssertation/Implementation/dataset_culture_textures"  # <-- Change this to your actual path

# Output CSV file
csv_file = os.path.join(base_dir, "all_metadata.csv")

# Columns for metadata
fields = ["filename", "culture", "object_type", "relative_path", "prompt"]

# Collect all records
records = []

for culture in os.listdir(base_dir):
    culture_path = os.path.join(base_dir, culture)
    if not os.path.isdir(culture_path):
        continue
    for object_type in os.listdir(culture_path):
        object_path = os.path.join(culture_path, object_type)
        if not os.path.isdir(object_path):
            continue
        for file in os.listdir(object_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.join(culture, object_type, file)
                prompt = f"{culture} style {object_type} texture"
                records.append([file, culture, object_type, rel_path, prompt])

# Write to CSV
with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(records)

print(f"✅ Metadata saved to: {csv_file}")
