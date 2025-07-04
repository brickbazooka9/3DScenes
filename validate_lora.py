import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import pandas as pd

# ---------- CONFIG ----------
cultures = ["Japanese"]
base_model_id = "stabilityai/stable-diffusion-2-1-base"
lora_dir = "out/lora-out"
output_dir = "eval_outputs"
prompt_template = "A {culture} temple with traditional red wood arches and curved tiled roofs under cherry blossom trees"
num_images = 5
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

os.makedirs(output_dir, exist_ok=True)
results = []

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable Params: {trainable:,} / {total:,} ({100 * trainable/total:.4f}%)")

def generate_and_score(pipe, culture, mode):
    prompt = prompt_template.format(culture=culture)
    for i in tqdm(range(num_images), desc=f"{culture} | {mode}"):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        image = pipe(prompt, generator=generator).images[0]

        img_path = os.path.join(output_dir, f"{culture}_{mode}_{i}.png")
        image.save(img_path)

        # CLIP score
        inputs = clip_processor(text=prompt, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            score = outputs.logits_per_image[0][0].item()

        print(f"[{culture} | {mode} | {i}] CLIP raw score: {score:.4f}")

        results.append({
            "culture": culture,
            "mode": mode,
            "image_path": img_path,
            "clip_score": round(score, 4)
        })

for culture in cultures:
    print(f"\n🔍 Processing: {culture}")

    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16
    ).to(device)

    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    # BASELINE
    generate_and_score(pipe, culture, mode="base")

    # Load LoRA adapters if available
    lora_unet_path = os.path.join(lora_dir, culture, "unet")
    lora_text_path = os.path.join(lora_dir, culture, "text_encoder")

    if os.path.exists(lora_unet_path):
        print(f"🔁 Loading LoRA for {culture}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_unet_path)
        pipe.unet.set_adapter("default")
        print("✅ UNET LoRA loaded & set")
        print_trainable_params(pipe.unet)

        if os.path.exists(lora_text_path):
            pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, lora_text_path)
            pipe.text_encoder.set_adapter("default")
            print("✅ Text Encoder LoRA loaded & set")
            print_trainable_params(pipe.text_encoder)

        pipe.unet.eval()
        pipe.text_encoder.eval()

        # Generate with LoRA
        generate_and_score(pipe, culture, mode="lora")
    else:
        print(f"⚠️ No LoRA adapter found for {culture} — skipping LoRA generation.")

# Export results to CSV
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "clip_scores.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to {csv_path}")
