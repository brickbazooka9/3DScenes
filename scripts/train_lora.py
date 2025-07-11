#!/usr/bin/env python
"""
LoRA fine-tune of SD-2.1 textures (diffusers ≥ 0.30 + peft ≥ 0.9)
"""

import argparse, itertools
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline, DDPMScheduler

import csv
import matplotlib.pyplot as plt

# ─── Dataset ────────────────────────────────────────────────────────────────
class TextureDS(Dataset):
    tfm = T.Compose([
        T.Resize(512, T.InterpolationMode.BILINEAR),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    def __init__(self, files: List[Path]):
        self.recs = []
        for fp in files:
            rec = __import__("json").loads(fp.read_text(encoding="utf-8"))
            # must contain "path" & "prompt"
            self.recs.append(rec)
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r["path"]).convert("RGB")
        return {
            "pixel_values": self.tfm(img),
            "prompt":       r["prompt"],
        }
torch.set_float32_matmul_precision('medium')

def main():

    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU available")
        # ─── CLI ───────────────────────────────────────────────────────────────────
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_name", help="Name for experiment/checkpoint separation", default=None)
    parser.add_argument("--data_root",   required=True, help="root directory of textures_lora")
    parser.add_argument("--culture",     required=True, help="subfolder name, e.g. Japanese")
    parser.add_argument("--steps",  type=int, default=1200)
    parser.add_argument("--batch",  type=int, default=4)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--rank",   type=int, default=8)
    parser.add_argument("--alpha",   type=int, default=16)
    parser.add_argument("--dropout",type=float, default=0.1)
    parser.add_argument("--text_encoder", action="store_true",
                        help="also train LoRA on the text encoder")
    args = parser.parse_args()

    run_id = args.run_name or args.culture  # fallback to culture if run_name is not given

    checkpoint_dir = Path("checkpoints") / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_dir / "latest.pt"

    Path("loss_logs").mkdir(exist_ok=True)
    csv_path = f"loss_logs/loss_{args.culture}.csv"
    loss_log = []

    # ─── Find all JSON side-cars ─────────────────────────────────────────────────
    DATA_DIR = Path(args.data_root) / args.culture
    print(f"🔍  Scanning for JSONs under: {DATA_DIR.resolve()}")
    json_files: List[Path] = list(DATA_DIR.rglob("*.json"))
    print(f"✅  Found {len(json_files)} JSON files\n")
    if not json_files:
        print("❌  No JSON side-cars detected!  Run `build_texture_jsons.py --culture {args.culture}` first.")
        raise SystemExit(1)

    ds     = TextureDS(json_files)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=2, pin_memory=True)
    print(f"📚  {len(ds)} records · batch {args.batch}\n")

    # ─── Pipeline & LoRA adapters ───────────────────────────────────────────────
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", use_safetensors=True
    )
    pipe.safety_checker = None
    pipe.unet.enable_gradient_checkpointing()
    pipe.text_encoder.gradient_checkpointing_enable()
    unet, tokenizer, text_enc, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae

    # freeze the base models
    for m in (unet, text_enc, vae):
        m.requires_grad_(False)

    # UNet LoRA
    unet_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",  # adds stable init
    )
    unet = get_peft_model(unet, unet_config)
    pipe.unet = unet  # 🔥 Important to make sure pipe uses LoRA adapter

    # optional text-encoder LoRA
    if args.text_encoder:
        text_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            init_lora_weights="gaussian",
        )
        text_enc = get_peft_model(text_enc, text_config)
        pipe.text_encoder = text_enc  # 🔥 Ensures inference uses LoRA-modified encoder
        text_enc_ckpt = checkpoint_dir / "text_enc.pt"
        if text_enc_ckpt.exists():
            text_enc.load_state_dict(torch.load(text_enc_ckpt, map_location="cpu"))

    # collect trainable params
    params = [
        p for p in itertools.chain(unet.parameters(),
                                text_enc.parameters() if args.text_encoder else [])
        if p.requires_grad
    ]
    optim = torch.optim.AdamW(params, lr=args.lr)

    start_step = 0
    
    if ckpt_file.exists():
        print(f"🔁 Resuming training from checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        unet.load_state_dict(ckpt["unet"])
        optim.load_state_dict(ckpt["optimizer"])
        loss_log = ckpt["loss_log"]
        start_step = ckpt["step"] + 1
    else:
        loss_log = []

    sched = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
    )

    # ─── Accelerator setup ───────────────────────────────────────────────────────
    acc = Accelerator(mixed_precision="fp16")
    unet, text_enc, optim, loader = acc.prepare(unet, text_enc, optim, loader)
    vae.to(acc.device)

    # ─── Training loop ───────────────────────────────────────────────────────────
    pbar = tqdm(range(start_step,args.steps), disable=not acc.is_main_process, desc="steps")
    for step, batch in zip(pbar, itertools.islice(itertools.cycle(loader), args.steps)):
        with acc.accumulate(unet):
            imgs = batch["pixel_values"].to(acc.device)

            # ✅ Latent encoding
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215  # 🧠 Use `sample()` not `mode()`

            noise = torch.randn_like(latents)

            # Clamp latents and noise before applying
            latents = torch.clamp(latents, -5, 5)
            noise = torch.clamp(noise, -5, 5)

            ts = torch.randint(0, sched.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy = sched.add_noise(latents, noise, ts)
            noisy = torch.clamp(noisy, -5, 5)  # Final safety clamp

            if acc.is_main_process and step < 5:
                print(f"[DEBUG] latents stats → min: {latents.min().item():.4f}, max: {latents.max().item():.4f}")
                print(f"[DEBUG] noisy stats   → min: {noisy.min().item():.4f}, max: {noisy.max().item():.4f}")

            # ✅ Tokenize and encode
            tok = tokenizer(batch["prompt"],
                            padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length,
                            return_tensors="pt").to(acc.device)

            if args.text_encoder:
                enc = text_enc(**tok)[0]
            else:
                with torch.no_grad():
                    enc = text_enc(**tok)[0]

            # ✅ Predict and loss
            pred = unet(noisy, ts, encoder_hidden_states=enc).sample
            pred = torch.clamp(pred, -10, 10)  # ← More conservative clamp

            if acc.is_main_process and step < 5:
                print(f"[DEBUG] pred mean={pred.mean().item():.4f}, std={pred.std().item():.4f}")
                print(f"[DEBUG] noise mean={noise.mean().item():.4f}, std={noise.std().item():.4f}")
                print(f"[DEBUG] UNet output stats → min: {pred.min().item():.4f}, max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}")

            loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

            # ✅ NaN/Inf checks
            if not torch.isfinite(loss):
                print(f"⚠️ NaN detected at step {step} — skipping")
                continue


            if acc.is_main_process and step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.6f}")

            acc.backward(loss)

            if acc.is_main_process:
                loss_log.append((step, loss.item()))
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, loss.item()])


            # ✅ Check gradients
            for name, p in unet.named_parameters():
                if p.grad is not None:
                    grad_nan = torch.isnan(p.grad).any().item()
                    grad_max = p.grad.abs().max().item()
                    if grad_nan or grad_max > 1000:
                        print(f"⚠️ Grad alert at {name} | max={grad_max:.4f} | nan={grad_nan}")

            torch.nn.utils.clip_grad_norm_(params, max_norm=1)
            for p in params:
                if p.grad is not None:
                    p.grad.data = p.grad.data.clamp_(-1, 1)
            optim.step()
            if acc.is_main_process and (step + 1) % 50 == 0:
                torch.save({
                    "step": step,
                    "unet": unet.state_dict(),
                    "optimizer": optim.state_dict(),
                    "loss_log": loss_log,
                }, ckpt_file)

                if args.text_encoder:
                    torch.save(text_enc.state_dict(), checkpoint_dir / "text_enc.pt")

            optim.zero_grad()

        if acc.is_main_process and ((step+1) % 50 == 0):
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # In train_lora.py
    total = sum(p.numel() for p in unet.parameters())
    nonzero = sum((p != 0).sum().item() for p in unet.parameters())
    print(f"UNet non-zero weights after training: {nonzero}/{total}")
    lora_nonzero = sum((p != 0).sum().item() for n, p in unet.named_parameters() if "lora" in n)
    print(f"LoRA non-zero weights: {lora_nonzero}")

    # ─── Save LoRA weights ───────────────────────────────────────────────────────
    if acc.is_main_process:
        out = Path("lora-out") / run_id
        out.mkdir(parents=True, exist_ok=True)

        unet.save_pretrained(out / "unet")
        if args.text_encoder:
            text_enc.save_pretrained(out / "text_encoder")

        pipe.tokenizer.save_pretrained(out / "tokenizer")
        
        steps, losses = zip(*loss_log)
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.savefig(f"loss_logs/loss_curve_{args.culture}.png")
        plt.close()

        print("🏁  finished — adapters saved to", out)
        print(f"[DEBUG] loss={loss.item():.4f}, model_pred stats → min: {pred.min().item():.4f}, max: {pred.max().item():.4f}")
        print(f"[DEBUG] noise stats → min: {noise.min().item():.4f}, max: {noise.max().item():.4f}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
