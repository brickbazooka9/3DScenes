# 3DScenes

# Culture-Enhanced 3D Scene Synthesis 🏺🛋️

> **MSc Dissertation • University of Bath**  
> *Author — Avneet Singh • 2024-25*

A research project that **injects cultural fidelity** into end-to-end 3-D indoor-scene generation.  
We combine **CommonScenes** (scene-graph → 3-D layout) with a **LoRA-fine-tuned Stable Diffusion** that produces **culturally-specific textures** for Japanese, Moroccan, Scandinavian and Indian interiors.

<p align="center">
  <img src="docs/teaser_pipeline.svg" width="780">
</p>

---

## Contents
| Section | Description |
|---------|-------------|
| [`dataset_culture_textures/`](dataset_culture_textures) | 1 041 curated texture images + `all_metadata_enriched.csv` (rich prompts) |
| [`scripts/`](scripts) | Python utilities for scraping, prompt enrichment, LoRA training |
| [`models/`](models) | Fine-tuned LoRA checkpoints (when available) |
| [`commonscenes/`](commonscenes) | Fork with minimal changes + integration hooks |
| [`notebooks/`](notebooks) | Exploratory data analysis, FID evaluation, smoke tests |
| [`docs/`](docs) | Figures, pipeline diagrams, paper drafts |

---

## 1 Quick Start

### 1.1 Clone

```bash
git clone https://github.com/<your-username>/culture-enhanced-3dscenes.git
cd culture-enhanced-3dscenes
```

## 1.2 Environment

conda env create -f environment.yml
conda activate culture3d
# or: pip install -r requirements.txt

## 2 Pipeline Overview

┌──────── 1. Scrape textures ─────┐
│ DuckDuckGo / Unsplash API       │→ JPG
└─────────────────────────────────┘
          │
┌──────── 2. Prompt enrichment (vision LLM) ────┐
│ local LLaVA via Ollama                        │→ all_metadata_enriched.csv
└───────────────────────────────────────────────┘
          │
┌──────── 3. LoRA fine-tune Stable Diffusion ───┐
│ rank=8 • res=512 • 12 k steps                 │→ textures_lora.safetensors
└───────────────────────────────────────────────┘
          │
┌──────── 4. Integrate with CommonScenes ───────┐
│ UV-map objects • txt2img • texture bake        │→ textured *.glb
└───────────────────────────────────────────────┘

## Progress:

## Project status

| Phase | Done? | Artefact |
|-------|-------|----------|
| 1. Data scraping            | ✔️ | `scrape_textures_duckduckgo.py`, JPEGs (local) |
| 2. Prompt enrichment (LLaVA) | ✔️ | `visual_enrich_llava.py`, `all_metadata_enriched.csv` |
| 3. LoRA texture fine-tune    | 🔜 | _scheduled July ’25_ |
| 4. CommonScenes integration  | 🔜 | _scheduled August ’25_ |
| 5. Evaluation & demo         | 🔜 | _scheduled August ’25_ |




