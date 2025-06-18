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
