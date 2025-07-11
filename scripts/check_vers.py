import torch, torchvision, diffusers, accelerate, peft, transformers
from diffusers.models.attention_processor import LoRAAttnProcessor
print("torch         :", torch.__version__)
print("torchvision   :", torchvision.__version__)
print("diffusers     :", diffusers.__version__)
print("accelerate    :", accelerate.__version__)
print("transformers  :", transformers.__version__)
print("peft          :", peft.__version__)
print("LoRAAttnProcessor.init sig ->", 
      LoRAAttnProcessor.__init__.__signature__)