# scripts/export_clip_full_onnx.py

import os, time
from pathlib import Path
import torch
from transformers import CLIPModel, CLIPTokenizer

# set your HF token if needed
TOKEN = "hf_"
assert len(TOKEN) > 10
os.environ["HUGGINGFACE_HUB_TOKEN"] = TOKEN

OUT = Path("onnx/clip_fp32")
OUT.mkdir(parents=True, exist_ok=True)

def export_full_clip():
    print("[EXPORT] Loading full CLIPModel…")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_auth_token=TOKEN).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_auth_token=TOKEN)

    # dummy inputs
    toks = tokenizer(
        "hello world",
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    )
    dummy_img = torch.randn(1, 3, 224, 224)

    onnx_path = OUT / "clip_full.onnx"
    print(f"[EXPORT] Exporting full CLIP to {onnx_path} …")
    t0 = time.time()
    torch.onnx.export(
        model,
        # positional args must match forward(input_ids, pixel_values, attention_mask)
        (toks.input_ids, dummy_img, toks.attention_mask),
        onnx_path,
        input_names=["input_ids", "pixel_values", "attention_mask"],
        output_names=["text_embeds", "image_embeds"],
        opset_version=16,
        dynamic_axes={
            "input_ids":      {0: "batch"},
            "pixel_values":   {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_embeds":    {0: "batch"},
            "image_embeds":   {0: "batch"},
        },
    )
    dt = time.time() - t0
    size_mb = onnx_path.stat().st_size / (1024*1024)
    print(f"[EXPORT] Done in {dt:.1f}s, file size {size_mb:.1f} MB")

if __name__ == "__main__":
    export_full_clip()
