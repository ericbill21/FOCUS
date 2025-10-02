from __future__ import annotations
from typing import List, Dict, Any, Optional
import argparse
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Transformers / models
import ImageReward as RM 
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizer
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import Qwen2VLForConditionalGeneration,  SiglipModel

# Text similarity
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False
    SentenceTransformer = None


def batched(iterable, n: int):
    """Yield lists of length n (last batch possibly smaller)."""
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine between two equal-length batches (N, D)."""
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a * b).sum(dim=-1)

# -------------------------------
# IMAGE-TEXT SIMILARITY
# -------------------------------

# BASIC OPEN AI CLIP
class CLIPScorer:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        # Process images and texts separately to use get_*_features()
        img_inputs = self.processor(images=images, return_tensors="pt")
        txt_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)

        img_feats = self.model.get_image_features(pixel_values=img_inputs["pixel_values"].to(self.device))
        txt_feats = self.model.get_text_features(
            input_ids=txt_inputs["input_ids"].to(self.device),
            attention_mask=txt_inputs["attention_mask"].to(self.device)
        )
        sims = cosine_sim(img_feats, txt_feats).detach().cpu().tolist()
        return sims

# SigLIP-2
class SigLIP2Scorer:
    """
    I–T similarity using SigLIP-2.
    Default checkpoint: google/siglip2-so400m-patch14-384 (good trade-off).
    """
    def __init__(self, device, model_id: str = "google/siglip2-so400m-patch14-384"):
        self.device = device
        self.model = SiglipModel.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if device.type == "cuda" else None),
        ).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        # Pack as a joint batch; logits_per_image is (N_images x N_texts)
        inputs = self.proc(
            images=images, text=texts,
            padding="max_length", 
            max_length=64,
            return_tensors="pt"
        ).to(self.model.device)

        out = self.model(**inputs)
        # Prefer cosine from normalized embeddings if available
        if hasattr(out, "image_embeds") and hasattr(out, "text_embeds"):
            img = out.image_embeds
            txt = out.text_embeds
            sims = cosine_sim(img, txt)
        else:
            # Fallback: take diagonal of similarity logits (scaled)
            sims = torch.diag(out.logits_per_image)
        return sims.detach().float().cpu().tolist()

# Larger CLIP (EVA-CLIP)
class EVAClipScorer:
    """
    I–T similarity using EVA-CLIP (e.g., BAAI/EVA-CLIP-8B).
    Smaller/cheaper alternatives from the EVA family also work.
    """
    def __init__(self, device, model_id: str = "BAAI/EVA-CLIP-8B", image_size: int = 224):
        self.device = device
        # EVA-CLIP uses its own code; trust_remote_code is required
        self.model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device).eval()
        # Use a CLIP-style image processor & tokenizer as recommended by the model card
        self.img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tok = CLIPTokenizer.from_pretrained(model_id)  # provided in the repo

    @torch.no_grad()
    def score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        pix = self.img_proc(images=images, return_tensors="pt").pixel_values.to(self.model.device)
        toks = self.tok(texts, return_tensors="pt", padding=True).input_ids.to(self.model.device)

        img_feats = self.model.encode_image(pix)
        txt_feats = self.model.encode_text(toks)
        sims = cosine_sim(img_feats, txt_feats)
        return sims.detach().float().cpu().tolist()


# -------------------------------
# TEXT-TEXT SIMILARITY
# -------------------------------
class BLIPTT:
    """
    Generates an image caption with BLIP, then computes semantic similarity
    between the original prompt and the generated caption using sentence-transformers.
    """
    def __init__(self, device: torch.device, st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = device
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device).eval()
        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        self._st_model = None
        if _HAS_ST:
            try:
                self._st_model = SentenceTransformer(st_model_name, device=device)
            except Exception as e:
                print(f"[WARN] Could not load SentenceTransformer '{st_model_name}': {e}")
        if self._st_model is None:
            print("[WARN] sentence-transformers not available. Falling back to CLIP text encoder for T–T.\n"
                  "      For more faithful text-text similarity, install 'sentence-transformers'.")

        # If falling back, reuse CLIP text encoder
        if self._st_model is None:
            self.clip_text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_text_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def captions(self, images: List[Image.Image], max_new_tokens: int = 30, num_beams: int = 3) -> List[str]:
        caps = []
        for img in images:
            inputs = self.blip_proc(images=img, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
            cap = self.blip_proc.decode(out[0], skip_special_tokens=True)
            caps.append(cap.strip())
        return caps

    @torch.no_grad()
    def text_emb(self, texts: List[str]) -> torch.Tensor:
        if self._st_model is not None:
            emb = self._st_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            return emb
        # Fallback to CLIP text features
        enc = self.clip_text_proc(text=texts, return_tensors="pt", padding=True, truncation=True)
        feats = self.clip_text_model.get_text_features(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device)
        )
        return feats

    @torch.no_grad()
    def score(self, prompts: List[str], captions: List[str]) -> List[float]:
        t1 = self.text_emb(prompts)
        t2 = self.text_emb(captions)
        sims = cosine_sim(t1, t2).detach().cpu().tolist()
        return sims
    
class Qwen2VLTT:
    """
    Generates captions with Qwen2-VL Instruct.
    Default ckpt: Qwen/Qwen2-VL-2B-Instruct  (use 7B for higher quality)
    """
    def __init__(self, 
        device, 
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        prompt: str = "Describe this image by listing the objects and the scenary, fluent English.",
        max_new_tokens: int = 96,
        st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto"
        ).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

        self._st_model = None
        if _HAS_ST:
            try:
                self._st_model = SentenceTransformer(st_model_name, device=device)
            except Exception as e:
                print(f"[WARN] Could not load SentenceTransformer '{st_model_name}': {e}")
        if self._st_model is None:
            print("[WARN] sentence-transformers not available. Falling back to CLIP text encoder for T–T.\n"
                  "      For more faithful text-text similarity, install 'sentence-transformers'.")

        # If falling back, reuse CLIP text encoder
        if self._st_model is None:
            self.clip_text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_text_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def captions(self, images: List[Image.Image]) -> List[str]:
        outs = []
        for img in images:
            # Build single-turn chat with one image + one text prompt
            conversation = [{"role": "user",
                             "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
            text = self.proc.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.proc(text=[text], images=[img], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            # Trim the prompt tokens, then decode
            trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
            caption = self.proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            outs.append(caption.strip())
        return outs
    
    @torch.no_grad()
    def text_emb(self, texts: List[str]) -> torch.Tensor:
        if self._st_model is not None:
            emb = self._st_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            return emb
        
        # Fallback to CLIP text features
        enc = self.clip_text_proc(text=texts, return_tensors="pt", padding=True, truncation=True)
        feats = self.clip_text_model.get_text_features(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device)
        )
        return feats

    @torch.no_grad()
    def score(self, prompts: List[str], captions: List[str]) -> List[float]:
        t1 = self.text_emb(prompts)
        t2 = self.text_emb(captions)
        sims = cosine_sim(t1, t2).detach().cpu().tolist()
        return sims

# -------------------------------
# PREFERENCE SCORES
# -------------------------------
class PickScoreScorer:
    def __init__(self, device,
        proc_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_id: str = "yuvalkirstain/PickScore_v1"
    ):
        self.device = device
        self.proc = AutoProcessor.from_pretrained(proc_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        # Batched scoring; PickScore expects one prompt vs N images;
        # we do 1:1 pairs, so compute per-item logits.
        out_scores = []
        for img, txt in zip(images, texts):
            img_inputs = self.proc(images=img, return_tensors="pt").to(self.model.device)
            txt_inputs = self.proc(text=txt,  return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.model.device)
            img_emb = self.model.get_image_features(**img_inputs)
            txt_emb = self.model.get_text_features(**txt_inputs)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
            txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
            # Same scoring as the official example (logit_scale * cosine)
            score = (self.model.logit_scale.exp() * (txt_emb @ img_emb.T))[0, 0]
            out_scores.append(float(score.detach().cpu()))
        return out_scores


class ImageRewardScorer:
    """
    ImageReward (NeurIPS'23) human-preference reward.
    """
    def __init__(self, device):
        self.model = RM.load("ImageReward-v1.0")

    @torch.no_grad()
    def score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        # The model can take a PIL image or a path.
        scores = []
        for img, txt in zip(images, texts):
            s = self.model.score(txt, img)
            scores.append(float(s))
        return scores


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init scorers
    print("Loading scorers...")
    scorers = [
        ("OpenAI0-CLIP_I–T", CLIPScorer),
        ("Google-SigLIP-2_I–T", SigLIP2Scorer),
        # ("EVA-CLIP_I–T", EVAClipScorer(device)),
        ("BLIP_T–T", BLIPTT),
        ("Qwen2VL_T-T", Qwen2VLTT),
        ("PickScore_I–T", PickScoreScorer),
        ("ImageReward_I–T", ImageRewardScorer),
    ]
    
    print("Collecting all image paths and prompts ...")
    results = {"image_path": [], "prompt": []}
    df = pd.read_csv(os.path.join(args.data_dir, "prompts.csv"))
    for ip, pr in zip(df["image_path"], df["prompt"]):
        try:
            Image.open(ip)
        except Exception as e:
            print(f"[WARN] Could not load image '{ip}': {e}")

        results["image_path"].append(ip)
        results["prompt"].append(pr)
 
    print("Scoring...")
    for name, score_init in tqdm(scorers):
        results[name] = []
        scorer = score_init(device)

        is_tt = False
        if hasattr(scorer, "captions"):
            results[f"{name}_caption"] = []
            is_tt = True

        for start in tqdm(range(0, len(results["image_path"]), args.batch_size)):
            tail = min(start + args.batch_size, len(results["image_path"]))

            images = [Image.open(results["image_path"][idx]) for idx in range(start, tail)]
            prompts = results["prompt"][start:tail]

            try:
                if is_tt:
                    caps = scorer.captions(images)
                    results[name] += scorer.score(prompts, caps)
                    results[f"{name}_caption"] += caps

                else:
                    results[name] += scorer.score(images, prompts)
                        
            except Exception as e:
                print(f"[WARN] {name} scoring failed on batch: {e}")
                results[name] += [None] * len(images)
                if is_tt:
                    results[f"{name}_caption"] += [None] * len(images)

        # Free up memory
        del scorer
        torch.cuda.empty_cache()

    out_path = os.path.join(args.data_dir, "metrics.csv")
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"Saving metrics to {out_path} ...")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute automatic metrics for text-to-image results.")
    parser.add_argument("--data-dir", required=True, type=str, help="CSV with at least image and prompt columns.")
    parser.add_argument("--batch-size", default=16, type=int)

    args = parser.parse_args()
    main(args)