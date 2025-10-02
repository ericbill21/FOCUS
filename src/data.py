from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import yaml
from tqdm.notebook import tqdm
from diffusers import StableDiffusion3Pipeline

import torch
from torch.utils.data import Dataset, DataLoader

class PromptDataset(Dataset):
    def __init__(
        self,
        prompt_emb: Sequence[Any]=None,
        pooled_emb: Sequence[Any]=None,
        t5_indicies: Sequence[Any]=None,
        clip_indices: Sequence[Any]=None,
        path: Optional[str]=None,
    ):
        if path is not None:
            data = torch.load(path)
            prompt_emb = data["prompt_emb"]
            pooled_emb = data["pooled_emb"]
            t5_indicies = data["t5"]
            clip_indices = data["clip"]
        
        if prompt_emb is None or pooled_emb is None or t5_indicies is None or clip_indices is None:
            raise ValueError("Either path or all data arguments must be provided.")

        lens = {len(prompt_emb), len(pooled_emb), len(t5_indicies), len(clip_indices)}
        if len(lens) != 1:
            raise ValueError(f"All lists must be the same length, got lengths={lens}.")

        self._data = (prompt_emb, pooled_emb, t5_indicies, clip_indices)
        
    def __len__(self) -> int:
        return len(self._data[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return (
            self._data[0][idx],
            self._data[1][idx],
            self._data[2][idx],
            self._data[3][idx],
        )

    def collate_fn(batch):
        # batch: list of tuples (prompt_emb, pooled_emb, t5, clip)
        prompt_embs, pooled_embs, t5_list, clip_list = zip(*batch)

        return {
            "prompt_emb": torch.cat(prompt_embs, dim=0),
            "pooled_emb": torch.cat(pooled_embs, dim=0),
            "t5": list(t5_list),
            "clip": list(clip_list),
        }

@torch.no_grad()
def encode_prompts(path):
    dataset = yaml.safe_load(open(path, "r"))

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    prompt_ls, pooled_ls, t5_ls, clip_ls = [], [], [], []

    for data in tqdm(dataset):
        prompt_emb, _, pooled_emb, _ = pipe.encode_prompt(
            prompt = data["prompt"],
            prompt_2 = data["prompt"],
            prompt_3 = data["prompt"],
            do_classifier_free_guidance=False,
            max_sequence_length=77,
        )

        prompt_ls.append(prompt_emb.cpu())
        pooled_ls.append(pooled_emb.cpu())
        t5_ls.append(data["t5"])
        clip_ls.append(data["clip"])

    torch.save({
        "prompt_emb": prompt_ls,
        "pooled_emb": pooled_ls,
        "t5": t5_ls,
        "clip": clip_ls,
    }, "embeddings.pt")