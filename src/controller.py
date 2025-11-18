from typing import List, Optional, Tuple, Union

from .losses import (
    controller_loss, 
    conform_loss, 
    attend_and_excite_loss,
    divide_and_bind_loss, 
    jedi_loss,
    self_cross_guidance_loss,
    enmmdt_loss,
)

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

class Controller:
    def __init__(
        self,
        model: str = "SD3",
        heuristic: str = "focus",
        lambda_scale: float = 1.0,
        t5_ids: List[List[int]] = [[[]]],
        clip_ids: List[List[int]] = [[[]]],
        attn_res: int=16
    ):  
        # Default to empty lists if no tokens are provided
        if len(t5_ids) == 0:
            t5_ids = [[[]]]
        if len(clip_ids) == 0:
            clip_ids = [[[]]]
        
        if isinstance(t5_ids[0], int) or isinstance(clip_ids[0], int):
            raise ValueError("t5_ids and clip_ids should be a list (BATCH) of lists (SUBJECTS) of lists (IDs) of ints")

        if isinstance(t5_ids[0][0], int):
            t5_ids = [t5_ids]
        if isinstance(clip_ids[0][0], int):
            clip_ids = [clip_ids]

        assert len(t5_ids) == len(clip_ids), "Mismatched number of subjects between T5 and CLIP"
        batch_size = len(t5_ids)
        
        # Initialize activation state
        self.activated = False

        # Store hyperparameters
        self.lambda_scale = lambda_scale
        self.heuristic = heuristic
        self.model = model
        self.attn_res = attn_res # For unet models with different attention map resolutions

        # Process token IDs and initialize storage
        self.storage = []
        self.token_groups = []
        self.label_groups = []
        
        batch_size = len(t5_ids)
        for idx in range(batch_size):
            token_group, label_group = self.process_ids(t5_ids[idx], clip_ids[idx])
            
            self.storage.append([])
            self.token_groups.append(token_group)
            self.label_groups.append(label_group)

        self.self_storage = [[] for _ in range(len(self.storage))]
        self.previous_storage = [[] for _ in range(len(self.storage))]

    def process_ids(self, t5_ids, clip_ids):
        if self.model == "SD3":
            # Offset T5 token IDs by 77 (first 77 tokens are for CLIP)
            t5_ids = [[id + 77 for id in group] for group in t5_ids]
        elif self.model == "FLUX":
            # CLIP IDs are not used in FLUX
            clip_ids = []
        elif self.model == "SD1":
            # T5 IDs are not used in SD1
            t5_ids = []
        else:
            raise ValueError(f"Unknown model: {self.model}")

        # Process T5 and CLIP tokens into a single list of subject groups
        glob_counter = 0
        token_groups, label_groups = [], []
        for idx in range(max(len(t5_ids), len(clip_ids))):
            local_counter = 0

            if idx < len(t5_ids):
                token_groups.extend(t5_ids[idx])
                local_counter += len(t5_ids[idx])

            if idx < len(clip_ids):
                token_groups.extend(clip_ids[idx])
                local_counter += len(clip_ids[idx])

            label_groups.append(
                torch.arange(glob_counter, glob_counter + local_counter)
            )
            glob_counter += local_counter

        token_groups = torch.tensor(token_groups)
        return token_groups, label_groups
     
    def update_ids(self, t5_ids: List[List[List[int]]], clip_ids: List[List[List[int]]]):
        assert len(t5_ids) == len(clip_ids), "Mismatched number of subjects between T5 and CLIP"
        
        self.storage = []
        self.token_groups = []
        self.label_groups = []

        batch_size = len(t5_ids)
        for idx in range(batch_size):
            token_group, label_group = self.process_ids(t5_ids[idx], clip_ids[idx])
            
            self.storage.append([])
            self.token_groups.append(token_group)
            self.label_groups.append(label_group)
        
        self.self_storage = [[] for _ in range(len(self.storage))]
        self.previous_storage = [[] for _ in range(len(self.storage))]

    def is_active(self) -> bool:
        """Check if storage is active."""
        return self.activated

    def activate_storage(self):
        """Activate storage for cross-attention."""
        self.activated = True

    def deactivate_storage(self):
        """Deactivate storage for cross-attention."""
        self.activated = False

    def reset_storage(self):
        """Reset the stored cross-attention tensors."""
        self.storage = [[] for _ in range(len(self.storage))]
        self.self_storage = [[] for _ in range(len(self.self_storage))]

    def save_self_attention(self, self_attn: torch.Tensor):
        """Save image self-attention matrices [B, L_img, L_img]."""
        if self.activated and self.heuristic == "self_cross_guidance":
            for idx in range(len(self.self_storage)):
                self.self_storage[idx].append(self_attn[idx])

    def return_self_storage(self, idx: int) -> torch.Tensor:
        """Average over blocks → [L_img, L_img]."""
        S = torch.stack(self.self_storage[idx]).to(torch.float32)   # [num_blocks, L_img, L_img]
        return S.mean(dim=0)

    def save_cross_attention(self, attn: torch.Tensor):
        """Save cross-attention maps for selected tokens."""
        if self.activated:
            if self.model == "SD1":
                dim = int(attn.shape[-1]**0.5)
                attn = F.interpolate(
                    attn.view(-1, 1, dim, dim), 
                    size=(self.attn_res, self.attn_res), 
                    mode='bilinear', 
                    align_corners=False
                ).view(attn.shape[0], attn.shape[1], -1)
            
            for idx in range(len(self.storage)):
                self.storage[idx].append(attn[idx, self.token_groups[idx]])

    def return_stroage_per_block(self, idx: int) -> List[torch.Tensor]:
        return torch.stack(self.storage[idx]).to(torch.float32)

    def return_storage(self, idx: int) -> torch.Tensor:
        """Retrieve stored cross-attention maps."""
        attn = torch.stack(self.storage[idx]).to(torch.float32)
        emb = attn.mean(dim=0)  # Average over blocks
        
        dim = int(emb.shape[-1]**0.5)
        emb = gaussian_blur(
            emb.view(-1, 1, dim, dim), 
            kernel_size=3, 
            sigma=1.0,
        ).view(emb.shape)

        return emb
    
    def compute_cost(self) -> torch.Tensor:
        loss_ls = []

        storage_for_later = []
        for idx in range(len(self.storage)):
            emb = self.return_storage(idx)

            if self.heuristic == "jedi":
                # JEDI loss requires per-block embeddings
                emb = self.return_stroage_per_block(idx)
                loss_ls.append(jedi_loss(emb, self.label_groups[idx]))

            elif self.heuristic == "conform":
                loss_ls.append(conform_loss(emb, self.previous_storage[idx], self.label_groups[idx]))

            elif self.heuristic == "focus":
                loss_ls.append(controller_loss(emb, self.label_groups[idx]))
            
            elif self.heuristic == "attend_and_excite":
                loss_ls.append(attend_and_excite_loss(emb, self.label_groups[idx]))

            elif self.heuristic == "divide_and_bind":
                loss_ls.append(divide_and_bind_loss(emb, self.label_groups[idx]))

            elif self.heuristic == "self_cross_guidance":
                selfA  = self.return_self_storage(idx)
                loss_ls.append(self_cross_guidance_loss(
                    emb, 
                    self.label_groups[idx], 
                    self_attn=selfA, 
                    lambda_cross=0.0
                ))

            elif self.heuristic == "enmmdt":
                loss_ls.append(enmmdt_loss(emb, self.label_groups[idx]))

            else:
                raise ValueError(f"Unknown heuristic: {self.heuristic}")
            
            if self.heuristic == "conform":
                storage_for_later.append(emb.detach().clone())
        
        # CONFORM requires the previous storage to compute the loss
        if self.heuristic == "conform":
            self.previous_storage = storage_for_later

        return torch.stack(loss_ls)