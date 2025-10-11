"""
python finetune_flux.py --res-dir "/home/jupyter/out_FLUX" \
    --dataset embeddings/flux_horse_bear.pt \
    --num-traj 2 --sub-extra 16 --ckpt-every 25 \
    --image-size 256 --lambda-value 1 --num-iterations 400 --learning-rate 5e-4
"""

from typing import Optional

import os
import yaml
import logging
from glob import glob

import torch
from torch import Tensor

from diffusers.models import FluxTransformer2DModel

from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import numpy as np

from src.flux.attention_processor import FluxAttnProcessor
from src.controller import Controller
from src.flux.finetune_utils import MemorylessFlowMatchScheduler, MemorylessSDESolver, LeanAdjoinSolver
from src.data import PromptDataset
from src.utils import append_dims, sample_time_indices


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def create_logger(logging_dir: Optional[str]=None, verbose: int=1):
    verbose_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    handlers = [logging.StreamHandler()]
    if logging_dir is not None:
        handlers.append(logging.FileHandler(os.path.join(logging_dir, "log.txt")))

    logging.basicConfig(
        level=verbose_map.get(verbose, logging.INFO),
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    return logger


def setup_experiment(results_dir: os.PathLike):
    """Create an experiment directory for the current run."""

    # Make results directory
    os.makedirs(results_dir, exist_ok=True)
    experiment_dir = results_dir
    checkpoint_dir = os.path.join(results_dir, "checkpoints")

    # Make experiment directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir


def g(x: Tensor) -> Tensor:
    """ Terminal cost function, which we fix to zero always Needs to be differentiable."""
    return x.mean(dim=[1, 2]) * 0.


def sample_x0(batch_size, size, generator, device, dtype):
    # Sample initial latents
    latents = torch.randn(
        (batch_size, 16, size, size), 
        generator=generator, 
        device=device, 
        dtype=dtype
    )

    # pack latents following FLUX convention
    latents = latents.view(batch_size, 16, size//2, 2, size//2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (size//2) * (size//2), 16 * 4)

    return latents


def main(args):
    # Setup devices and data types
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Setup experiment directory
    exp_dir = setup_experiment(args.res_dir)
    logger = create_logger(exp_dir, verbose=args.verbose)
    logger.info(f"experiment directory created at {exp_dir}")

    # Save the configuration
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    
    scheduler = MemorylessFlowMatchScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="scheduler",
        torch_dtype=dtype,
    )

    # Prepare the base velocity and finetune velocity models
    transformer_base = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device)
    transformer_base.requires_grad_(False)
    transformer_base.eval()
    # transformer_base.enable_gradient_checkpointing()

    transformer_fine = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device)
    transformer_fine.requires_grad_(False)
    transformer_fine.eval()
    transformer_fine.enable_gradient_checkpointing()

    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", # latent image attention
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out" # prompt conditional attention
        ],
        bias="none",
    )
    transformer_fine = get_peft_model(transformer_fine, lora_config).to(dtype)
    # transformer_fine.train()

    # Printing info for both models
    num_train_params = lambda x: sum(p.numel() for p in x.parameters() if p.requires_grad)
    num_total_params = lambda x: sum(p.numel() for p in x.parameters())

    logger.info(f"Number of trainable parameters in base model: {num_train_params(transformer_base):,}" \
                f" out of {num_total_params(transformer_base):,} ->" \
                f" {num_train_params(transformer_base) / num_total_params(transformer_base) * 100:.2f}%")
    
    logger.info(f"Number of trainable parameters in finetune model: {num_train_params(transformer_fine):,}" \
                f" out of {num_total_params(transformer_fine):,} ->" \
                f" {num_train_params(transformer_fine) / num_total_params(transformer_fine) * 100:.2f}%")
    
    
    # Prepare solvers for forward denoising and backward adjoint equation
    sde_solver = MemorylessSDESolver(transformer_fine, scheduler)
    adj_solver = LeanAdjoinSolver(transformer_base, scheduler, g)

    # Install Controller module, as the Controller loss will be used as the running cost f(xt)
    controller = Controller(model="FLUX", heuristic=args.heuristic)

    for block in transformer_fine.transformer_blocks:   
        block.attn.set_processor(FluxAttnProcessor(controller))

    parameters = [p for n, p in transformer_fine.named_parameters() if "lora_" in n and p.requires_grad]
    optim = torch.optim.Adam([{
        "params" : parameters, 
        "weight_decay": 0.01,
        "lr": args.learning_rate,
        "betas": (0.95, 0.999),
        "eps": 1e-8,
    }])

    # Dataset with optimized data loading
    dataset = PromptDataset(path=args.dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=PromptDataset.collate_fn,
        drop_last=True,
        pin_memory=True,
        num_workers=4,  # Reduced to save memory
        prefetch_factor=2,  # Limit prefetching to save memory
        persistent_workers=True  # Keep workers alive to reduce overhead
    )

    # FLUX SPECIFIC CONSTANTS
    VAE_SCALE_FACTOR = 8
    SIZE = 2 * (int(args.image_size) // (VAE_SCALE_FACTOR * 2))

    latent_image_ids = torch.zeros(SIZE//2, SIZE//2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(SIZE//2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(SIZE//2)[None, :]
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    LATENT_IMAGE_IDS = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    ).to(device=device, dtype=dtype)

    GUIDANCE = torch.full((1,), 3.5, device=device, dtype=torch.float32)

    # Scheduler
    x0 = sample_x0(args.num_traj, SIZE, generator, device, dtype)
    sigmas = np.linspace(1.0, 1 / args.k, args.k)
    image_seq_len = x0.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)

    # Prepare everything with accelerator - optimize for memory and mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16"
    )
    transformer_fine, optim, loader = accelerator.prepare(transformer_fine, optim, loader)

    # Main training loop
    step = 0
    while step < args.num_iterations:
        for batch in loader:
            t5 = [batch["t5"][0] for _ in range(args.num_traj)]
            clip = [batch["clip"][0] for _ in range(args.num_traj)]
            controller.update_ids(t5, clip)

            prompt = batch["prompt_emb"].to(device=device, non_blocking=True)
            prompt = prompt.expand(args.num_traj, *prompt.shape[1:])
            
            pooled = batch["pooled_emb"].to(device=device, non_blocking=True)
            pooled = pooled.expand(args.num_traj, *pooled.shape[1:])

            TXT_IDS = torch.zeros(prompt.shape[1], 3).to(device=prompt.device, dtype=prompt.dtype)
            
            # 1.) Subsample time indices
            sub_idx = sample_time_indices(
                K=args.k-1, 
                start_inter=args.sub_start, 
                end_inter=args.sub_end, 
                n_rand=args.sub_extra, 
                generator=generator
            )

            # 2.) Forward SDE: Sample full trajectory [K+1, NUM_TRAJ, VAE_CHANNELS, LATENT_WIDTH, LATENT_HEIGHT]
            x0 = sample_x0(args.num_traj, SIZE, generator, device, dtype)
            scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)

            # 2.) Forward SDE: Sample trajectory
            x_traj, f_traj, cost = sde_solver.sample_memoryless_traj(
                x0=x0,
                prompt=prompt,
                pooled=pooled,
                num_steps=args.k,
                generator=generator,
                controller=controller,
                lambda_val=args.lambda_value,
                calc_cost_gradient=True,
                txt_ids=TXT_IDS,
                img_ids=LATENT_IMAGE_IDS,
                guidance=GUIDANCE.expand(x0.shape[0]),
            )

            # 3.) Backward ODE: Solve adjoint equation
            a_traj = adj_solver.sample(
                x_traj=x_traj,
                prompt=prompt,
                pooled=pooled,
                txt_ids=TXT_IDS,
                img_ids=LATENT_IMAGE_IDS,
                guidance=GUIDANCE.expand(x0.shape[0]),
                f_traj=f_traj
            )

            # Drop singularity points t=0 and t=1 and subsample
            x_traj = x_traj[1:-1][sub_idx]
            a_traj = a_traj[1:-1][sub_idx]
            t_traj = scheduler.timesteps[1:][sub_idx]

            # Keep everything on GPU to avoid expensive CPU transfers
            # Expand by time dimension [NUM_TRAJ, ...] -> [K, NUM_TRAJ, ...]
            exp_prompt = prompt.unsqueeze(0).expand(x_traj.shape[0], *prompt.shape)
            exp_pooled = pooled.unsqueeze(0).expand(x_traj.shape[0], *pooled.shape)
            t_traj = t_traj.unsqueeze(1).expand(-1, x_traj.shape[1])

            # Flatten time and batch dimensions [K, NUM_TRAJ, ...] -> [K * NUM_TRAJ, ...]
            x_traj = x_traj.reshape(-1, *x_traj.shape[2:])
            a_traj = a_traj.reshape(-1, *a_traj.shape[2:])
            t_traj = t_traj.reshape(-1, *t_traj.shape[2:])
            exp_prompt = exp_prompt.reshape(-1, *exp_prompt.shape[2:])
            exp_pooled = exp_pooled.reshape(-1, *exp_pooled.shape[2:])

            with accelerator.accumulate(transformer_fine):
                # Data is already on GPU, just prepare for model
                t_gpu = t_traj / 1000
                guidance_gpu = GUIDANCE.expand(x_traj.shape[0])

                with torch.no_grad():
                    vt_base = -1.0 * transformer_base(
                        hidden_states=x_traj,
                        timestep=t_gpu,
                        guidance=guidance_gpu,
                        pooled_projections=exp_pooled,
                        encoder_hidden_states=exp_prompt,
                        txt_ids=TXT_IDS,
                        img_ids=LATENT_IMAGE_IDS,
                    ).sample

                vt_fine = -1.0 * transformer_fine(
                    hidden_states=x_traj,
                    timestep=t_gpu,
                    guidance=guidance_gpu,
                    pooled_projections=exp_pooled,
                    encoder_hidden_states=exp_prompt,
                    txt_ids=TXT_IDS,
                    img_ids=LATENT_IMAGE_IDS,
                ).sample

                vt_diff = vt_fine - vt_base

                # 6.) Compute loss
                ts = scheduler.normalize_timesteps(t_traj)
                sigma = append_dims(scheduler.sigma(ts), x_traj.ndim)

                sigma_inv = 2.0 / sigma
                loss_term1 = sigma_inv * vt_diff
                loss_term2 = sigma * a_traj
                loss = (loss_term1 + loss_term2).square().mean()

                accelerator.backward(loss)

                # Clear intermediate tensors to free memory
                del t_gpu, guidance_gpu, vt_base, vt_fine, vt_diff, ts, sigma
                        
                # Use accelerator's step and zero_grad for better memory management
                optim.step()
                optim.zero_grad(set_to_none=True)

                # Clear cache periodically to prevent memory fragmentation
                if step % 10 == 0:
                    torch.cuda.empty_cache()

            # 8.) Log progress
            logger.info(f"Step {step+1:05d} / {args.num_iterations:05d}, Loss: {loss.item():.6f}, Cost: {cost.mean():.2f}±{cost.std():.2f}")
            step += 1

            # 9.) Save model checkpoint
            if (step + 1) % args.ckpt_every == 0:
                checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{step+1:05d}")
                logger.info(f"saving checkpoint to {checkpoint_path} at step {step + 1}...")
                transformer_fine.save_pretrained(checkpoint_path)
            
            if step >= args.num_iterations:
                break

    logger.info(f"Saving model to {exp_dir}")
    transformer_fine.save_pretrained(exp_dir + "/model")    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a flow matching model on CIFAR-10")

    # Experiment parameters
    parser.add_argument("--res-dir", type=str, required=True, help="Directory to save the results and checkpoints")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (parquet format)")

    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for logging (0: warning, 1: info, 2: debug)")
    parser.add_argument("--ckpt-every", type=int, default=100, help="Save a checkpoint every N iterations")

    # Standard training parameters
    parser.add_argument("--num-iterations", type=int, default=400, help="Number of training iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for the model (e.g., bfloat16, float32)")

    # Scheduler and solver parameters
    parser.add_argument("--num-traj", type=int, default=5, help="Number of trajectories to sample per iteration (reduced for speed)")
    parser.add_argument("--k", type=int, default=28, help="Number of time steps for the ODE solver (reduced for speed)")
    parser.add_argument("--lambda-value", type=float, default=1.0, help="Lambda value for the reward function")
    parser.add_argument("--heuristic", type=str, choices=["focus", "conform", "attend_and_excite", "divide_and_bind", "jedi"], default="focus", help="Controller heuristic to use")
    
    parser.add_argument("--sub-start", type=int, default=0, help="Start index for subsampling time steps")
    parser.add_argument("--sub-end", type=int, default=0, help="End index for subsampling time steps")
    parser.add_argument("--sub-extra", type=int, default=5, help="Number of extra random time steps to sample")
    parser.add_argument("--chuck-size", type=int, default=12, help="Chunk size for gradient computation (increased for better memory efficiency)")

    # LoRA parameters
    parser.add_argument("--lora-rank", type=int, default=4, help="Rank for LoRA layers")
    parser.add_argument("--lora-alpha", type=int, default=16, help="Alpha value for LoRA layers")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--image-size", type=int, default=256, help="Image size for the model")

    args = parser.parse_args()
    main(args)
