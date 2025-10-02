import os
import argparse

import torch
from src.sd3 import SD3ControllerPipeline
from src.flux import FluxControllerPipeline

from src.controller import Controller
from peft import PeftModel

import yaml
import pandas as pd

def main(args):
    # Create experiment directory
    exp_idx = len([name for name in os.listdir(args.exp_dir) if os.path.isdir(os.path.join(args.exp_dir, name))])
    exp_dir = os.path.join(args.exp_dir, f"exp_{exp_idx}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save the configuration
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    
    # Load the model
    if args.model == "SD3":
        pipe = SD3ControllerPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", 
            torch_dtype=torch.float16
        )
        NUM_INFERENCE_STEPS = 28
        GUIDANCE_SCALE = 4.5
        MAX_SEQ_LENGTH = 77

    elif args.model == "FLUX":
        pipe = FluxControllerPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        NUM_INFERENCE_STEPS = 28
        GUIDANCE_SCALE = 3.5
        MAX_SEQ_LENGTH = 256

    if args.path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.path).merge_and_unload()

    # For weaker GPUs
    if args.save_memory == 0:
        # Everything to GPU
        pipe.to("cuda")
    if args.save_memory == 1:
        # Moves larges modules to CPU
        pipe.enable_model_cpu_offload()
    elif args.save_memory == 2:
        # Sequentially moves modules to CPU
        pipe.enable_sequential_cpu_offload()
    elif args.save_memory == 3:
        # Sequentially moves modules to CPU + gradient checkpointing
        pipe.enable_sequential_cpu_offload()

        from functools import partial
        pipe.transformer.enable_gradient_checkpointing(
            gradient_checkpointing_func=partial(torch.utils.checkpoint.checkpoint, 
            use_reentrant=False)
        )

    # Disable gradients for all parameters
    if getattr(pipe, "transformer", None):      pipe.transformer.requires_grad_(False)
    if getattr(pipe, "vae", None):              pipe.vae.requires_grad_(False)
    if getattr(pipe, "text_encoder", None):     pipe.text_encoder.requires_grad_(False)
    if getattr(pipe, "text_encoder_2", None):   pipe.text_encoder_2.requires_grad_(False)
    if getattr(pipe, "text_encoder_3", None):   pipe.text_encoder_3.requires_grad_(False)
    print("Model loaded...")

    # Load dataset
    with open(args.dataset, "r") as file:
        dataset = yaml.safe_load(file)
    
    # Set seed(s)
    if args.seed_range:
        SEEDS = list(range(args.seed_range[0], args.seed_range[1]))
    else:
        SEEDS = [args.seed]
    print(f"Sampling {len(dataset)} prompts with {len(SEEDS)} seeds each...")

    # Override default parameters
    NUM_INFERENCE_STEPS = args.num_steps if args.num_steps is not None else NUM_INFERENCE_STEPS
    GUIDANCE_SCALE = args.guidance_scale if args.guidance_scale is not None else GUIDANCE_SCALE
    print(f"Using {NUM_INFERENCE_STEPS} inference steps and guidance scale {GUIDANCE_SCALE}...")

    image_counter = 0
    output_paths = []
    for seed in SEEDS:
        for datum in dataset:
            print(f"Sampling image {image_counter} with seed {seed} and prompt: {datum['prompt']}")
            optim = Controller(
                t5_ids=datum["t5"],
                clip_ids=datum["clip"],
                lambda_scale=args.lambda_scale,
                method=args.method,
                model=args.model
            )

            image = pipe(
                datum["prompt"],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                max_sequence_length=MAX_SEQ_LENGTH,
                height=args.image_size,
                width=args.image_size,
                generator=torch.Generator("cpu").manual_seed(seed),
                controller=optim if args.lambda_scale != 0 else None,
            ).images[0]

            if args.t2i:
                image_path = os.path.join(exp_dir, f"{datum['prompt']}_{seed:06d}.png")
            else:
                image_path = os.path.join(exp_dir, f"image_{image_counter}.png")

            output_paths.append({
                "image_path" : image_path,
                "prompt": datum["prompt"],
            })
            image.save(image_path)
            image_counter += 1

    df = pd.DataFrame(output_paths)
    df.to_csv(os.path.join(exp_dir, "prompts.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from the model")

    # Conditioning parameters
    parser.add_argument("--exp-dir", type=str, default="images", help="Experiment directory")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset YAML file")

    parser.add_argument("--path", type=str, default=None, help="Path to the JEDI model weights")
    parser.add_argument("--image-size", type=int, default=512, help="Size of the generated images")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of diffusion steps, if not specified, use model default")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Guidance scale for classifier-free guidance")

    # Sampling parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--seed-range", type=int, nargs=2, help="Range of seeds for sampling")

    # JEDI parameters
    parser.add_argument("--lambda-scale", type=float, default=1, help="Lambda scaling factor for Controller, set to 0 to disable Controller")
    parser.add_argument("--method", type=str, choices=["focus", "conform", "attend_and_excite", "divide_and_bind", "jedi"], default="focus", help="Controller method to use")
    parser.add_argument("--model", type=str, choices=["SD3", "FLUX"], default="SD3", help="Base model to use")

    parser.add_argument("--save-memory", type=int, choices=[0, 1, 2, 3], default=0, help="Memory saving mode: 0 (none), 1 (offload), 2 (sequential offload), 3 (sequential + grad checkpointing)")
    parser.add_argument("--t2i", action="store_true", help="Use T2I output format")
    
    args = parser.parse_args()
    main(args)