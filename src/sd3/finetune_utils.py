from typing import Optional, Union, Dict, Any, Tuple, List, Callable

import torch
from torch import Tensor
import torch.autograd as autograd

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

from src.controller import Controller


class MemorylessFlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    def register_storage(self, storage: List[Tensor]) -> None:
        self.storage = storage

    def deregister_storage(self) -> None:
        self.storage = None

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:

        # <-- CODE FROM PARENT CLASS
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)
            dt = (per_token_sigmas - lower_sigmas)[..., None]
        else:
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
            dt = sigma_next - sigma

        # CODE FROM PARENT CLASS -->

        # Using notation from Adjoint Matching paper
        dtype = model_output.dtype 
        t = self.normalize_timesteps(timestep).to(torch.float32)
        h = -1.0 * dt.to(torch.float32)
        vt = -1.0 * model_output.to(torch.float32)

        noise = torch.randn(
            model_output.shape, device=model_output.device, generator=generator
        )
        prev_sample = (
            sample + h * (2 * vt - self.kappa(t) * sample)
            + torch.sqrt(h) * self.sigma(t) * noise
        ).to(dtype)

        if getattr(self, "storage", None) is not None:
            self.storage.append(prev_sample.to(dtype))

        # CODE FROM PARENT CLASS <--
        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
        # CODE FROM PARENT CLASS -->

    def alpha(self, t: Tensor) -> Tensor:
        return t

    def alpha_dot(self, t: Tensor) -> Tensor:
        return 1 * torch.ones_like(t)

    def beta(self, t: Tensor) -> Tensor:
        return 1 - t

    def beta_dot(self, t: Tensor) -> Tensor:
        return -1 * torch.ones_like(t)

    def sigma(self, t: Tensor, eps=1e-3) -> Tensor:
        return torch.sqrt(2 * (1 - t + eps) / (t + eps))

    def kappa(self, t: Tensor, eps=1e-3) -> Tensor:
        return (self.alpha_dot(t) + eps) / (self.alpha(t) + eps)

    def normalize_timesteps(self, t):
        return (self.config.num_train_timesteps - t) / self.config.num_train_timesteps

    def unnormalize_timesteps(self, t):
        return self.config.num_train_timesteps - t * self.config.num_train_timesteps


class LeanAdjoinSolver:
    def __init__(
        self,
        transformer: torch.nn.Module,
        scheduler: MemorylessFlowMatchScheduler,
        g: Callable[[Tensor], Tensor],
    ):
        self.transformer = transformer
        self.scheduler = scheduler
        self.g = g
    
    def sample_forward(
        self,
        x_traj: Tensor,
        prompt: Tensor,
        pooled: Tensor,
        f_traj: Optional[Tensor] = None,
    ) -> Tensor:
        """ Solve the adjoint ODE forward in time for a given trajectory."""
        K, B, *spatial = x_traj.shape
        device = self.transformer.device

        x_traj = x_traj.to(device)
        a_traj = torch.zeros_like(x_traj)
        if f_traj is not None:
            f_traj = f_traj.to(device)

        # timesteps and step sizes
        timesteps = self.scheduler.timesteps.to(device)
        ts_norm = self.scheduler.normalize_timesteps(timesteps)
        hs = -1.0 * (self.scheduler.sigmas[1:] - self.scheduler.sigmas[:-1]).to(device)

        # Since we are solving forward in time, we start with a0 = 0
        a_traj[0] = torch.zeros_like(x_traj[0])

        # expand prompts and timesteps for reuse
        expanded_prompt = prompt.expand(B, *prompt.shape[1:]).to(device)
        expanded_pooled = pooled.expand(B, *pooled.shape[1:]).to(device)
        expanded_ts = timesteps.view(-1, 1).expand(K - 1, B)

        for idx in range(K - 1):

            def drift_vjp(x: Tensor) -> Tensor:
                kappa = append_dims(self.scheduler.kappa(ts_norm[idx]), x_traj[idx].ndim)
                v_pred = -1.0 * self.transformer(
                            hidden_states=x,
                            timestep=expanded_ts[idx],
                            encoder_hidden_states=expanded_prompt,
                            pooled_projections=expanded_pooled,
                        ).sample
        
                return 2 * v_pred - kappa * x

            _, vjp_t = autograd.functional.vjp(
                func=drift_vjp,
                inputs=x_traj[idx],
                v=a_traj[idx],
                create_graph=False,
            )

            if f_traj is not None:
                vjp_t += f_traj[idx]

            a_traj[idx+1] = a_traj[idx] - hs[idx] * vjp_t

        return a_traj


    def sample(
        self,
        x_traj: Tensor,
        prompt: Tensor,
        pooled: Tensor,
        f_traj: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Solve the adjoint ODE backward in time for a given trajectory.

        Args:
            x_traj: Tensor of shape (T+1, B, C, H, W)
            prompt_embeds: Tensor of shape (B, D)
            pooled_prompt_embeds: Tensor of shape (B, D)
            f_traj: optional Tensor of running-cost gradients, shape (T, B, C, H, W)

        Returns:
            a_traj: Tensor of adjoint values, same shape as x_traj
        """
        K, B, *spatial = x_traj.shape
        device = self.transformer.device

        x_traj = x_traj.to(device)
        a_traj = torch.zeros_like(x_traj)
        if f_traj is not None:
            f_traj = f_traj.to(device)

        # timesteps and step sizes (Adding terminal timestep t=1.0)
        timesteps = torch.cat([self.scheduler.timesteps.to(device), torch.tensor([0.0], device=device)])
        ts_norm = self.scheduler.normalize_timesteps(timesteps)
        hs = -1.0 * (self.scheduler.sigmas[1:] - self.scheduler.sigmas[:-1]).to(device)

        # Terminal cost a1 = ∇_x g(x1)
        x1 = x_traj[-1].detach().requires_grad_(True)
        gx1 = self.g(x1)
        a_traj[-1] = torch.autograd.grad(
            outputs=gx1, inputs=x1, grad_outputs=torch.ones_like(gx1), create_graph=False
        )[0].detach()

        # expand prompts and timesteps for reuse
        expanded_prompt = prompt.expand(B, *prompt.shape[1:]).to(device)
        expanded_pooled = pooled.expand(B, *pooled.shape[1:]).to(device)
        expanded_ts = timesteps.view(-1, 1).expand(K, B)

        for idx in reversed(range(0, K - 1)):

            def drift_vjp(x: Tensor) -> Tensor:
                kappa = append_dims(self.scheduler.kappa(ts_norm[idx + 1]), x_traj[idx + 1].ndim)
                v_pred = (
                    -1.0
                    * self.transformer(
                        hidden_states=x,
                        timestep=expanded_ts[idx + 1],
                        encoder_hidden_states=expanded_prompt,
                        pooled_projections=expanded_pooled,
                    ).sample
                )

                return 2 * v_pred - kappa * x

            _, vjp_t = autograd.functional.vjp(
                func=drift_vjp,
                inputs=x_traj[idx + 1],
                v=a_traj[idx + 1],
                create_graph=False,
            )

            if f_traj is not None:
                vjp_t += f_traj[idx]

            a_traj[idx] = a_traj[idx + 1] + hs[idx] * vjp_t

        return a_traj


class MemorylessSDESolver:
    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.scheduler = scheduler

    def sample_memoryless_traj(
        self,
        x0: Tensor,
        prompt: Tensor,
        pooled: Tensor,
        num_steps: int = 28,
        generator: Optional[torch.Generator] = None,
        controller: Optional[Controller] = None,
        lambda_val: float = 1.0,
        calc_cost_gradient: bool = False,
    ) -> Tensor:
        """
        Args:
            x0: initial noise tensor, shape (B, C, H, W)
            prompt_embeds: text embeddings, shape (B, D)
            pooled_prompt_embeds: pooled text embeddings, shape (B, D)
            num_steps: number of discretization steps
            generator: optional torch.Generator for noise
            controller: optional Controller instance for running-cost storage
            lambda_reward: scaling for running cost
            calc_cost_gradient: whether to compute gradients of running cost

        Returns:
            - x_traj: tensor of shape (num_steps+1, B, C, H, W)
            - (optional) f_traj: per-step running-cost gradient if calc_running_reward
            - (optional) cum_reward: cumulative Controller reward if controller is provided
        """
        device = self.transformer.device

        # pre-move static inputs to device
        prompt = prompt.to(device)
        pooled = pooled.to(device)
        x_traj = torch.empty((num_steps + 1, *x0.shape), dtype=x0.dtype, device=device)

        # running cost storage
        cum_cost = torch.zeros(x0.shape[0], device=device)
        
        if calc_cost_gradient:
            f_traj = torch.zeros((num_steps, *x0.shape), dtype=x0.dtype, device=device)

        # setup scheduler
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps.to(dtype=x0.dtype, device=device)

        # sample trajectory
        x_traj[0] = x0.to(device)
        with torch.set_grad_enabled(calc_cost_gradient):
            for idx in range(num_steps):
                # detach to create leaf node
                xt = x_traj[idx].detach()
                xt.requires_grad_(calc_cost_gradient)

                if controller is not None:
                    controller.activate_storage()

                neg_vt = self.transformer(
                    hidden_states=xt,
                    timestep=timesteps[idx].expand(xt.size(0)),
                    encoder_hidden_states=prompt,
                    pooled_projections=pooled,
                ).sample

                if controller is not None:
                    cost = lambda_val * controller.compute_cost()
                    cum_cost += cost

                    if calc_cost_gradient:
                        torch.sum(cost).backward()
                        f_traj[idx] = xt.grad.detach()
                    controller.reset_storage()
                    controller.deactivate_storage()

                x_traj[idx + 1] = self.scheduler.step(
                    model_output=neg_vt.detach(),
                    timestep=self.scheduler.timesteps[idx],
                    sample=xt.requires_grad_(False),
                    generator=generator,
                ).prev_sample

        outputs = (x_traj,)
        if calc_cost_gradient:
            outputs += (f_traj, cum_cost)
        elif controller:
            outputs += (cum_cost,)

        return outputs