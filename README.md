# JEDI: The Force of Jensen-Shannon Divergence in Disentangling Diffusion Models

[![Project Website](https://img.shields.io/badge/Project-Website-green)](https://ericbill21.github.io/JEDI/) [![arXiv](https://img.shields.io/badge/arXiv-2505.19166-b31b1b.svg)](https://arxiv.org/abs/2505.19166)

><p align="center">

>[Eric Tillmann Bill](https://www.linkedin.com/in/ericbill21/), [Enis Simsar](https://enis.dev/), [Thomas Hofmann](https://da.inf.ethz.ch/people/ThomasHofmann)

></p>
>
> Text-to-image (T2I) models excel on single-entity prompts but struggle with multi-subject descriptions, often showing attribute leakage, identity entanglement, and subject omissions. We introduce the first theoretical framework with a principled, optimizable objective for steering sampling dynamics toward multi-subject fidelity. Viewing flow matching (FM) through stochastic optimal control (SOC), we formulate subject disentanglement as control over a trained FM sampler. This yields two architecture-agnostic algorithms: (i) a training-free test-time controller that perturbs the base velocity with a single-pass update, and (ii) Adjoint Matching, a lightweight fine-tuning rule that regresses a control network to a backward adjoint signal while preserving base-model capabilities. The same formulation unifies prior attention heuristics, extends to diffusion models via a flow–diffusion correspondence, and provides the first fine-tuning route explicitly designed for multi-subject fidelity. Empirically, on Stable Diffusion 3.5, FLUX, and Stable Diffusion XL, both algorithms consistently improve multi-subject alignment while maintaining base-model style. Test-time control runs efficiently on commodity GPUs, and fine-tuned controllers trained on limited prompts generalize to unseen ones. We further highlight <b>FOCUS</b> (Flow Optimal Control for Unentangled Subjects), which achieves state-of-the-art multi-subject fidelity across models.

<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <img src="static/images/base.png" width="300"/><br>
    <sub><b>Base Models</b></sub>
  </div>
  <div style="text-align: center;">
    <img src="static/images/focus.png" width="300"/><br>
    <sub><b>Base Models + FOCUS (Ours)</b></sub>
  </div>
</div>


## 📄 Citation

If you find our work useful, please consider citing our paper:

```
@misc{bill2025focus,
    title={Optimal Control Meets Flow Matching: A Principled Route to Multi-Subject Fidelity},
    author={Eric Tillmann Bill and Enis Simsar and Thomas Hofmann},
    year={2025},
    eprint={2510.02315},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```